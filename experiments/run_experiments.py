"""
Step 6:
Running controlled experimental matrix for systematic RAG pipeline evaluation

This script runs all pipeline variants under systematically varied conditions
to isolate the effect of each component on final claim verification accuracy.

The experimental matrix is:

    Retrieval condition : BM25, dense, dense + stance reranking
    k (docs retrieved)  : 1, 5, 10
    Stance threshold    : loose (0.5), strict (0.8)
    Datasets            : SciFact, SciClaimHunt

The three k values are chosen deliberately:
    k=1  captures minimal retrieval -- one document, lowest noise
    k=5  captures moderate retrieval -- balanced signal and noise
    k=10 captures high retrieval -- most candidate documents, highest noise risk

The two thresholds test whether the strictness of stance filtering matters.
For BM25 and dense conditions the threshold is not applicable (recorded as N/A).

Output is a single JSON file with all conditions and a markdown summary table
suitable for direct inclusion in the thesis results chapter.

References:
    Wadden et al. (2020) -- SciFact dataset and baseline
    Stammbach & Neumann (2019) -- NLI filtering for scientific claims
"""

#importing os for directory creation and path handling
import os

#importing sys to add the project root to the import path
import sys

#adding the project root so all models and data imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing json for saving all results to disk
import json

#importing argparse so we can select the dataset from the command line
import argparse

#importing torch for device detection and model inference
import torch

#importing the tokenizer and model class for loading the fine-tuned RoBERTa
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#importing the retrieval classes for BM25 and dense retrieval
from models.retrieval import BM25Retriever, DenseRetriever

#importing the stance reranker for the dense + reranking condition
from models.reranker import StanceReranker

#importing the shared data loading functions for consistent data access
from data.utils import load_scifact, load_sciclaimhunt, LABEL_SUPPORT, LABEL_CONTRADICT, LABEL_NEI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#defining the three k values to evaluate across all retrieval conditions
K_VALUES = [1, 5, 10]

#defining the loose stance threshold for the reranking condition
LOOSE_THRESHOLD = 0.5

#defining the strict stance threshold for the reranking condition
STRICT_THRESHOLD = 0.8

#defining the rerank pool size -- always retrieve 10 before reranking down to k
RERANK_POOL_SIZE = 10

#defining the unified label to integer mapping for both datasets
LABEL_TO_ID = {
    LABEL_SUPPORT: 0,
    LABEL_CONTRADICT: 1,
    LABEL_NEI: 2,
}

# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def truncate_and_concatenate(claim_text, document_texts, tokenizer, max_total_length=512):
    #reserving 4 tokens for the special tokens the tokenizer will add
    special_tokens_count = 4

    #calculating total available token budget after accounting for special tokens
    available_tokens = max_total_length - special_tokens_count

    #tokenising the claim to find out how many tokens it uses
    claim_tokens = tokenizer.encode(claim_text, add_special_tokens=False)

    #calculating the remaining token budget for the documents after the claim
    document_token_budget = available_tokens - len(claim_tokens)

    #initialising an empty string to accumulate document text into
    concatenated_documents = ""

    #iterating over each retrieved document and adding it within the budget
    for document_text in document_texts:
        #tokenising this document to count its tokens
        document_tokens = tokenizer.encode(document_text, add_special_tokens=False)

        #truncating this document if it would exceed the remaining budget
        if len(document_tokens) > document_token_budget:
            document_tokens = document_tokens[:document_token_budget]

        #decoding the possibly-truncated tokens back to a readable string
        truncated_document = tokenizer.decode(document_tokens, skip_special_tokens=True)

        #adding this document's text to the growing concatenated string
        concatenated_documents += " " + truncated_document

        #reducing the budget by however many tokens we just used
        document_token_budget -= len(document_tokens)

        #stopping early if we have used up the full document token budget
        if document_token_budget <= 0:
            break

    #combining the claim and the document context with a [SEP] boundary marker
    combined_input = claim_text + " [SEP] " + concatenated_documents.strip()

    #returning the combined input string ready for the tokenizer
    return combined_input

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predicted_labels, true_labels, dataset_name):
    #importing sklearn for classification metrics
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

    #setting the class name list depending on which dataset we are evaluating
    if dataset_name == "scifact":
        target_names = ["SUPPORT", "CONTRADICT", "NEI"]
    else:
        target_names = ["Supported", "Refuted"]

    #computing macro F1 across all classes
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

    #computing macro precision across all classes
    macro_precision = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)

    #computing macro recall across all classes
    macro_recall = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)

    #printing the full per-class breakdown for visibility during the run
    print(classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=0))

    #printing the summary line for quick reading
    print(f"Macro F1: {macro_f1:.4f}  Precision: {macro_precision:.4f}  Recall: {macro_recall:.4f}")

    #returning the three metrics as a dictionary for saving to JSON
    return {
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
    }

# ---------------------------------------------------------------------------
# Single condition runner
# ---------------------------------------------------------------------------

def run_condition(condition_name, claims, labels, corpus, model, tokenizer, device, dataset_name, k, threshold=None):
    #printing a clear header for this experimental condition
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}  |  k={k}  |  threshold={threshold if threshold else 'N/A'}")
    print(f"{'='*60}")

    #initialising an empty list to collect predicted class indices
    predicted_labels = []

    #setting the model to evaluation mode so dropout layers are turned off
    model.eval()

    #handling the BM25 retrieval condition
    if condition_name == "bm25":
        #initialising the BM25 retriever with the full corpus
        bm25_retriever = BM25Retriever(corpus)

        #iterating over every claim to retrieve then classify
        for claim_text in claims:
            #retrieving the top-k document dicts using BM25 scoring
            retrieved_documents = bm25_retriever.retrieve(claim_text, k=k)

            #collecting the text of each retrieved document
            retrieved_document_texts = [doc["text"] for doc in retrieved_documents]

            #building the combined input from claim and retrieved docs
            combined_input_text = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)

            #tokenising the combined input with truncation and padding
            encoded_input = tokenizer(combined_input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)

            #moving tensors to the device
            encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

            #running the forward pass without gradient tracking
            with torch.no_grad():
                model_output = model(**encoded_input)

            #taking the argmax as the predicted class
            predicted_class = torch.argmax(model_output.logits, dim=1).item()

            #appending the prediction to the list
            predicted_labels.append(predicted_class)

    #handling the dense retrieval condition
    elif condition_name == "dense":
        #initialising the dense retriever with the full corpus
        dense_retriever = DenseRetriever(corpus)

        #iterating over every claim to retrieve then classify
        for claim_text in claims:
            #retrieving the top-k document dicts using dense similarity
            retrieved_documents = dense_retriever.retrieve(claim_text, k=k)

            #collecting the text of each retrieved document
            retrieved_document_texts = [doc["text"] for doc in retrieved_documents]

            #building the combined input from claim and retrieved docs
            combined_input_text = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)

            #tokenising the combined input with truncation and padding
            encoded_input = tokenizer(combined_input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)

            #moving tensors to the device
            encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

            #running the forward pass without gradient tracking
            with torch.no_grad():
                model_output = model(**encoded_input)

            #taking the argmax as the predicted class
            predicted_class = torch.argmax(model_output.logits, dim=1).item()

            #appending the prediction to the list
            predicted_labels.append(predicted_class)

    #handling the dense + stance reranking condition
    elif condition_name == "dense_reranked":
        #initialising the dense retriever with the full corpus
        dense_retriever = DenseRetriever(corpus)

        #initialising the stance reranker
        stance_reranker = StanceReranker()

        #iterating over every claim to retrieve, rerank, then classify
        for claim_text in claims:
            #retrieving a larger pool to give the reranker more to work with
            retrieved_documents = dense_retriever.retrieve(claim_text, k=RERANK_POOL_SIZE)

            #reranking the retrieved documents by stance score
            reranked_documents = stance_reranker.rerank(claim_text, retrieved_documents, neutral_threshold=threshold)

            #taking only the top-k documents after reranking
            top_reranked_document_texts = [doc["text"] for doc in reranked_documents[:k]]

            #building the combined input from claim and reranked docs
            combined_input_text = truncate_and_concatenate(claim_text, top_reranked_document_texts, tokenizer)

            #tokenising the combined input with truncation and padding
            encoded_input = tokenizer(combined_input_text, return_tensors="pt", truncation=True, max_length=512, padding=True)

            #moving tensors to the device
            encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

            #running the forward pass without gradient tracking
            with torch.no_grad():
                model_output = model(**encoded_input)

            #taking the argmax as the predicted class
            predicted_class = torch.argmax(model_output.logits, dim=1).item()

            #appending the prediction to the list
            predicted_labels.append(predicted_class)

    #computing and returning the metrics for this condition
    metrics = compute_metrics(predicted_labels, labels, dataset_name)

    #returning the metrics dictionary
    return metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    #setting up argparse for command line arguments
    parser = argparse.ArgumentParser(description="Running the controlled experimental matrix for Step 6")

    #adding the dataset argument
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scifact", "sciclaimhunt"],
        help="Selecting the dataset to run experiments on",
    )

    #adding the model path argument
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Providing the path to the fine-tuned RoBERTa checkpoint",
    )

    #adding the output path argument
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/step6_experiments.json",
        help="Specifying where to save the experiment results JSON",
    )

    #parsing all command line arguments
    parsed_arguments = parser.parse_args()

    #detecting the best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #printing which device will be used
    print(f"Using device: {device}")

    #loading the tokenizer from the fine-tuned RoBERTa checkpoint
    roberta_tokenizer = AutoTokenizer.from_pretrained(parsed_arguments.model_path)

    #loading the fine-tuned RoBERTa model from the checkpoint
    roberta_model = AutoModelForSequenceClassification.from_pretrained(parsed_arguments.model_path)

    #moving the model to the selected device
    roberta_model = roberta_model.to(device)

    #loading claims and corpus using the shared data utility functions
    if parsed_arguments.dataset == "scifact":
        #loading SciFact validation claims and corpus
        claims_data, corpus = load_scifact(split="validation")
    else:
        #loading SciClaimHunt val claims and corpus
        claims_data, corpus = load_sciclaimhunt(split="val")

    #extracting claim texts from the loaded claim dicts
    claims = [claim_dict["claim"] for claim_dict in claims_data]

    #converting string labels to integer ids using the unified label map
    labels = [LABEL_TO_ID[claim_dict["label"]] for claim_dict in claims_data]

    #printing dataset statistics for verification
    print(f"Loaded {len(claims)} claims and {len(corpus)} corpus documents")

    #initialising the results dictionary to hold all experimental conditions
    all_results = {
        "dataset": parsed_arguments.dataset,
        "conditions": []
    }

    #running BM25 conditions across all k values
    print("\n--- BM25 Retrieval Conditions ---")
    for k_value in K_VALUES:
        #running this BM25 condition and collecting metrics
        metrics = run_condition(
            condition_name="bm25",
            claims=claims,
            labels=labels,
            corpus=corpus,
            model=roberta_model,
            tokenizer=roberta_tokenizer,
            device=device,
            dataset_name=parsed_arguments.dataset,
            k=k_value,
            threshold=None,
        )

        #storing this condition result with its configuration
        all_results["conditions"].append({
            "condition": "bm25",
            "k": k_value,
            "threshold": None,
            "metrics": metrics,
        })

    #running dense retrieval conditions across all k values
    print("\n--- Dense Retrieval Conditions ---")
    for k_value in K_VALUES:
        #running this dense condition and collecting metrics
        metrics = run_condition(
            condition_name="dense",
            claims=claims,
            labels=labels,
            corpus=corpus,
            model=roberta_model,
            tokenizer=roberta_tokenizer,
            device=device,
            dataset_name=parsed_arguments.dataset,
            k=k_value,
            threshold=None,
        )

        #storing this condition result with its configuration
        all_results["conditions"].append({
            "condition": "dense",
            "k": k_value,
            "threshold": None,
            "metrics": metrics,
        })

    #running dense + reranking conditions across all k values and both thresholds
    print("\n--- Dense + Stance Reranking Conditions ---")
    for k_value in K_VALUES:
        for threshold_value in [LOOSE_THRESHOLD, STRICT_THRESHOLD]:
            #running this reranked condition and collecting metrics
            metrics = run_condition(
                condition_name="dense_reranked",
                claims=claims,
                labels=labels,
                corpus=corpus,
                model=roberta_model,
                tokenizer=roberta_tokenizer,
                device=device,
                dataset_name=parsed_arguments.dataset,
                k=k_value,
                threshold=threshold_value,
            )

            #labelling the threshold as loose or strict for clarity
            threshold_label = "loose" if threshold_value == LOOSE_THRESHOLD else "strict"

            #storing this condition result with its configuration
            all_results["conditions"].append({
                "condition": "dense_reranked",
                "k": k_value,
                "threshold": threshold_label,
                "metrics": metrics,
            })

    #creating the output directory if it does not already exist
    os.makedirs(os.path.dirname(parsed_arguments.output_path), exist_ok=True)

    #saving all experimental results to a JSON file
    with open(parsed_arguments.output_path, "w") as output_file:
        json.dump(all_results, output_file, indent=2)

    #printing confirmation that results were saved
    print(f"\nAll experiment results saved to {parsed_arguments.output_path}")

    #returning the results dictionary for use in Colab
    return all_results


#running main only when this script is called directly
if __name__ == "__main__":
    main()
"""
Failure taxonomy analysis for RAG claim verification

This script includes the implementation of the failure analysis component of the thesis.
It collects prediction errors from the dense retrieval pipeline on SciFact,
saves them with full context for manual annotation, and computes quantitative
failure rates across pipeline conditions for SciClaimHunt.

The four failure categories defined upfront (hypothesis-driven, not post-hoc):

    1. Irrelevant retrieval:
       Retrieved documents are topically unrelated to the claim.
       The model receives no useful evidence and is essentially guessing.

    2. Contradictory retrieval:
       Retrieved documents argue against the correct label.
       The model is actively misled by its own retrieved context.

    3. Evidence overload:
       Too many retrieved documents with conflicting signals dilute
       the correct evidence, confusing the model's prediction.

    4. Confident wrong prediction:
       The model predicts incorrectly despite reasonably relevant retrieval.
       The failure is in the classifier, not the retrieval component.


The stance reranker is specifically designed to address categories 1 and 2
by promoting stance-bearing documents over topically similar but neutral ones.
The failure analysis directly evaluates whether this design goal is achieved.


Scope decision: full manual annotation (50-75 errors) is performed on SciFact
only, as it is the primary dataset. SciClaimHunt failure rates are computed
quantitatively across conditions without full manual annotation, which is a
reasonable and honest scoping decision for a solo research project.
"""


#importing os for file path handling and directory creation
import os

#importing sys to add the project root to the import path
import sys

#adding the project root so all project imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing json for saving error records to disk
import json

#importing argparse for command line dataset selection
import argparse

#importing torch for device detection and inference
import torch

#importing the tokenizer and model class for the fine-tuned RoBERTa
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#importing the retrieval classes for BM25 and dense retrieval
from models.retrieval import BM25Retriever, DenseRetriever

#importing the stance reranker
from models.reranker import StanceReranker

#importing the shared data loading functions
from data.utils import load_scifact, load_sciclaimhunt, LABEL_SUPPORT, LABEL_CONTRADICT, LABEL_NEI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#defining the four failure category labels used for manual annotation
CATEGORY_IRRELEVANT = "irrelevant_retrieval"
CATEGORY_CONTRADICTORY = "contradictory_retrieval"
CATEGORY_OVERLOAD = "evidence_overload"
CATEGORY_CONFIDENT_WRONG = "confident_wrong_prediction"

#defining all valid category labels for validation
VALID_CATEGORIES = [
    CATEGORY_IRRELEVANT,
    CATEGORY_CONTRADICTORY,
    CATEGORY_OVERLOAD,
    CATEGORY_CONFIDENT_WRONG,
]

#defining the unified label to integer mapping
LABEL_TO_ID = {
    LABEL_SUPPORT: 0,
    LABEL_CONTRADICT: 1,
    LABEL_NEI: 2,
}

#defining the integer to label string mapping for display
ID_TO_LABEL = {0: LABEL_SUPPORT, 1: LABEL_CONTRADICT, 2: LABEL_NEI}

#defining the k value to use for failure collection
ANALYSIS_K = 5

#defining the rerank pool size for the reranked condition
RERANK_POOL_SIZE = 10

#defining the loose threshold for reranking
LOOSE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def truncate_and_concatenate(claim_text, document_texts, tokenizer, max_total_length=512):
    #reserving 4 tokens for the special tokens the tokenizer will add
    special_tokens_count = 4

    #calculating total available token budget
    available_tokens = max_total_length - special_tokens_count

    #tokenising the claim to count its tokens
    claim_tokens = tokenizer.encode(claim_text, add_special_tokens=False)

    #calculating the document token budget after the claim
    document_token_budget = available_tokens - len(claim_tokens)

    #initialising an empty string to accumulate document text
    concatenated_documents = ""

    #iterating over each document and adding within the budget
    for document_text in document_texts:
        #tokenising this document
        document_tokens = tokenizer.encode(document_text, add_special_tokens=False)

        #truncating if it exceeds the remaining budget
        if len(document_tokens) > document_token_budget:
            document_tokens = document_tokens[:document_token_budget]

        #decoding back to a string
        truncated_document = tokenizer.decode(document_tokens, skip_special_tokens=True)

        #adding to the concatenated string
        concatenated_documents += " " + truncated_document

        #reducing the remaining budget
        document_token_budget -= len(document_tokens)

        #stopping early if budget is exhausted
        if document_token_budget <= 0:
            break

    #combining claim and documents with SEP separator
    combined_input = claim_text + " [SEP] " + concatenated_documents.strip()

    #returning the combined string
    return combined_input


# ---------------------------------------------------------------------------
# Error collector
# ---------------------------------------------------------------------------

def collect_errors(condition_name, claims_data, labels, model, tokenizer, device,
                   bm25_retriever=None, dense_retriever=None, stance_reranker=None, k=ANALYSIS_K):
    #printing which condition we are collecting errors for
    print(f"\nCollecting errors for condition: {condition_name} (k={k})")

    #initialising an empty list to collect error records
    error_records = []

    #setting the model to evaluation mode
    model.eval()

    #iterating over every claim to run inference and check for errors
    for claim_index, claim_dict in enumerate(claims_data):
        #extracting the claim text
        claim_text = claim_dict["claim"]

        #getting the gold integer label
        gold_label_int = labels[claim_index]

        #getting the gold label string for display
        gold_label_string = ID_TO_LABEL[gold_label_int]

        #initialising retrieved documents as empty for no-retrieval condition
        retrieved_document_texts = []
        retrieved_document_dicts = []

        #handling no retrieval condition
        if condition_name == "no_retrieval":
            #encoding just the claim text with no retrieved context
            encoded_input = tokenizer(claim_text, return_tensors="pt", truncation=True, max_length=512, padding=True)

        #handling BM25 retrieval condition
        elif condition_name == "bm25":
            #retrieving top-k documents using BM25
            retrieved_document_dicts = bm25_retriever.retrieve(claim_text, k=k)
            retrieved_document_texts = [doc["text"] for doc in retrieved_document_dicts]
            combined_input = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)
            encoded_input = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=512, padding=True)

        #handling dense retrieval condition
        elif condition_name == "dense":
            #retrieving top-k documents using dense similarity
            retrieved_document_dicts = dense_retriever.retrieve(claim_text, k=k)
            retrieved_document_texts = [doc["text"] for doc in retrieved_document_dicts]
            combined_input = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)
            encoded_input = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=512, padding=True)

        #handling dense + reranking condition
        elif condition_name == "dense_reranked":
            #retrieving a larger pool then reranking
            retrieved_document_dicts = dense_retriever.retrieve(claim_text, k=RERANK_POOL_SIZE)
            reranked_dicts = stance_reranker.rerank(claim_text, retrieved_document_dicts, neutral_threshold=LOOSE_THRESHOLD)
            top_reranked = reranked_dicts[:k]
            retrieved_document_texts = [doc["text"] for doc in top_reranked]
            combined_input = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)
            encoded_input = tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=512, padding=True)

        #moving tensors to the device
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        #running the forward pass without gradient tracking
        with torch.no_grad():
            model_output = model(**encoded_input)

        #getting the predicted class
        predicted_class_int = torch.argmax(model_output.logits, dim=1).item()
        predicted_label_string = ID_TO_LABEL[predicted_class_int]

        #checking if this prediction is an error
        if predicted_class_int != gold_label_int:
            #building the error record with full context for manual annotation
            error_record = {
                "error_id": len(error_records) + 1,
                "claim_id": claim_dict.get("id", str(claim_index)),
                "claim_text": claim_text,
                "gold_label": gold_label_string,
                "predicted_label": predicted_label_string,
                "retrieved_documents": retrieved_document_texts,
                "condition": condition_name,
                "k": k,
                "failure_category": None,
                "annotation_notes": "",
            }

            #appending this error to the error records list
            error_records.append(error_record)

    #printing how many errors were found
    print(f"Found {len(error_records)} errors out of {len(claims_data)} claims ({len(error_records)/len(claims_data)*100:.1f}%)")

    #returning the full list of error records
    return error_records


# ---------------------------------------------------------------------------
# Quantitative failure rate analysis
# ---------------------------------------------------------------------------

def compute_failure_rates(all_condition_errors):
    #initialising a dictionary to hold failure rates per condition
    failure_rates = {}

    #iterating over each condition and computing its error rate
    for condition_name, error_records in all_condition_errors.items():
        #counting total claims from the error records
        total_errors = len(error_records)

        #counting how many errors fall into each category
        category_counts = {category: 0 for category in VALID_CATEGORIES}
        annotated_count = 0

        #iterating over error records to count annotated categories
        for error in error_records:
            if error.get("failure_category") in VALID_CATEGORIES:
                category_counts[error["failure_category"]] += 1
                annotated_count += 1

        #storing the failure rate information for this condition
        failure_rates[condition_name] = {
            "total_errors": total_errors,
            "annotated_errors": annotated_count,
            "category_counts": category_counts,
        }

        #computing category percentages if there are annotated errors
        if annotated_count > 0:
            failure_rates[condition_name]["category_percentages"] = {
                category: round(count / annotated_count * 100, 1)
                for category, count in category_counts.items()
            }

    #returning the failure rates dictionary
    return failure_rates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    #setting up argparse for command line arguments
    parser = argparse.ArgumentParser(description="Running failure taxonomy analysis for Step 7")

    #adding the dataset argument
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scifact", "sciclaimhunt"],
        help="Selecting the dataset to analyse failures on",
    )

    #adding the model path argument
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Providing the path to the fine-tuned RoBERTa checkpoint",
    )

    #adding the output directory argument
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Specifying the directory to save failure analysis outputs",
    )

    #parsing command line arguments
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

    #loading claims and corpus using the shared data utility
    if parsed_arguments.dataset == "scifact":
        #loading SciFact validation claims and corpus
        claims_data, corpus = load_scifact(split="validation")
    else:
        #loading SciClaimHunt val claims and corpus
        claims_data, corpus = load_sciclaimhunt(split="val")

    #converting string labels to integer ids
    labels = [LABEL_TO_ID[claim_dict["label"]] for claim_dict in claims_data]

    #printing dataset statistics
    print(f"Loaded {len(claims_data)} claims and {len(corpus)} corpus documents")

    #building all retrievers once to avoid redundant corpus encoding
    print("\nBuilding BM25 index (happens once)...")
    bm25_retriever = BM25Retriever(corpus)

    #building the dense retriever once
    print("\nBuilding dense retriever and encoding corpus (happens once)...")
    dense_retriever = DenseRetriever(corpus)

    #loading the stance reranker once
    print("\nLoading stance reranker (happens once)...")
    stance_reranker = StanceReranker()

    #collecting errors for all four conditions
    all_condition_errors = {}

    #collecting no retrieval errors
    all_condition_errors["no_retrieval"] = collect_errors(
        "no_retrieval", claims_data, labels, roberta_model, roberta_tokenizer, device
    )

    #collecting BM25 errors
    all_condition_errors["bm25"] = collect_errors(
        "bm25", claims_data, labels, roberta_model, roberta_tokenizer, device,
        bm25_retriever=bm25_retriever
    )

    #collecting dense errors
    all_condition_errors["dense"] = collect_errors(
        "dense", claims_data, labels, roberta_model, roberta_tokenizer, device,
        dense_retriever=dense_retriever
    )

    #collecting dense + reranking errors
    all_condition_errors["dense_reranked"] = collect_errors(
        "dense_reranked", claims_data, labels, roberta_model, roberta_tokenizer, device,
        dense_retriever=dense_retriever, stance_reranker=stance_reranker
    )

    #creating the output directory if it does not exist
    output_dir = parsed_arguments.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    #saving error records for each condition to separate JSON files for manual annotation
    for condition_name, error_records in all_condition_errors.items():
        #building the output file path for this condition
        output_filename = f"step7_errors_{parsed_arguments.dataset}_{condition_name}.json"
        output_path = os.path.join(output_dir, output_filename)

        #saving the error records to JSON
        with open(output_path, "w") as output_file:
            json.dump(error_records, output_file, indent=2)

        #printing confirmation
        print(f"Saved {len(error_records)} errors to {output_path}")

    #computing quantitative failure rates across all conditions
    failure_rates = compute_failure_rates(all_condition_errors)

    #saving the failure rates summary
    failure_rates_path = os.path.join(output_dir, f"step7_failure_rates_{parsed_arguments.dataset}.json")
    with open(failure_rates_path, "w") as output_file:
        json.dump(failure_rates, output_file, indent=2)

    #printing the failure rates summary
    print(f"\nFailure rates summary saved to {failure_rates_path}")
    print("\nError counts per condition:")
    for condition_name, rates in failure_rates.items():
        print(f"  {condition_name:20s}  {rates['total_errors']} errors")

    #returning the full results for Colab use
    return {"errors": all_condition_errors, "failure_rates": failure_rates}

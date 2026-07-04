"""
full RAG pipeline evaluation for scientific claim verification


This script connects all the components built in previous steps:
retrieval (BM25 and dense), stance-aware reranking, and the fine-tuned
RoBERTa classifier. We run four pipeline conditions and compare them.


The four conditions are:
1. No retrieval -- RoBERTa only, no context given
2. BM25 + RoBERTa -- sparse retrieval then classify
3. Dense + RoBERTa -- dense retrieval then classify
4. Dense + stance reranking + RoBERTa -- retrieve, rerank, then classify


The key question this step answers: does giving the model retrieved
evidence actually help it verify claims better than no context at all?
And does reranking that evidence by stance make it even better?


Design decisions (document these in your thesis):
- top_k=5 for classifier input: keeps the input under 512 tokens
- rerank_top_k=10: retrieve a bigger pool first then rerank down to 5
- truncate_and_concatenate: fair token budget split between claim and docs
- claim and evidence passed as a text PAIR to the tokenizer, so RoBERTa inserts its
  own correct segment boundary (</s></s>) rather than a hand-built separator
- all four pipelines use the same fine-tuned checkpoint for fair comparison
"""

#importing os for file path handling and making directories
import os

#importing json for saving results to disk after each run
import json

#importing argparse so we can select the dataset from the command line
import argparse

#importing torch for running inference on GPU or CPU
import torch

#importing the tokenizer and model classes from transformers for RoBERTa
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#importing BM25Retriever and DenseRetriever from our retrieval module
from models.retrieval import BM25Retriever, DenseRetriever

#importing StanceReranker from our reranking module
from models.reranker import StanceReranker

#importing the shared data loading functions so we use one consistent loading path
from data.utils import load_scifact, load_scifact_open, LABEL_SUPPORT, LABEL_CONTRADICT, LABEL_NEI

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

#defining how many documents to retrieve for reranking before cutting to top_k
RERANK_POOL_SIZE = 10

#defining the unified label to integer mapping used by both datasets
LABEL_TO_ID = {
    LABEL_SUPPORT: 0,
    LABEL_CONTRADICT: 1,
    LABEL_NEI: 2,
}

#defining the integer to label string mapping for SciFact display
SCIFACT_INT_TO_LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}

# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def truncate_and_concatenate(claim_text, document_texts, tokenizer, max_total_length=512):
    """
    Building the claim and the concatenated evidence as TWO separate strings, so the
    tokenizer can join them as a proper text pair (claim, evidence). RoBERTa then inserts
    its correct segment boundary (</s></s>) itself -- we do not hand-build a separator.

    Returns: (claim_text, evidence_text) tuple, ready for tokenizer(claim, evidence, ...)
    """
    #reserving tokens for the special tokens the tokenizer adds around a pair
    #RoBERTa pair format is <s> claim </s></s> evidence </s> which is 4 special tokens
    #asking the tokenizer how many special tokens it adds to a pair, rather than hardcoding as 4
    special_tokens_count = tokenizer.num_special_tokens_to_add(pair=True)
    available_tokens = max_total_length - special_tokens_count

    #tokenising the claim to measure how many tokens it uses
    claim_tokens = tokenizer.encode(claim_text, add_special_tokens=False)

    #the evidence gets whatever budget remains after the claim
    document_token_budget = available_tokens - len(claim_tokens)

    #accumulating evidence text within the remaining budget
    concatenated_documents = ""
    for document_text in document_texts:
        document_tokens = tokenizer.encode(document_text, add_special_tokens=False)
        if len(document_tokens) > document_token_budget:
            document_tokens = document_tokens[:document_token_budget]
        truncated_document = tokenizer.decode(document_tokens, skip_special_tokens=True)
        concatenated_documents += " " + truncated_document
        document_token_budget -= len(document_tokens)
        if document_token_budget <= 0:
            break

    #returning claim and evidence as a PAIR -- the tokenizer will join them correctly
    return claim_text, concatenated_documents.strip()


# ---------------------------------------------------------------------------
# Pipeline conditions
# ---------------------------------------------------------------------------

def run_no_retrieval_pipeline(claims, labels, model, tokenizer, device, dataset_name):
    #printing which pipeline condition we are now running
    print("\nRunning pipeline: No Retrieval (RoBERTa only)")

    #initialising an empty list to collect predicted class indices
    predicted_labels = []

    #setting the model to evaluation mode so dropout layers are turned off
    model.eval()

    #iterating over every claim and classifying it without any retrieved context
    for claim_text in claims:
        #tokenising just the claim text with truncation and padding enabled
        encoded_input = tokenizer(
            claim_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        #moving all input tensors to the device the model is on
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        #running the forward pass through the model without tracking gradients
        with torch.no_grad():
            model_output = model(**encoded_input)

        #taking the class with the highest logit as the prediction
        predicted_class = torch.argmax(model_output.logits, dim=1).item()

        #appending this prediction to the predictions list
        predicted_labels.append(predicted_class)

    #computing classification metrics and printing the results
    metrics = compute_metrics(predicted_labels, labels, dataset_name)

    #returning the metrics dictionary for saving later
    return metrics


def run_bm25_pipeline(claims, labels, corpus, model, tokenizer, device, dataset_name, top_k=5):
    #printing which pipeline condition we are now running
    print("\nRunning pipeline: BM25 + RoBERTa")

    #initialising the BM25 retriever with the full corpus
    bm25_retriever = BM25Retriever(corpus)

    #initialising an empty list to collect predicted class indices
    predicted_labels = []

    #setting the model to evaluation mode so dropout layers are turned off
    model.eval()

    #iterating over every claim, retrieving with BM25, then classifying
    for claim_text in claims:
        retrieved_documents = bm25_retriever.retrieve(claim_text, k=top_k)

        #collecting the actual text of each retrieved document from the retriever output
        retrieved_document_texts = [doc["text"] for doc in retrieved_documents]

        #building the combined input string from the claim and retrieved docs
        claim_part, evidence_part = truncate_and_concatenate(claim_text, top_reranked_document_texts, tokenizer)

        #tokenising the combined input with truncation and padding enabled
        encoded_input = tokenizer(
            claim_part,
            #passing evidence as the second segment for proper pair
            evidence_part,          
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        #moving all input tensors to the device the model is on
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        #running the forward pass through the model without tracking gradients
        with torch.no_grad():
            model_output = model(**encoded_input)

        #taking the class with the highest logit as the prediction
        predicted_class = torch.argmax(model_output.logits, dim=1).item()

        #appending this prediction to the predictions list
        predicted_labels.append(predicted_class)

    #computing classification metrics and printing the results
    metrics = compute_metrics(predicted_labels, labels, dataset_name)

    #returning the metrics dictionary for saving later
    return metrics

#passing in the dense retriever as an argument to avoid re-encoding the corpus every time
def run_dense_pipeline(claims, labels, dense_retriever, model, tokenizer, device, dataset_name, top_k=5):
    #printing which pipeline condition we are now running
    print("\nRunning pipeline: Dense + RoBERTa")

    #initialising an empty list to collect predicted class indices
    predicted_labels = []

    #setting the model to evaluation mode so dropout layers are turned off
    model.eval()

    #iterating over every claim, retrieving with dense similarity, then classifying
    for claim_text in claims:
        #retrieving the top-k document dictionaries using dense embedding similarity
        retrieved_documents = dense_retriever.retrieve(claim_text, k=top_k)

        #collecting the actual text of each retrieved document from the corpus
        retrieved_document_texts = [doc["text"] for doc in retrieved_documents]

        #building the combined input string from the claim and retrieved docs
        claim_part, evidence_part = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)

        #tokenising the combined input with truncation and padding enabled
        encoded_input = tokenizer(
            claim_part,
            #passing evidence as the second segment for proper pair
            evidence_part,          
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        #moving all input tensors to the device the model is on
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        #running the forward pass through the model without tracking gradients
        with torch.no_grad():
            model_output = model(**encoded_input)

        #taking the class with the highest logit as the prediction
        predicted_class = torch.argmax(model_output.logits, dim=1).item()

        #appending this prediction to the predictions list
        predicted_labels.append(predicted_class)

    #computing classification metrics and printing the results
    metrics = compute_metrics(predicted_labels, labels, dataset_name)

    #returning the metrics dictionary for saving later
    return metrics

#passing in the dense retriever as an argument to avoid re-encoding the corpus every time
def run_dense_reranked_pipeline(claims, labels, dense_retriever, stance_reranker, model, tokenizer, device, dataset_name, top_k=5):
    #printing which pipeline condition we are now running
    print("\nRunning pipeline: Dense + Stance Reranking + RoBERTa")

    #initialising an empty list to collect predicted class indices
    predicted_labels = []

    #setting the model to evaluation mode so dropout layers are turned off
    model.eval()

    #iterating over every claim, retrieving a bigger pool, reranking, then classifying
    for claim_text in claims:
        #retrieving a larger pool of documents to give the reranker more to work with
        retrieved_documents = dense_retriever.retrieve(claim_text, k=RERANK_POOL_SIZE)

        #reranking all retrieved documents by their stance score using the NLI model
        reranked_documents = stance_reranker.rerank(claim_text, retrieved_documents, neutral_threshold=0.5)

        #taking only the top-k documents after reranking for the classifier input
        #to match the reranker.py as it returns a list of dicts with keys
        top_reranked_document_texts = [doc["text"] for doc in reranked_documents[:top_k]]

        #building the combined input string from the claim and reranked docs
        claim_part, evidence_part = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)

        #tokenising the combined input with truncation and padding enabled
        encoded_input = tokenizer(
            claim_part,
            #passing evidence as the second segment for proper pair
            evidence_part,          
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        #moving all input tensors to the device the model is on
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        #running the forward pass through the model without tracking gradients
        with torch.no_grad():
            model_output = model(**encoded_input)

        #taking the class with the highest logit as the prediction
        predicted_class = torch.argmax(model_output.logits, dim=1).item()

        #appending this prediction to the predictions list
        predicted_labels.append(predicted_class)

    #computing classification metrics and printing the results
    metrics = compute_metrics(predicted_labels, labels, dataset_name)

    #returning the metrics dictionary for saving later
    return metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predicted_labels, true_labels, dataset_name):
    #importing sklearn classification tools for the evaluation metrics
    from sklearn.metrics import classification_report, f1_score

    #both datasets use the same 3-class scheme (SUPPORT/CONTRADICT/NEI); SciFact-Open
    #simply has no NEI gold labels, which is handled by scoping to present labels
    id_to_name = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}

    #determining which label ids actually appear in the gold labels
    present_ids = sorted(set(true_labels))
    present_names = [id_to_name[i] for i in present_ids]

    #computing macro F1 over only the present classes (consistent with the baseline)
    macro_f1_score = f1_score(
        true_labels, predicted_labels,
        labels=present_ids, average="macro", zero_division=0,
    )

    print(classification_report(
        true_labels, predicted_labels,
        labels=present_ids, target_names=present_names, zero_division=0,
    ))

    #printing the overall macro F1 score for easy reading
    print(f"Macro F1: {macro_f1_score:.4f}")

    from sklearn.metrics import precision_score, recall_score
    macro_precision = precision_score(true_labels, predicted_labels,
                                      labels=present_ids, average="macro", zero_division=0)
    macro_recall = recall_score(true_labels, predicted_labels,
                                labels=present_ids, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1_score:.4f}  Precision: {macro_precision:.4f}  Recall: {macro_recall:.4f}")
    return {
        "macro_f1": macro_f1_score,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    #setting up argparse so we can choose the dataset from the command line
    parser = argparse.ArgumentParser(description="Running the full RAG pipeline evaluation")

    #adding the dataset argument to pick between scifact and scifact_open
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scifact", "scifact_open"],
        help="Selecting the dataset to run the pipeline evaluation on",
    )

    #adding the model path argument pointing to the fine-tuned RoBERTa checkpoint
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Providing the path to the saved fine-tuned RoBERTa model",
    )

    #adding the output path argument for saving the results JSON file
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/step5_pipeline_results.json",
        help="Specifying where to save the pipeline results JSON",
    )

    #adding the top-k argument to control how many documents go into the classifier
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Setting how many retrieved documents to pass to the classifier",
    )

    #parsing all the command line arguments
    parsed_arguments = parser.parse_args()

    #detecting whether a GPU is available and setting the device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #printing which device will be used for all inference
    print(f"Using device: {device}")

    #loading the tokenizer from the fine-tuned RoBERTa checkpoint
    roberta_tokenizer = AutoTokenizer.from_pretrained(parsed_arguments.model_path)

    #loading the fine-tuned RoBERTa model from the checkpoint
    roberta_model = AutoModelForSequenceClassification.from_pretrained(parsed_arguments.model_path)

    #moving the model to the selected device
    roberta_model = roberta_model.to(device)

    #loading claims and corpus using the shared data utils to keep one consistent loading path
    if parsed_arguments.dataset == "scifact":
        claims_data, corpus = load_scifact(split="validation")
    elif parsed_arguments.dataset == "scifact_open":
        claims_data, corpus = load_scifact_open(corpus_file="full")
    else:
        raise ValueError(f"Unknown dataset: {parsed_arguments.dataset}")

    #extracting just the claim texts from the claims dicts
    claims = [claim_dict["claim"] for claim_dict in claims_data]

    #converting the string labels to integer ids using the unified label map
    labels = [LABEL_TO_ID[claim_dict["label"]] for claim_dict in claims_data]

    #printing how many claims were loaded so we can verify it looks right
    print(f"Loaded {len(claims)} claims for evaluation")

    #printing how many corpus documents were loaded
    print(f"Loaded corpus with {len(corpus)} documents")

    #building the dense retriever ONCE (encodes the corpus a single time) and the
    #stance reranker ONCE, then reusing them across pipelines -- avoids re-encoding
    #the 500K corpus multiple times
    print("\nBuilding dense retriever (one-time corpus encoding)...")
    shared_dense_retriever = DenseRetriever(corpus)
    shared_stance_reranker = StanceReranker(device=device)

    #initialising the results dictionary to hold all four pipeline outputs
    all_pipeline_results = {}

    #running pipeline 1: no retrieval, RoBERTa only
    no_retrieval_metrics = run_no_retrieval_pipeline(
        claims, labels, roberta_model, roberta_tokenizer, device, parsed_arguments.dataset
    )
    all_pipeline_results["no_retrieval"] = no_retrieval_metrics

    #running pipeline 2: BM25 retrieval then RoBERTa
    bm25_metrics = run_bm25_pipeline(
        claims, labels, corpus, roberta_model, roberta_tokenizer, device,
        parsed_arguments.dataset, top_k=parsed_arguments.top_k
    )
    all_pipeline_results["bm25_roberta"] = bm25_metrics

    #running pipeline 3: dense retrieval then RoBERTa
    dense_metrics = run_dense_pipeline(
        claims, labels, shared_dense_retriever, roberta_model, roberta_tokenizer, device,
        parsed_arguments.dataset, top_k=parsed_arguments.top_k
    )
    all_pipeline_results["dense_roberta"] = dense_metrics

    #running pipeline 4: dense retrieval then stance reranking then RoBERTa
    dense_reranked_metrics = run_dense_reranked_pipeline(
        claims, labels, shared_dense_retriever, shared_stance_reranker,
        roberta_model, roberta_tokenizer, device,
        parsed_arguments.dataset, top_k=parsed_arguments.top_k
    )
    all_pipeline_results["dense_reranked_roberta"] = dense_reranked_metrics

    #creating the results directory if it does not already exist
    os.makedirs(os.path.dirname(parsed_arguments.output_path), exist_ok=True)

    #saving all four pipeline results to a JSON file at the output path
    with open(parsed_arguments.output_path, "w") as output_file:
        json.dump(all_pipeline_results, output_file, indent=2)

    #printing a confirmation message so we know the file was saved
    print(f"\nAll pipeline results saved to {parsed_arguments.output_path}")

    #returning the results dictionary so Colab can capture it directly
    return all_pipeline_results


#running main only when this script is called directly, not when imported
if __name__ == "__main__":
    main()
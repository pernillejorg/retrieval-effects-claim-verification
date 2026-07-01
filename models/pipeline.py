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
- [SEP] separator between claim and evidence: signals the boundary clearly
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


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

#defining the label string to integer mapping for SciFact
SCIFACT_LABEL_MAP = {
    "SUPPORT": 0,
    "CONTRADICT": 1,
    "NEI": 2,
    "": 2,
}

#defining the label string to integer mapping for SciClaimHunt
SCICLAIMHUNT_LABEL_MAP = {
    "Supported": 0,
    "Refuted": 1,
    "NEI": 2,
}

#defining the integer to label string mapping for SciFact
SCIFACT_INT_TO_LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}

#defining the integer to label string mapping for SciClaimHunt
SCICLAIMHUNT_INT_TO_LABEL = {0: "Supported", 1: "Refuted", 2: "NEI"}

#defining how many documents to retrieve for reranking before cutting to top_k
RERANK_POOL_SIZE = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scifact_data(scifact_data_path):
    #importing datasets here to keep the top-level imports clean
    from datasets import load_dataset

    #loading the SciFact claims config from the local cache directory
    scifact_dataset = load_dataset(
        "allenai/scifact",
        "claims",
        cache_dir=scifact_data_path,
    )

    #initialising empty lists for collecting claims and integer labels
    claims = []
    labels = []

    #iterating over the test split to collect claims and their gold labels
    for row in scifact_dataset["test"]:
        #extracting the claim text from this row
        claim_text = row["claim"]

        #mapping empty evidence label string to NEI using the label map
        label_string = row["evidence_label"] if row["evidence_label"] != "" else "NEI"

        #appending the claim text and integer label to their lists
        claims.append(claim_text)
        labels.append(SCIFACT_LABEL_MAP[label_string])

    #returning the collected claims and labels for evaluation
    return claims, labels


def load_sciclaimhunt_data(sciclaimhunt_data_path):
    #importing datasets here to keep the top-level imports clean
    from datasets import load_dataset

    #loading the SciClaimHunt dataset from the local cache directory
    sciclaimhunt_dataset = load_dataset(
        "Skatinger/SciClaimHunt",
        cache_dir=sciclaimhunt_data_path,
    )

    #initialising empty lists for collecting claims and integer labels
    claims = []
    labels = []

    #iterating over the test split to collect claims and their gold labels
    for row in sciclaimhunt_dataset["test"]:
        #skipping any rows that have no claim text
        if not row["Claim"]:
            continue

        #extracting the claim text and label string from this row
        claim_text = row["Claim"]
        label_string = row["Type"]

        #appending the claim text and integer label to their lists
        claims.append(claim_text)
        labels.append(SCICLAIMHUNT_LABEL_MAP[label_string])

    #returning the collected claims and labels for evaluation
    return claims, labels


def load_corpus(dataset_name, scifact_data_path, sciclaimhunt_data_path):
    #importing datasets here to keep the top-level imports clean
    from datasets import load_dataset

    #initialising an empty dictionary to map document ids to their text
    corpus = {}

    #loading the SciFact corpus if the selected dataset is scifact
    if dataset_name == "scifact":
        #loading the corpus config of SciFact from the local cache
        scifact_corpus_dataset = load_dataset(
            "allenai/scifact",
            "corpus",
            cache_dir=scifact_data_path,
        )

        #iterating over all corpus documents and building the id to text mapping
        for row in scifact_corpus_dataset["train"]:
            #joining the list of abstract sentences into one string
            abstract_text = " ".join(row["abstract"])

            #storing the abstract text keyed by its doc_id string
            corpus[str(row["doc_id"])] = abstract_text

    #loading the SciClaimHunt corpus if the selected dataset is sciclaimhunt
    elif dataset_name == "sciclaimhunt":
        #loading the SciClaimHunt dataset from the local cache directory
        sciclaimhunt_dataset = load_dataset(
            "Skatinger/SciClaimHunt",
            cache_dir=sciclaimhunt_data_path,
        )

        #iterating over the train split to build a corpus from evidence fields
        for index, row in enumerate(sciclaimhunt_dataset["train"]):
            #skipping rows that have no claim text to avoid empty corpus entries
            if not row["Claim"]:
                continue

            #storing the evidence text keyed by its index as a string
            corpus[str(index)] = row["Evidence"]

    #returning the filled corpus dictionary
    return corpus


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
        #retrieving the top-k document ids using BM25 keyword scoring
        retrieved_document_ids = bm25_retriever.retrieve(claim_text, top_k=top_k)

        #collecting the actual text of each retrieved document from the corpus
        retrieved_document_texts = [corpus[doc_id] for doc_id in retrieved_document_ids if doc_id in corpus]

        #building the combined input string from the claim and retrieved docs
        combined_input_text = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)

        #tokenising the combined input with truncation and padding enabled
        encoded_input = tokenizer(
            combined_input_text,
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


def run_dense_pipeline(claims, labels, corpus, model, tokenizer, device, dataset_name, top_k=5):
    #printing which pipeline condition we are now running
    print("\nRunning pipeline: Dense + RoBERTa")

    #initialising the dense retriever with the full corpus
    dense_retriever = DenseRetriever(corpus)

    #building the dense FAISS index over all corpus documents
    dense_retriever.build_index()

    #initialising an empty list to collect predicted class indices
    predicted_labels = []

    #setting the model to evaluation mode so dropout layers are turned off
    model.eval()

    #iterating over every claim, retrieving with dense similarity, then classifying
    for claim_text in claims:
        #retrieving the top-k document ids using dense embedding similarity
        retrieved_document_ids = dense_retriever.retrieve(claim_text, top_k=top_k)

        #collecting the actual text of each retrieved document from the corpus
        retrieved_document_texts = [corpus[doc_id] for doc_id in retrieved_document_ids if doc_id in corpus]

        #building the combined input string from the claim and retrieved docs
        combined_input_text = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)

        #tokenising the combined input with truncation and padding enabled
        encoded_input = tokenizer(
            combined_input_text,
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


def run_dense_reranked_pipeline(claims, labels, corpus, model, tokenizer, device, dataset_name, top_k=5):
    #printing which pipeline condition we are now running
    print("\nRunning pipeline: Dense + Stance Reranking + RoBERTa")

    #initialising the dense retriever with the full corpus
    dense_retriever = DenseRetriever(corpus)

    #building the dense FAISS index over all corpus documents
    dense_retriever.build_index()

    #initialising the stance reranker which uses the NLI model internally
    stance_reranker = StanceReranker()

    #initialising an empty list to collect predicted class indices
    predicted_labels = []

    #setting the model to evaluation mode so dropout layers are turned off
    model.eval()

    #iterating over every claim, retrieving a bigger pool, reranking, then classifying
    for claim_text in claims:
        #retrieving a larger pool of documents to give the reranker more to work with
        retrieved_document_ids = dense_retriever.retrieve(claim_text, top_k=RERANK_POOL_SIZE)

        #collecting the actual text of each retrieved document from the corpus
        retrieved_document_texts = [corpus[doc_id] for doc_id in retrieved_document_ids if doc_id in corpus]

        #reranking all retrieved documents by their stance score using the NLI model
        reranked_documents = stance_reranker.rerank(claim_text, retrieved_document_texts)

        #taking only the top-k documents after reranking for the classifier input
        #to match the reranker.py as it returns a list of dicts with keys
        top_reranked_document_texts = [doc["text"] for doc in reranked_documents[:top_k]]

        #building the combined input string from the claim and reranked docs
        combined_input_text = truncate_and_concatenate(claim_text, top_reranked_document_texts, tokenizer)

        #tokenising the combined input with truncation and padding enabled
        encoded_input = tokenizer(
            combined_input_text,
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

    #setting the class name list depending on which dataset we are evaluating
    if dataset_name == "scifact":
        target_names = ["SUPPORT", "CONTRADICT", "NEI"]
    else:
        target_names = ["Supported", "Refuted", "NEI"]

    #computing the macro F1 score across all three classes
    macro_f1_score = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

    #printing the full per-class precision, recall, and F1 breakdown
    print(classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=0))

    #printing the overall macro F1 score for easy reading
    print(f"Macro F1: {macro_f1_score:.4f}")

    #returning a dictionary with the macro F1 for saving to JSON
    return {"macro_f1": macro_f1_score}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    #setting up argparse so we can choose the dataset from the command line
    parser = argparse.ArgumentParser(description="Running the full RAG pipeline evaluation")

    #adding the dataset argument to pick between scifact and sciclaimhunt
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scifact", "sciclaimhunt"],
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

    #setting the local cache path for SciFact data
    scifact_data_path = "data/scifact"

    #setting the local cache path for SciClaimHunt data
    sciclaimhunt_data_path = "data/sciclaimhunt"

    #detecting whether a GPU is available and setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #printing which device will be used for all inference
    print(f"Using device: {device}")

    #loading the tokenizer from the fine-tuned RoBERTa checkpoint
    roberta_tokenizer = AutoTokenizer.from_pretrained(parsed_arguments.model_path)

    #loading the fine-tuned RoBERTa model from the checkpoint
    roberta_model = AutoModelForSequenceClassification.from_pretrained(parsed_arguments.model_path)

    #moving the model to the selected device
    roberta_model = roberta_model.to(device)

    #loading claims and labels from whichever dataset was selected
    if parsed_arguments.dataset == "scifact":
        #loading SciFact test claims and gold labels
        claims, labels = load_scifact_data(scifact_data_path)
    else:
        #loading SciClaimHunt test claims and gold labels
        claims, labels = load_sciclaimhunt_data(sciclaimhunt_data_path)

    #printing how many claims were loaded so we can verify it looks right
    print(f"Loaded {len(claims)} claims for evaluation")

    #loading the document corpus for whichever dataset was selected
    corpus = load_corpus(parsed_arguments.dataset, scifact_data_path, sciclaimhunt_data_path)

    #printing how many corpus documents were loaded
    print(f"Loaded corpus with {len(corpus)} documents")

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
        claims, labels, corpus, roberta_model, roberta_tokenizer, device,
        parsed_arguments.dataset, top_k=parsed_arguments.top_k
    )
    all_pipeline_results["dense_roberta"] = dense_metrics

    #running pipeline 4: dense retrieval then stance reranking then RoBERTa
    dense_reranked_metrics = run_dense_reranked_pipeline(
        claims, labels, corpus, roberta_model, roberta_tokenizer, device,
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
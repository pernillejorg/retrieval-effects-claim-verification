"""
Full RAG pipeline evaluation for scientific claim verification (Step 5):

This script connects all components built in previous steps: retrieval (BM25 and
dense/mpnet), soft stance-aware reranking, and the fine-tuned RoBERTa classifiers.
Four pipeline conditions are run and compared.

The four conditions are:
1. No retrieval:                    Model 1 (claim-only), no evidence context
2. BM25 + RoBERTa:                  sparse retrieval, then classify with Model 2
3. Dense + RoBERTa:                 dense (mpnet) retrieval, then classify with Model 2
4. Dense + soft rerank + RoBERTa:   retrieve, soft stance rerank, then classify with Model 2

The key questions this step answers: does giving the classifier retrieved evidence help
it verify claims better than no context at all, and does stance reranking that evidence
help or hurt at the classification level (Step 4 showed it degrades retrieval recall,
this step tests whether that damage propagates to final F1)?

--- Design decisions and their justifications (for the thesis) ---

Two classifiers (from Step 2, produced by baseline.py):
- Model 1 (claim-only, saved as baseline_<dataset>) is used ONLY for the no-retrieval
  condition, because that condition has no evidence and Model 1 was trained on claim
  text alone.
- Model 2 (claim+evidence, saved as evidence_<dataset>) is used for the three retrieval
  conditions, because it was trained on claim+evidence pairs and can therefore read
  evidence. Using each classifier for the input format it was trained on is the
  methodologically correct RAG setup, and makes the comparison fair rather than feeding
  evidence to a model that never learned to use it.

Retriever choice (mpnet): all-mpnet-base-v2 is used as the dense retriever. Both it and
the lighter all-MiniLM-L6-v2 were evaluated in Step 3; mpnet was chosen empirically as it
won 4 of 6 dense recall metrics and resolved MiniLM's anomalous top-rank underperformance.
This choice is therefore evidenced, not assumed.

Reranking mode (soft): soft reranking (reorder, keep all documents) is used rather than
hard filtering. Both were evaluated in Step 4; hard filtering was shown to remove ~9 of 10
documents and destroy recall, so soft is the only viable mode to carry into the pipeline.
This choice is likewise evidenced.

top_k = 3 (documents fed to the classifier, the default): anchored to MAPLE (Zeng and
Zubiaga, 2024), which uses the top-3 retrieved abstracts, so Step 5 is reported at this
depth for comparability with prior work. Step 6 sweeps k in {1, 3, 5, 10} via the
--top_k argument to analyse retrieval-depth sensitivity.

rerank_pool_size = 10 (documents retrieved before reranking): the reranker is given a
larger candidate pool than the top_k it ultimately selects from, following the standard
"retrieve more, then select down" pattern, and reranking can only reorder documents it is
shown, so a pool larger than top_k gives it room to promote a better document into the
final set. Retrieving 10 and selecting the top_k documents balances this against the cost of scoring more
documents per claim.

neutral_threshold = 0.5 (loose) by default: 0.5 is the "majority-neutral" cut-point (the
NLI model considers a document stance-free if its neutral probability exceeds one half),
and 0.8 is the "confidently-neutral" cut-point. Both were used in Step 4, where the
threshold genuinely affects hard filtering. In THIS pipeline, however, soft reranking sorts
documents by stance score and never removes any document by threshold: the reranker only
computes a per-document `would_be_filtered` flag from neutral_threshold and logs it, without
dropping the document (see reranker.rerank). The threshold therefore does not change which
documents reach the classifier or the resulting F1, and the loose value 0.5 is retained for
consistency with Step 4 rather than swept here; sweeping it would produce identical pipeline
results, as confirmed by the identical soft loose/strict rows in Step 4. (Note: this is
unrelated to the separate 0.5 strong-stance cut-point used inside the Step 7 analysis script,
which is a different diagnostic on stance_score, not this neutral_threshold.)

max_length: the no-retrieval condition uses max_length=128, matching Model 1's training
(the longest claim is 75 tokens, so 128 never truncates a claim). The retrieval conditions
use max_length=512, RoBERTa's maximum, because claim+evidence pairs are long and need the
full budget.

Input construction (truncate_and_concatenate): the claim and the concatenated evidence are
passed to the tokenizer as a text PAIR, so RoBERTa inserts its own segment boundary
(</s></s>) rather than a hand-built separator string. The token budget is split so the
claim is never truncated and the evidence fills the remainder.

Per-claim records: for every condition, the pipeline saves per-claim outputs (claim, true
and predicted label, confidence, retrieved document ids and full retrieved text, and correctness) so
the Step 7 failure taxonomy and Step 8 confidence analysis can be run without re-executing
the pipeline. The reranked condition additionally saves the pre-rerank document order, so
the effect of reranking on ordering can be inspected directly.
"""

#importing os for file path handling and making directories
import os

#importing json for saving results to disk after each run
import json

#importing argparse so we can select the dataset from the command line
import argparse

#importing torch for running inference on GPU or CPU
import torch

#importing the system module so we can add the project root to the Python path
import sys

#adding the project root to the path so we can import from data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

#defining the unified label to integer mapping used by both datasets
LABEL_TO_ID = {
    LABEL_SUPPORT: 0,
    LABEL_CONTRADICT: 1,
    LABEL_NEI: 2,
}

ID_TO_LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}

# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
'''
#This is the Part A done for RAG pipeline run, however as seen from the experiments, top-k 
is not affecting the retrieval enough therefore doing Part B code (below) to equal token 
budget per document so k becomes a real variable.
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

    #guarding against an unusually long claim consuming the whole budget
    if document_token_budget <= 0:
        #no room left for evidence after the claim -- return claim with empty evidence
        return claim_text, ""

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
'''

'''
Part B done to equal token budget per document so k becomes a real variable.
'''
def truncate_and_concatenate(claim_text, document_texts, tokenizer, max_total_length=512):
    """
    Building the claim and the concatenated evidence as TWO separate strings, so the
    tokenizer can join them as a proper text pair (claim, evidence). RoBERTa then inserts
    its correct segment boundary (</s></s>) itself -- we do not hand-build a separator.

    Option B (equal per-document budget): the evidence token budget is divided EQUALLY
    across the k retrieved documents, so each document contributes a fair share and
    increasing k genuinely changes the evidence the model sees. This avoids the
    saturation of naive concatenation (Option A), where long scientific abstracts fill
    the 512-token window after ~2 documents and any further documents are truncated away,
    making k above ~2 have no effect.

    Returns: (claim_text, evidence_text) tuple, ready for tokenizer(claim, evidence, ...)
    """
    #reserving tokens for the special tokens the tokenizer adds around a pair
    special_tokens_count = tokenizer.num_special_tokens_to_add(pair=True)
    available_tokens = max_total_length - special_tokens_count

    #tokenising the claim to measure how many tokens it uses
    claim_tokens = tokenizer.encode(claim_text, add_special_tokens=False)

    #the evidence gets whatever budget remains after the claim
    document_token_budget = available_tokens - len(claim_tokens)

    #guarding against an unusually long claim consuming the whole budget,
    #or against an empty document list
    if document_token_budget <= 0 or len(document_texts) == 0:
        return claim_text, ""

    #dividing the evidence budget EQUALLY across the k documents (Option B).
    #each document gets an equal per-document share so all k contribute.
    num_documents = len(document_texts)
    per_document_budget = document_token_budget // num_documents

    #if there are so many documents that each share rounds down to zero,
    #fall back to giving each at least 1 token so every document is represented
    if per_document_budget < 1:
        per_document_budget = 1

    #accumulating evidence text, truncating each document to its equal share
    concatenated_documents = ""
    for document_text in document_texts:
        document_tokens = tokenizer.encode(document_text, add_special_tokens=False)
        #truncating this document to its per-document budget
        document_tokens = document_tokens[:per_document_budget]
        truncated_document = tokenizer.decode(document_tokens, skip_special_tokens=True)
        concatenated_documents += " " + truncated_document

    #returning claim and evidence as a PAIR -- the tokenizer will join them correctly
    return claim_text, concatenated_documents.strip()


# ---------------------------------------------------------------------------
# Optional: capture the EXACT classifier input (for the Step 7 taxonomy)
# ---------------------------------------------------------------------------

def capture_classifier_input(tokenizer, claim_part, evidence_part, max_length=512):
    """
    Reconstruct exactly what the classifier saw for this prediction, so Step 7 can distinguish
    a genuine confident_wrong_prediction from an input-construction / truncation failure.

    This is PURELY ADDITIVE metadata: it does not change which documents are retrieved, the
    concatenation, the tokenisation used for inference, or the prediction. It only records the
    final input string and how much truncation the 512-token limit applied.

    Returns a dict with:
      classifier_input_text                - the decoded final input actually classified
      input_token_count_before_truncation  - token count after equal per-document
                                            budgeting but before the tokenizer's
                                            final max_length cap
      input_token_count_after_truncation   - token count after the final max_length cap
      was_truncated                        - True if that final cap removed additional tokens
    """
    #encoding without truncation to measure the full length the input would have had
    full_ids = tokenizer(claim_part, evidence_part, truncation=False)["input_ids"]
    #encoding with the same cap the inference call uses, to get what the model really saw
    kept_ids = tokenizer(claim_part, evidence_part, truncation=True,
                         max_length=max_length)["input_ids"]
    return {
        "classifier_input_text": tokenizer.decode(kept_ids),
        "input_token_count_before_truncation": len(full_ids),
        "input_token_count_after_truncation": len(kept_ids),
        "was_truncated": len(full_ids) > len(kept_ids),
    }


# ---------------------------------------------------------------------------
# Pipeline conditions
# ---------------------------------------------------------------------------

def run_no_retrieval_pipeline(claims_data, model, tokenizer, device, dataset_name):
    print("\nRunning pipeline: No Retrieval (Model 1, claim-only)")
    model.eval()
    predicted_labels, true_labels, records = [], [], []

    for claim_dict in claims_data:
        claim_text = claim_dict["claim"]
        true_id = LABEL_TO_ID[claim_dict["label"]]

        #classifying the claim with no evidence context
        encoded_input = tokenizer(
            claim_text,
            return_tensors="pt", truncation=True, max_length=128, padding=True,
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            logits = model(**encoded_input).logits

        #softmax to get probabilities and the confidence (max prob) for Step 8
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_id].item())

        predicted_labels.append(pred_id)
        true_labels.append(true_id)
        records.append({
            "claim_id": claim_dict["id"],
            "claim": claim_text,
            "true_label": ID_TO_LABEL[true_id],
            "predicted_label": ID_TO_LABEL[pred_id],
            "confidence": confidence,
            "probabilities": [float(p) for p in probabilities.tolist()],
            "correct": pred_id == true_id,
            "retrieved_doc_ids": [],
            "condition": "no_retrieval",
        })

    metrics = compute_metrics(predicted_labels, true_labels, dataset_name)
    return metrics, records


def run_bm25_pipeline(claims_data, bm25_retriever, model, tokenizer, device, dataset_name, top_k=3):
    print("\nRunning pipeline: BM25 + RoBERTa (Model 2)")
    model.eval()
    predicted_labels, true_labels, records = [], [], []

    for claim_dict in claims_data:
        claim_text = claim_dict["claim"]
        true_id = LABEL_TO_ID[claim_dict["label"]]

        retrieved_documents = bm25_retriever.retrieve(claim_text, k=top_k)
        retrieved_document_texts = [doc["text"] for doc in retrieved_documents]
        retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_documents]
        #saving id, score, and full retrieved text for Step 7 manual failure analysis
        retrieved_docs = [
            {"doc_id": doc["doc_id"], "score": float(doc["score"]), "text": doc["text"]}
            for doc in retrieved_documents
        ]

        claim_part, evidence_part = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)
        encoded_input = tokenizer(
            claim_part, evidence_part,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            logits = model(**encoded_input).logits

        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_id].item())

        predicted_labels.append(pred_id)
        true_labels.append(true_id)
        records.append({
            "claim_id": claim_dict["id"],
            "claim": claim_text,
            "true_label": ID_TO_LABEL[true_id],
            "predicted_label": ID_TO_LABEL[pred_id],
            "confidence": confidence,
            "probabilities": [float(p) for p in probabilities.tolist()],
            "correct": pred_id == true_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_docs": retrieved_docs,
            #additive: exact classifier input + truncation info for Step 7
            **capture_classifier_input(tokenizer, claim_part, evidence_part),
            "condition": "bm25_roberta",
        })

    metrics = compute_metrics(predicted_labels, true_labels, dataset_name)
    return metrics, records

#passing in the dense retriever as an argument to avoid re-encoding the corpus every time
def run_dense_pipeline(claims_data, dense_retriever, model, tokenizer, device, dataset_name, top_k=3):
    print("\nRunning pipeline: Dense + RoBERTa (Model 2)")
    model.eval()
    predicted_labels, true_labels, records = [], [], []

    for claim_dict in claims_data:
        claim_text = claim_dict["claim"]
        true_id = LABEL_TO_ID[claim_dict["label"]]

        retrieved_documents = dense_retriever.retrieve(claim_text, k=top_k)
        retrieved_document_texts = [doc["text"] for doc in retrieved_documents]
        retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_documents]
        #saving id, score, and full retrieved text for Step 7 manual failure analysis
        retrieved_docs = [
            {"doc_id": doc["doc_id"], "score": float(doc["score"]), "text": doc["text"]}
            for doc in retrieved_documents
        ]

        claim_part, evidence_part = truncate_and_concatenate(claim_text, retrieved_document_texts, tokenizer)
        encoded_input = tokenizer(
            claim_part, evidence_part,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            logits = model(**encoded_input).logits

        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_id].item())

        predicted_labels.append(pred_id)
        true_labels.append(true_id)
        records.append({
            "claim_id": claim_dict["id"],
            "claim": claim_text,
            "true_label": ID_TO_LABEL[true_id],
            "predicted_label": ID_TO_LABEL[pred_id],
            "confidence": confidence,
            "probabilities": [float(p) for p in probabilities.tolist()],
            "correct": pred_id == true_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_docs": retrieved_docs,
            #additive: exact classifier input + truncation info for Step 7
            **capture_classifier_input(tokenizer, claim_part, evidence_part),
            "condition": "dense_roberta",
        })

    metrics = compute_metrics(predicted_labels, true_labels, dataset_name)
    return metrics, records

#passing in the dense retriever as an argument to avoid re-encoding the corpus every time
def run_dense_reranked_pipeline(claims_data, dense_retriever, stance_reranker, model, tokenizer,
                                device, dataset_name, top_k=3, rerank_pool_size=10, neutral_threshold=0.5):
    print("\nRunning pipeline: Dense + Soft Stance Reranking + RoBERTa (Model 2)")
    model.eval()
    predicted_labels, true_labels, records = [], [], []

    for claim_dict in claims_data:
        claim_text = claim_dict["claim"]
        true_id = LABEL_TO_ID[claim_dict["label"]]

        #retrieve a larger pool, soft-rerank by stance, then take top_k
        retrieved_documents = dense_retriever.retrieve(claim_text, k=rerank_pool_size)
        #capturing the ORIGINAL dense order (before reranking) so Step 7 can see what
        #the reranker changed -- directly supports the Step 4 reranking-effect analysis
        pre_rerank_doc_ids = [doc["doc_id"] for doc in retrieved_documents]

        reranked_documents = stance_reranker.rerank(claim_text, retrieved_documents,
                                                    neutral_threshold=neutral_threshold)
        top_reranked = reranked_documents[:top_k]

        top_reranked_document_texts = [doc["text"] for doc in top_reranked]
        retrieved_doc_ids = [doc["doc_id"] for doc in top_reranked]
        #saving retrieval, stance, and neutral scores plus full text for Step 7 analysis
        retrieved_docs = [
            {
                "doc_id": doc["doc_id"],
                "score": float(doc.get("score", 0.0)),
                "stance_score": float(doc.get("stance_score", 0.0)),
                "neutral_score": float(doc.get("neutral_score", 0.0)),
                "text": doc["text"],
            }
            for doc in top_reranked
        ]

        claim_part, evidence_part = truncate_and_concatenate(claim_text, top_reranked_document_texts, tokenizer)
        encoded_input = tokenizer(
            claim_part, evidence_part,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            logits = model(**encoded_input).logits

        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_id].item())

        predicted_labels.append(pred_id)
        true_labels.append(true_id)
        records.append({
            "claim_id": claim_dict["id"],
            "claim": claim_text,
            "true_label": ID_TO_LABEL[true_id],
            "predicted_label": ID_TO_LABEL[pred_id],
            "confidence": confidence,
            "probabilities": [float(p) for p in probabilities.tolist()],
            "correct": pred_id == true_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_docs": retrieved_docs,
            "pre_rerank_doc_ids": pre_rerank_doc_ids,
            #additive: exact classifier input + truncation info for Step 7
            **capture_classifier_input(tokenizer, claim_part, evidence_part),
            "condition": "dense_reranked_roberta",
        })

    metrics = compute_metrics(predicted_labels, true_labels, dataset_name)
    return metrics, records


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

    #two model paths: Model 1 (claim-only) for no-retrieval, Model 2 (evidence) for RAG conditions
    parser.add_argument("--model1_path", type=str, required=True,
                        help="Model 1 (claim-only baseline) for the no-retrieval condition")
    parser.add_argument("--model2_path", type=str, required=True,
                        help="Model 2 (claim+evidence) for the retrieval conditions")
    parser.add_argument("--output_path", type=str, default="results/step5_pipeline_scifact.json")
    parser.add_argument("--records_path", type=str, default="results/step5_records_scifact.json")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--rerank_pool_size", type=int, default=10)
    parser.add_argument("--neutral_threshold", type=float, default=0.5)
    parsed_arguments = parser.parse_args()

    #detecting the best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    #loading BOTH classifiers
    print(f"Loading Model 1 (claim-only) from: {parsed_arguments.model1_path}")
    model1_tokenizer = AutoTokenizer.from_pretrained(parsed_arguments.model1_path)
    model1 = AutoModelForSequenceClassification.from_pretrained(parsed_arguments.model1_path).to(device)

    print(f"Loading Model 2 (claim+evidence) from: {parsed_arguments.model2_path}")
    model2_tokenizer = AutoTokenizer.from_pretrained(parsed_arguments.model2_path)
    model2 = AutoModelForSequenceClassification.from_pretrained(parsed_arguments.model2_path).to(device)

    #loading claims and corpus for the chosen dataset
    if parsed_arguments.dataset == "scifact":
        claims_data, corpus = load_scifact(split="validation")
    else:
        claims_data, corpus = load_scifact_open(corpus_file="full")
    print(f"Loaded {len(claims_data)} claims and corpus of {len(corpus)} documents")

    #building retrievers + reranker ONCE
    print("\nBuilding BM25 retriever...")
    bm25_retriever = BM25Retriever(corpus)
    print("Building dense retriever (one-time encoding)...")
    dense_retriever = DenseRetriever(corpus)
    print("Building stance reranker...")
    stance_reranker = StanceReranker(device=device)

    all_pipeline_results = {}
    all_records = []

    #1. no retrieval -> Model 1
    m, r = run_no_retrieval_pipeline(claims_data, model1, model1_tokenizer, device, parsed_arguments.dataset)
    all_pipeline_results["no_retrieval"] = m
    all_records.extend(r)

    #2. BM25 -> Model 2
    m, r = run_bm25_pipeline(claims_data, bm25_retriever, model2, model2_tokenizer, device,
                             parsed_arguments.dataset, top_k=parsed_arguments.top_k)
    all_pipeline_results["bm25_roberta"] = m
    all_records.extend(r)

    #3. Dense -> Model 2
    m, r = run_dense_pipeline(claims_data, dense_retriever, model2, model2_tokenizer, device,
                              parsed_arguments.dataset, top_k=parsed_arguments.top_k)
    all_pipeline_results["dense_roberta"] = m
    all_records.extend(r)

    #4. Dense + soft rerank -> Model 2
    m, r = run_dense_reranked_pipeline(claims_data, dense_retriever, stance_reranker, model2, model2_tokenizer,
                                       device, parsed_arguments.dataset, top_k=parsed_arguments.top_k,
                                       rerank_pool_size=parsed_arguments.rerank_pool_size,
                                       neutral_threshold=parsed_arguments.neutral_threshold)
    all_pipeline_results["dense_reranked_roberta"] = m
    all_records.extend(r)

    #saving aggregate metrics (safe dirname handling)
    out_dir = os.path.dirname(parsed_arguments.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    #inserting the neutral threshold into the filename so loose/strict runs don't overwrite
    thr_str = str(parsed_arguments.neutral_threshold).replace(".", "_")
    base, ext = os.path.splitext(parsed_arguments.output_path)
    output_path = f"{base}_k{parsed_arguments.top_k}_thr{thr_str}{ext}"
    with open(output_path, "w") as f:
        json.dump({
            "dataset": parsed_arguments.dataset,
            "top_k": parsed_arguments.top_k,
            "neutral_threshold": parsed_arguments.neutral_threshold,
            "metrics": all_pipeline_results,
        }, f, indent=2)
    print(f"\nAggregate metrics saved to {output_path}")

    #saving per-claim records for Steps 7 and 8
    rec_dir = os.path.dirname(parsed_arguments.records_path)
    if rec_dir:
        os.makedirs(rec_dir, exist_ok=True)
    rbase, rext = os.path.splitext(parsed_arguments.records_path)
    records_path = f"{rbase}_k{parsed_arguments.top_k}_thr{thr_str}{rext}"
    rec_dir = os.path.dirname(records_path)
    if rec_dir:
        os.makedirs(rec_dir, exist_ok=True)
    with open(records_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Per-claim records saved to {records_path}")

    #returning the results dictionary so Colab can capture it directly
    return all_pipeline_results


#running main only when this script is called directly, not when imported
if __name__ == "__main__":
    main()
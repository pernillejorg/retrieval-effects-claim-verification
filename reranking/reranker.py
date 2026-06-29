"""
stance-aware reranking for claim verification

After standard BM25 or dense retrieval gives a candidate pool of documents,
this module filters and reranks those documents by whether they take a
stance on the claim of entailment or contradiction, rather than just
being topically related.

The key insight: topical similarity is not enough for fact-checking.
A document about omega-3 and cardiovascular health might be retrieved for
a related claim, but if it says nothing specific about that claim, it adds
noise rather than signal. The stance filter catches this.

Design decisions (document these in your thesis):
- cross-encoder/nli-deberta-v3-small: small enough for CPU/MacBook,
  strong enough for zero-shot NLI stance scoring
- Two thresholds (loose=0.5, strict=0.8): tests threshold sensitivity
  without over-engineering
- Neutral score is the filter criterion: documents where the model
  assigns high probability to neutral are filtered out
- Stance score = max(entailment_prob, contradiction_prob): captures
  whether a document takes any stance at all on the claim

References:
- He et al. (2021) -- DeBERTa: Decoding-enhanced BERT with Disentangled Attention
- Stammbach & Neumann (2019) -- NLI-filtering for health claim verification
"""

#importing the operating system module for path handling
import os

#importing the system module to add the project root to the Python path
import sys

#adding the project root to the path so we can import from data/ and retrieval/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing PyTorch for running the NLI model inference
import torch

#importing json for saving results to disk
import json

#importing argparse so we can specify the dataset from the command line
import argparse

#importing the cross-encoder pipeline for zero-shot NLI stance scoring
from transformers import pipeline

#importing our unified data loading functions for both datasets
from data.utils import load_scifact, load_sciclaimhunt

#importing the retrieval classes so we can build the candidate pool
from retrieval.retrieval import BM25Retriever, DenseRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#defining the NLI model used for stance scoring
#cross-encoder/nli-deberta-v3-small is fast, accurate, and runs on CPU
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

#defining the number of candidate documents to retrieve before reranking
#we retrieve more than needed so the filter has candidates to work with
RETRIEVAL_K = 10

#defining the loose stance threshold -- documents with neutral score above
#this are filtered out in the loose filtering condition
LOOSE_NEUTRAL_THRESHOLD = 0.5

#defining the strict stance threshold -- documents with neutral score above
#this are filtered out in the strict filtering condition
STRICT_NEUTRAL_THRESHOLD = 0.8

#defining the label order that cross-encoder/nli-deberta-v3-small outputs
#the model outputs scores in this fixed order: contradiction, entailment, neutral
NLI_LABEL_ORDER = ["contradiction", "entailment", "neutral"]

# ---------------------------------------------------------------------------
# Stance Reranker
# ---------------------------------------------------------------------------

class StanceReranker:
    """
    Implementing stance-aware reranking using zero-shot NLI inference.

    For each (claim, document) pair, the NLI model scores entailment,
    contradiction, and neutral probabilities. Documents with high neutral
    scores are filtered out -- only stance-bearing documents are kept.

    This is the novel technical contribution of this project.
    """

    def __init__(self, device=None):
        """
        Loading the NLI model and setting up the inference pipeline.

        device: torch.device or None (auto-detects if None)
        """

        #auto-detecting the best available device for inference
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        #storing the device for use during inference
        self.device = device

        #printing which device will be used
        print(f"Loading stance reranker: {NLI_MODEL_NAME}")
        print(f"Using device: {device}")

        #converting device to integer index for the transformers pipeline
        #transformers pipeline expects -1 for CPU, 0 for first GPU
        if str(device) == "cpu":
            device_index = -1
        elif str(device) == "mps":
            device_index = -1
        else:
            device_index = 0

        #loading the NLI cross-encoder model as a zero-shot classification pipeline
        self.nli_pipeline = pipeline(
            "zero-shot-classification",
            model=NLI_MODEL_NAME,
            device=device_index,
        )

        #printing confirmation that the model is ready
        print(f"Stance reranker loaded successfully.")

    def score_document(self, claim_text, document_text):
        """
        Scoring a single (claim, document) pair for stance.

        claim_text: str -- the claim to verify
        document_text: str -- the candidate evidence document

        Returns: dict with keys:
            entailment_score   (float) probability document supports claim
            contradiction_score(float) probability document contradicts claim
            neutral_score      (float) probability document is irrelevant
            stance_score       (float) max(entailment, contradiction) -- overall stance strength
            predicted_label    (str)   the highest-scoring NLI label
        """

        #running zero-shot NLI inference with entailment/contradiction/neutral as candidate labels
        nli_result = self.nli_pipeline(
            claim_text,
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template="{}",
        )

        #building a dictionary mapping label to score for easy access
        label_to_score = {
            label: score
            for label, score in zip(nli_result["labels"], nli_result["scores"])
        }

        #extracting individual scores for each NLI label
        entailment_score = label_to_score.get("entailment", 0.0)
        contradiction_score = label_to_score.get("contradiction", 0.0)
        neutral_score = label_to_score.get("neutral", 0.0)

        #computing stance score as the maximum of entailment and contradiction
        #this captures whether the document takes any position on the claim at all
        stance_score = max(entailment_score, contradiction_score)

        #finding the predicted label as the one with the highest score
        predicted_label = max(label_to_score, key=label_to_score.get)

        #returning a complete score dict for this document
        return {
            "entailment_score": entailment_score,
            "contradiction_score": contradiction_score,
            "neutral_score": neutral_score,
            "stance_score": stance_score,
            "predicted_label": predicted_label,
        }

    def rerank(self, claim_text, retrieved_documents, neutral_threshold):
        """
        Reranking retrieved documents by stance score and filtering neutrals.

        claim_text: str -- the claim to verify
        retrieved_documents: list of dicts (from BM25Retriever or DenseRetriever)
            each dict has keys: doc_id, text, score, rank
        neutral_threshold: float -- documents with neutral_score above this are filtered out

        Returns: list of dicts, each containing the original retrieval fields plus:
            entailment_score   (float)
            contradiction_score(float)
            neutral_score      (float)
            stance_score       (float)
            predicted_label    (str)
            reranked_position  (int) position after reranking (1 = most stance-bearing)
        """

        #scoring each retrieved document for stance against the claim
        scored_documents = []
        for document in retrieved_documents:

            #computing NLI stance scores for this (claim, document) pair
            stance_scores = self.score_document(claim_text, document["text"])

            #combining the original retrieval fields with the new stance scores
            scored_document = {
                "doc_id": document["doc_id"],
                "text": document["text"],
                "retrieval_score": document["score"],
                "retrieval_rank": document["rank"],
                "entailment_score": stance_scores["entailment_score"],
                "contradiction_score": stance_scores["contradiction_score"],
                "neutral_score": stance_scores["neutral_score"],
                "stance_score": stance_scores["stance_score"],
                "predicted_label": stance_scores["predicted_label"],
            }
            scored_documents.append(scored_document)

        #filtering out documents whose neutral score exceeds the threshold
        #these are documents that are topically related but take no stance on the claim
        stance_bearing_documents = [
            document for document in scored_documents
            if document["neutral_score"] <= neutral_threshold
        ]

        #sorting remaining documents by stance score descending
        #most stance-bearing documents come first
        stance_bearing_documents.sort(key=lambda d: d["stance_score"], reverse=True)

        #adding reranked position to each document
        for position, document in enumerate(stance_bearing_documents, start=1):
            document["reranked_position"] = position

        #returning the filtered and reranked list
        return stance_bearing_documents

# ---------------------------------------------------------------------------
# Recall@k computation
# ---------------------------------------------------------------------------

def compute_recall_at_k(claims, retrieved_results, k_values):
    """
    Computing Recall@k for a set of claims and their retrieved documents.

    claims: list of claim dicts (must have evidence_doc_ids field)
    retrieved_results: dict {claim_id: list of doc dicts with doc_id field}
    k_values: list of int -- k values to compute recall at

    Returns: dict {k: recall_score}
    """

    #filtering to only claims that have annotated evidence
    claims_with_evidence = [
        claim for claim in claims if claim["evidence_doc_ids"]
    ]

    #initialising recall accumulators for each k value
    recall_accumulators = {k: 0.0 for k in k_values}

    #computing recall for each claim with annotated evidence
    for claim in claims_with_evidence:

        #getting the retrieved documents for this claim
        claim_retrieved = retrieved_results.get(claim["id"], [])

        #checking recall at each k value
        for k in k_values:

            #getting the top-k retrieved doc ids
            top_k_doc_ids = [
                document["doc_id"] for document in claim_retrieved[:k]
            ]

            #checking if any of the annotated evidence docs appear in top-k
            evidence_found = any(
                evidence_doc_id in top_k_doc_ids
                for evidence_doc_id in claim["evidence_doc_ids"]
            )

            #adding 1 if evidence was found, 0 otherwise
            if evidence_found:
                recall_accumulators[k] += 1.0

    #dividing by number of claims with evidence to get recall rate
    number_of_claims_with_evidence = len(claims_with_evidence)
    recall_at_k = {
        k: recall_accumulators[k] / number_of_claims_with_evidence
        if number_of_claims_with_evidence > 0 else 0.0
        for k in k_values
    }

    return recall_at_k

# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_reranking_evaluation(dataset_name="scifact"):
    """
    Running the full stance-aware reranking evaluation pipeline.

    Loads the dataset, builds retrieval index, retrieves candidates,
    applies stance reranking at two thresholds, and reports Recall@k
    before and after reranking.

    dataset_name: str -- 'scifact' or 'sciclaimhunt'

    Returns: dict with recall numbers for thesis tables
    """

    #printing the evaluation header
    print(f"\n{'='*60}")
    print(f"  Stance-Aware Reranking Evaluation -- {dataset_name.upper()}")
    print(f"{'='*60}\n")

    #auto-detecting the best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #loading the validation split and corpus for the chosen dataset
    if dataset_name == "scifact":
        #loading SciFact validation claims and retrieval corpus
        val_claims, corpus = load_scifact(split="validation")

    elif dataset_name == "sciclaimhunt":
        #loading SciClaimHunt validation claims and retrieval corpus
        val_claims, corpus = load_sciclaimhunt(split="val")

    else:
        #raising an error for unrecognised dataset names
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'scifact' or 'sciclaimhunt'.")

    #printing basic corpus and claim statistics
    print(f"Validation claims : {len(val_claims)}")
    print(f"Corpus size       : {len(corpus)} documents\n")

    #filtering to claims with annotated evidence for recall computation
    claims_with_evidence = [c for c in val_claims if c["evidence_doc_ids"]]
    print(f"Claims with evidence (non-NEI): {len(claims_with_evidence)}")
    print(f"NEI claims (excluded from Recall@k): {len(val_claims) - len(claims_with_evidence)}\n")

    # -----------------------------------------------------------------------
    # Dense retrieval -- building candidate pool
    # -----------------------------------------------------------------------

    #building the dense retriever over the full corpus
    print("--- Dense Retrieval (candidate pool) ---")
    dense_retriever = DenseRetriever(corpus, device=device)

    #retrieving top-k candidates for every validation claim
    print(f"Running dense retrieval (k={RETRIEVAL_K})...")
    dense_retrieved = {}
    for claim_index, claim in enumerate(val_claims):

        #printing progress every 100 claims
        if (claim_index + 1) % 100 == 1:
            print(f"  Retrieving for claim {claim_index + 1} / {len(val_claims)}...")

        #retrieving top-k documents for this claim
        dense_retrieved[claim["id"]] = dense_retriever.retrieve(
            claim["text"] if "text" in claim else claim["claim"],
            k=RETRIEVAL_K,
        )

    #computing baseline dense recall before reranking
    k_values = [1, 5, 10]
    dense_recall_before = compute_recall_at_k(val_claims, dense_retrieved, k_values)

    # -----------------------------------------------------------------------
    # Stance reranking -- applying NLI filter at two thresholds
    # -----------------------------------------------------------------------

    #loading the stance reranker model
    print("\n--- Stance-Aware Reranking ---")
    stance_reranker = StanceReranker(device=device)

    #running reranking at loose threshold
    print(f"\nApplying loose threshold (neutral <= {LOOSE_NEUTRAL_THRESHOLD})...")
    loose_reranked = {}
    for claim_index, claim in enumerate(val_claims):

        #printing progress every 100 claims
        if (claim_index + 1) % 100 == 1:
            print(f"  Reranking claim {claim_index + 1} / {len(val_claims)}...")

        #applying stance filter with loose threshold
        claim_text = claim["claim"]
        candidates = dense_retrieved[claim["id"]]
        loose_reranked[claim["id"]] = stance_reranker.rerank(
            claim_text, candidates, LOOSE_NEUTRAL_THRESHOLD
        )

    #computing recall after loose reranking
    loose_recall = compute_recall_at_k(val_claims, loose_reranked, k_values)

    #running reranking at strict threshold
    print(f"\nApplying strict threshold (neutral <= {STRICT_NEUTRAL_THRESHOLD})...")
    strict_reranked = {}
    for claim_index, claim in enumerate(val_claims):

        #printing progress every 100 claims
        if (claim_index + 1) % 100 == 1:
            print(f"  Reranking claim {claim_index + 1} / {len(val_claims)}...")

        #applying stance filter with strict threshold
        claim_text = claim["claim"]
        candidates = dense_retrieved[claim["id"]]
        strict_reranked[claim["id"]] = stance_reranker.rerank(
            claim_text, candidates, STRICT_NEUTRAL_THRESHOLD
        )

    #computing recall after strict reranking
    strict_recall = compute_recall_at_k(val_claims, strict_reranked, k_values)

    # -----------------------------------------------------------------------
    # Results reporting
    # -----------------------------------------------------------------------

    #printing the recall comparison table
    print(f"\n{'='*60}")
    print(f"  Reranking Recall@k -- {dataset_name.upper()} (validation)")
    print(f"{'='*60}")
    print(f"\n  {'Method':<30} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print(f"  {'-'*50}")
    print(f"  {'Dense (before reranking)':<30} "
          f"{dense_recall_before[1]:>6.3f} "
          f"{dense_recall_before[5]:>6.3f} "
          f"{dense_recall_before[10]:>6.3f}")
    print(f"  {'Dense + loose filter':<30} "
          f"{loose_recall[1]:>6.3f} "
          f"{loose_recall[5]:>6.3f} "
          f"{loose_recall[10]:>6.3f}")
    print(f"  {'Dense + strict filter':<30} "
          f"{strict_recall[1]:>6.3f} "
          f"{strict_recall[5]:>6.3f} "
          f"{strict_recall[10]:>6.3f}")
    print(f"{'='*60}\n")

    #assembling results dict for saving to Drive
    results = {
        "dataset": dataset_name,
        "retrieval_k": RETRIEVAL_K,
        "loose_neutral_threshold": LOOSE_NEUTRAL_THRESHOLD,
        "strict_neutral_threshold": STRICT_NEUTRAL_THRESHOLD,
        "recall_at_k": {
            "dense_before_reranking": {
                str(k): dense_recall_before[k] for k in k_values
            },
            "dense_loose_reranking": {
                str(k): loose_recall[k] for k in k_values
            },
            "dense_strict_reranking": {
                str(k): strict_recall[k] for k in k_values
            },
        },
    }

    #returning results dict for saving in Colab
    return results

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    #importing argparse for command line argument parsing
    import argparse

    #parsing command line arguments so we can specify the dataset without editing the file
    parser = argparse.ArgumentParser(description="Stance-aware reranking evaluation")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "sciclaimhunt"],
                        help="Dataset to run reranking evaluation on")
    args = parser.parse_args()

    #running the reranking evaluation on the chosen dataset
    run_reranking_evaluation(dataset_name=args.dataset)
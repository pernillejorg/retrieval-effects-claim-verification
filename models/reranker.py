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
- Cross-encoder/nli-deberta-v3-small: small enough for CPU/MacBook,
  strong enough for zero-shot NLI stance scoring
- Model used directly (not via pipeline) for transparency and control over
  label order, which is fixed as: contradiction, entailment, neutral
- Two thresholds (loose=0.5, strict=0.8): tests threshold sensitivity
  without over-engineering
- Batched inference per claim: all documents for one claim encoded together
  for efficiency across all (claim, document) pairs in the evaluation set
- Soft reranking (mode="soft"): all documents are kept but reordered by stance
  score so stance-bearing documents appear first. Hard filtering (mode="hard")
  additionally removes documents whose neutral probability exceeds the threshold.
  Both are evaluated so the choice between them is empirically justified, not assumed.
- Stance score = max(entailment_prob, contradiction_prob): captures whether
  a document takes any stance at all on the claim

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

import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification

#importing our unified data loading functions for both datasets
from data.utils import load_scifact, load_scifact_open

#loading the saved retrieval candidates from Step 3 instead of re-running
#retrieval, so no retriever import is needed here.
#from models.retrieval import DenseRetriever

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
#NLI_LABEL_ORDER = ["contradiction", "entailment", "neutral"]

#naming index constants instead of a list, to be used for the tensor below
NLI_LABEL_INDEX_CONTRADICTION = 0
NLI_LABEL_INDEX_ENTAILMENT = 1
NLI_LABEL_INDEX_NEUTRAL = 2

# ---------------------------------------------------------------------------
# Stance Reranker
# ---------------------------------------------------------------------------

class StanceReranker:
    """
    Implementing stance-aware reranking using zero-shot NLI inference.

    For each (claim, document) pair, the NLI model scores entailment,
    contradiction, and neutral probabilities. Documents are first scored for stance. 
    In soft mode, all documents are kept and reranked by stance score. 
    In hard mode, documents with neutral scores above the threshold are removed.
    """

    def __init__(self, device=None):
        """
        Loading the NLI model and tokenizer directly for transparent inference.

        Using the model directly rather than the zero-shot pipeline gives us
        explicit control over label ordering and batching, and is more
        defensible and explainable in the thesis.

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

        #loading the tokenizer for the NLI model
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)

        #loading the NLI sequence classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)

        #moving the model to the selected device
        self.model.to(self.device)

        #setting model to evaluation mode so dropout layers are disabled
        self.model.eval()

        #verifying the label order matches the index constants defined above
        print(f"Model label order: {self.model.config.id2label}")

        #printing confirmation that the model is ready
        print(f"Stance reranker loaded successfully.")

    def rerank(self, claim_text, retrieved_documents, neutral_threshold):
        """
        Reranking retrieved documents by stance score using soft reranking.

        All documents for a single claim are scored in one batched forward pass
        for efficiency. This is important for SciFact_open where we run
        inference calls across the full validation set.

        Unlike hard filtering, soft reranking keeps all documents but reorders
        them so stance-bearing documents (high entailment or contradiction score)
        appear first. The neutral threshold is used to identify documents whose neutral 
        probability exceeds the chosen threshold. Documents are marked using the would_be_filtered 
        flag but are not removed inside this function. 
        The final reranking mode is applied afterwards: soft mode keeps all documents, 
        whereas hard mode removes those marked as would_be_filtered. This avoids the over-filtering
        failure mode where genuine evidence documents are discarded because
        zero-shot NLI on scientific text tends to output high neutral scores.
        """

        #handling the edge case where no documents were retrieved
        if not retrieved_documents:
            return []

        #building all (claim, document) text pairs for batched tokenisation
        claim_texts = [claim_text] * len(retrieved_documents)
        document_texts = [document["text"] for document in retrieved_documents]

        #tokenising all pairs together in one batch
        tokenised_inputs = self.tokenizer(
            claim_texts,
            document_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        #running batched forward pass through the NLI model without gradient tracking
        with torch.no_grad():
            logits = self.model(**tokenised_inputs).logits

        #applying softmax to convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)

        #extracting per-document scores using the fixed label order for this model
        contradiction_scores = probabilities[:, NLI_LABEL_INDEX_CONTRADICTION].cpu().tolist()
        entailment_scores = probabilities[:, NLI_LABEL_INDEX_ENTAILMENT].cpu().tolist()
        neutral_scores = probabilities[:, NLI_LABEL_INDEX_NEUTRAL].cpu().tolist()

        #assembling scored documents with all NLI scores attached
        scored_documents = []
        for document_index, document in enumerate(retrieved_documents):
            stance_score = max(
                entailment_scores[document_index],
                contradiction_scores[document_index]
            )
            scores_dict = {
                "entailment": entailment_scores[document_index],
                "contradiction": contradiction_scores[document_index],
                "neutral": neutral_scores[document_index],
            }
            predicted_label = max(scores_dict, key=scores_dict.get)

            #recording whether this document would have been filtered by the threshold
            #this is logged for analysis but does NOT remove the document
            would_be_filtered = neutral_scores[document_index] > neutral_threshold

            scored_document = {
                "doc_id": document["doc_id"],
                "text": document["text"],
                "retrieval_score": document["score"],
                "retrieval_rank": document["rank"],
                "entailment_score": entailment_scores[document_index],
                "contradiction_score": contradiction_scores[document_index],
                "neutral_score": neutral_scores[document_index],
                "stance_score": stance_score,
                "predicted_label": predicted_label,
                "would_be_filtered": would_be_filtered,
            }
            scored_documents.append(scored_document)

        #sorting all documents by stance score descending so stance-bearing docs come first
        #no documents are removed -- this is soft reranking not hard filtering
        scored_documents.sort(key=lambda d: d["stance_score"], reverse=True)

        #adding reranked position to each document after sorting
        for position, document in enumerate(scored_documents, start=1):
            document["reranked_position"] = position

        #returning all documents reranked by stance score
        return scored_documents

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
# Loading saved retrieval candidates from Step 3
# ---------------------------------------------------------------------------
'''
#For MiniLM
def load_saved_candidates(dataset_name, corpus, retriever="dense"):
    """
    Loading the retrieval candidates saved by Step 3 (retrieval.py) instead of
    re-running the expensive dense retrieval. The saved file stores only
    doc_id + score per candidate, so we look up each document's text from the
    corpus and reconstruct the rank from list position.

    dataset_name: str -- used to find results/retrieval_candidates_<dataset>.json
    corpus: dict {doc_id: text} -- used to resolve doc_id -> document text
    retriever: "dense" or "bm25" -- which retriever's candidates to load

    Returns: dict {claim_id: [ {doc_id, text, score, rank}, ... ]}
    """
    #locating the candidates file saved by Step 3
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    candidates_path = os.path.join(results_dir, f"retrieval_candidates_{dataset_name}.json")
'''
#For MPNet
def load_saved_candidates(dataset_name, corpus, retriever="dense", retriever_name="mpnet"):
    """
    ...
    dataset_name: str -- part of the candidates filename
    retriever_name: str -- which retriever's candidates to load ("mpnet" or "minilm").
        "mpnet" loads retrieval_candidates_mpnet_<dataset>.json (the primary retriever).
        "minilm" loads retrieval_candidates_<dataset>.json (the lighter comparison model).
    ...
    """
    #locating the candidates file saved by Step 3, choosing the retriever's file
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    if retriever_name == "mpnet":
        candidates_filename = f"retrieval_candidates_mpnet_{dataset_name}.json"
    else:
        candidates_filename = f"retrieval_candidates_{dataset_name}.json"
    candidates_path = os.path.join(results_dir, candidates_filename)
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(
            f"No saved candidates at {candidates_path}. "
            f"Run Step 3 (retrieval.py --dataset {dataset_name}) first."
        )

    #loading the saved candidates
    with open(candidates_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    #selecting the chosen retriever's candidates (dense by default)
    retriever_candidates = saved[retriever]

    #rebuilding each candidate into the format rerank() expects:
    #it needs doc_id, text (from corpus), score, and rank (from position)
    rebuilt = {}
    missing_docs = 0
    for claim_id, docs in retriever_candidates.items():
        rebuilt_docs = []
        for position, doc in enumerate(docs, start=1):
            doc_id = doc["doc_id"]
            #looking up the document text from the corpus by doc_id
            text = corpus.get(doc_id)
            if text is None:
                #a saved doc_id not present in the corpus -- skip it and count it
                missing_docs += 1
                continue
            rebuilt_docs.append({
                "doc_id": doc_id,
                "text": text,
                "score": doc["score"],
                "rank": position,
            })
        rebuilt[claim_id] = rebuilt_docs

    if missing_docs > 0:
        print(f"  Warning: {missing_docs} saved candidate doc_ids not found in corpus (skipped).")

    return rebuilt

# ---------------------------------------------------------------------------
# Helper function for hard filtering
# ---------------------------------------------------------------------------

def apply_mode(reranked_docs, mode):
    """
    Applying the chosen reranking mode to an already-scored, reranked doc list.
      soft -> keep all documents, just reordered by stance (no removal)
      hard -> remove documents flagged would_be_filtered (neutral above threshold)
    Both use the SAME NLI scores, so the comparison is fair -- only removal differs.
    Hard filtering can only lower or maintain recall (it only removes docs).
    """
    if mode == "hard":
        return [doc for doc in reranked_docs if not doc["would_be_filtered"]]
    #soft: keep everything as-is
    return reranked_docs

# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_reranking_evaluation(dataset_name="scifact", mode="soft"):
    """
    Running the full stance-aware reranking evaluation pipeline.

    Loads the dataset, builds dense retrieval index, retrieves candidates,
    applies stance reranking at two thresholds, and reports Recall@k
    before and after reranking. Also logs average documents surviving each filter.

    dataset_name: str -- 'scifact' or 'scifact_open'

    Returns: dict with recall numbers for thesis tables

    dataset_name: str -- 'scifact' or 'scifact_open'
    mode: str -- 'soft' (rerank, keep all docs) or 'hard' (remove would_be_filtered docs)
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

    #printing which device will be used  
    print(f"Using device: {device}\n")

    #loading the validation split and corpus for the chosen dataset
    if dataset_name == "scifact":
        val_claims, corpus = load_scifact(split="validation")
    elif dataset_name == "scifact_open":
        val_claims, corpus = load_scifact_open(corpus_file="full")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'scifact' or 'scifact_open'.")

    #printing basic corpus and claim statistics
    print(f"Validation claims : {len(val_claims)}")
    print(f"Corpus size       : {len(corpus)} documents\n")

    #filtering to claims with annotated evidence for recall computation
    claims_with_evidence = [c for c in val_claims if c["evidence_doc_ids"]]
    print(f"Claims with evidence (non-NEI): {len(claims_with_evidence)}")
    print(f"NEI claims (excluded from Recall@k): {len(val_claims) - len(claims_with_evidence)}\n")

    print("--- Loading saved dense retrieval candidates from Step 3 ---")
    #loading candidates saved by retrieval.py instead of re-encoding 500K docs.
    #The corpus (loaded above) is used only to resolve doc_id -> document text.
    #For MiniLM
    #dense_retrieved = load_saved_candidates(dataset_name, corpus, retriever="dense")
    #using the mpnet retriever candidates (the primary retriever selected in Step 3)
    #For MPNet
    dense_retrieved = load_saved_candidates(dataset_name, corpus, retriever="dense", retriever_name="mpnet")
    print(f"Loaded candidates for {len(dense_retrieved)} claims.\n")

    k_values = [1, 5, 10]
    dense_recall_before = compute_recall_at_k(val_claims, dense_retrieved, k_values)

    print("\n--- Stance-Aware Reranking ---")
    stance_reranker = StanceReranker(device=device)

    print(f"\nApplying loose threshold (neutral <= {LOOSE_NEUTRAL_THRESHOLD})...")
    loose_reranked = {}
    #accumulating filter counts from the FULL reranked list, before apply_mode trims it,
    #so the statistic is correct in hard mode too (where docs are actually removed)
    total_would_filter_loose = 0
    for claim_index, claim in enumerate(val_claims):
        if (claim_index + 1) % 100 == 1:
            print(f"  Reranking claim {claim_index + 1} / {len(val_claims)}...")
        candidates = dense_retrieved.get(claim["id"], [])
        reranked = stance_reranker.rerank(
            claim["claim"], candidates, LOOSE_NEUTRAL_THRESHOLD
        )
        #counting filtered docs on the FULL list BEFORE applying the mode
        total_would_filter_loose += sum(1 for doc in reranked if doc["would_be_filtered"])
        #applying the chosen mode (soft keeps all, hard removes would_be_filtered)
        loose_reranked[claim["id"]] = apply_mode(reranked, mode)

    loose_recall = compute_recall_at_k(val_claims, loose_reranked, k_values)

    #average docs the loose threshold would filter (identical across modes -- it's a
    #property of the threshold, computed before removal)
    average_docs_would_filter_loose = total_would_filter_loose / len(val_claims)

    print(f"\nApplying strict threshold (neutral <= {STRICT_NEUTRAL_THRESHOLD})...")
    strict_reranked = {}
    total_would_filter_strict = 0
    for claim_index, claim in enumerate(val_claims):
        if (claim_index + 1) % 100 == 1:
            print(f"  Reranking claim {claim_index + 1} / {len(val_claims)}...")
        candidates = dense_retrieved.get(claim["id"], [])
        reranked = stance_reranker.rerank(
            claim["claim"], candidates, STRICT_NEUTRAL_THRESHOLD
        )
        #counting filtered docs on the FULL list BEFORE applying the mode
        total_would_filter_strict += sum(1 for doc in reranked if doc["would_be_filtered"])
        strict_reranked[claim["id"]] = apply_mode(reranked, mode)

    strict_recall = compute_recall_at_k(val_claims, strict_reranked, k_values)

    #average docs the strict threshold would filter (computed before removal)
    average_docs_would_filter_strict = total_would_filter_strict / len(val_claims)

    #average docs actually SURVIVING after apply_mode -- differs by mode:
    #soft keeps all (== RETRIEVAL_K), hard removes filtered docs (fewer). This is the
    #number that makes hard-mode over-filtering visible when comparing the two runs.
    average_docs_surviving_loose = (
        sum(len(docs) for docs in loose_reranked.values()) / len(val_claims)
    )
    average_docs_surviving_strict = (
        sum(len(docs) for docs in strict_reranked.values()) / len(val_claims)
    )

    print(f"\n{'='*60}")
    print(f"  Reranking Recall@k -- {dataset_name.upper()} -- MODE: {mode.upper()}")
    print(f"{'='*60}")
    print(f"\n  {'Method':<35} {'R@1':>6} {'R@5':>6} {'R@10':>6}")  
    print(f"  {'-'*55}")  
    print(f"  {'Dense (before reranking)':<35} "  
          f"{dense_recall_before[1]:>6.3f} "
          f"{dense_recall_before[5]:>6.3f} "
          f"{dense_recall_before[10]:>6.3f}")
    print(f"  {f'Dense + stance rerank {mode} (loose)':<35} "
          f"{loose_recall[1]:>6.3f} "
          f"{loose_recall[5]:>6.3f} "
          f"{loose_recall[10]:>6.3f}")
    print(f"  {f'Dense + stance rerank {mode} (strict)':<35} "
          f"{strict_recall[1]:>6.3f} "
          f"{strict_recall[5]:>6.3f} "
          f"{strict_recall[10]:>6.3f}")
    print(f"{'='*60}")  

    #printing filter survival statistics
    print(f"\n  Avg docs retrieved            : {RETRIEVAL_K:.1f}")
    print(f"  Avg docs would filter loose   : {average_docs_would_filter_loose:.1f}")
    print(f"  Avg docs would filter strict  : {average_docs_would_filter_strict:.1f}")
    print(f"  Avg docs SURVIVING loose ({mode:<4}): {average_docs_surviving_loose:.1f}")
    print(f"  Avg docs SURVIVING strict ({mode:<4}): {average_docs_surviving_strict:.1f}\n")

    results = {
        "dataset": dataset_name,
        "mode": mode,
        "retrieval_k": RETRIEVAL_K,
        "loose_neutral_threshold": LOOSE_NEUTRAL_THRESHOLD,
        "strict_neutral_threshold": STRICT_NEUTRAL_THRESHOLD,
        "avg_docs_retrieved": RETRIEVAL_K,         
        "avg_docs_would_filter_loose": average_docs_would_filter_loose,
        "avg_docs_would_filter_strict": average_docs_would_filter_strict,
        "avg_docs_surviving_loose": average_docs_surviving_loose,
        "avg_docs_surviving_strict": average_docs_surviving_strict,
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

    #saving reranking results to disk for the thesis and later analysis
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    #For MiniLM
    #results_path = os.path.join(results_dir, f"reranking_{mode}_{dataset_name}.json")
    #For MPNet
    results_path = os.path.join(results_dir, f"reranking_mpnet_{mode}_{dataset_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Reranking results saved to {results_path}")

    #saving the actual reranked candidates so Step 5 (verification) can consume them
    #without re-running reranking. Saved per mode+dataset so runs don't overwrite.
    def _slim_candidates(reranked_dict):
        return {
            claim_id: [
                {
                    "doc_id": d["doc_id"],
                    "stance_score": d["stance_score"],
                    "entailment_score": d["entailment_score"],
                    "contradiction_score": d["contradiction_score"],
                    "neutral_score": d["neutral_score"],
                    "predicted_label": d["predicted_label"],
                    "reranked_position": d["reranked_position"],
                }
                for d in docs
            ]
            for claim_id, docs in reranked_dict.items()
        }

    candidates_out = {
        "dataset": dataset_name,
        "mode": mode,
        "loose": _slim_candidates(loose_reranked),
        "strict": _slim_candidates(strict_reranked),
    }
    #For MiniLM
    #candidates_path = os.path.join(results_dir, f"reranked_candidates_{mode}_{dataset_name}.json")
    #For MPNet
    candidates_path = os.path.join(results_dir, f"reranked_candidates_mpnet_{mode}_{dataset_name}.json")
    with open(candidates_path, "w") as f:
        json.dump(candidates_out, f, indent=2)
    print(f"Reranked candidates saved to {candidates_path}")

    return results

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    #parsing command line arguments so we can specify the dataset without editing the file
    parser = argparse.ArgumentParser(description="Stance-aware reranking evaluation")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "scifact_open"],
                        help="Dataset to run reranking evaluation on")
    parser.add_argument("--mode", default="soft", choices=["soft", "hard"],
                        help="soft = rerank keep all docs; hard = remove would_be_filtered docs")
    args = parser.parse_args()

    #running the reranking evaluation in the chosen mode
    run_reranking_evaluation(dataset_name=args.dataset, mode=args.mode)
"""
BM25 and dense retrieval for claim verification

Implements two standard retrieval methods used throughout this project:

    1. BM25 (sparse retrieval): keyword-based, using rank-bm25
    2. Dense retrieval: semantic similarity using sentence-transformers

For each claim, both methods retrieve a ranked list of candidate
documents from the evidence corpus. These candidates are then passed
to the stance-aware reranker in Step 4.

Why these two methods?
- BM25 is the standard lexical baseline in information retrieval.
  It is well understood, fast, and requires no training.
- Dense retrieval captures semantic similarity beyond keyword overlap,
  which is important for scientific claims that use varied vocabulary.
- Comparing both lets us isolate the effect of retrieval method
  independently of the stance reranking step.

References:
- Robertson & Zaragoza (2009) -- The Probabilistic Relevance Framework: BM25
- Reimers & Gurevych (2019) -- Sentence-BERT: Sentence Embeddings using Siamese Networks
"""

#importing the operating system module for path handling
import os

#importing the system module to add the project root to the Python path
import sys

#adding the project root to the path so we can import from data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing PyTorch for tensor operations during dense retrieval
import torch

#importing numpy for array operations used in similarity computation
import numpy as np

#importing the BM25 implementation from the rank-bm25 library
from rank_bm25 import BM25Okapi

#importing the SentenceTransformer model for dense retrieval embeddings
from sentence_transformers import SentenceTransformer

#importing our unified data loading functions
from data.utils import load_scifact, load_scifact_open

#importing Counter for corpus statistics
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#defining the sentence-transformer model used for dense retrieval
#all-MiniLM-L6-v2 is fast, lightweight, and strong for semantic similarity
#DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#for later using a bigger model
DENSE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

#defining the default number of documents to retrieve per claim
DEFAULT_K = 10

#defining the batch size for encoding documents during dense retrieval
#larger batches are faster but use more GPU memory
ENCODING_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Implementing BM25 sparse retrieval over a document corpus.

    BM25 tokenises both queries (claims) and documents by whitespace,
    then ranks documents by term frequency and inverse document frequency.
    This is the standard lexical baseline for information retrieval tasks.
    """

    def __init__(self, corpus):
        """
        Building the BM25 index from the provided corpus.

        corpus: dict {doc_id (str): document_text (str)}
        """

        #storing the corpus as a list of (doc_id, text) pairs for index alignment
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        #printing how many documents are being indexed
        print(f"Building BM25 index over {len(self.doc_ids)} documents...")

        #tokenising each document by whitespace for BM25 indexing
        tokenised_corpus = [text.lower().split() for text in self.doc_texts]

        #building the BM25 index from the tokenised corpus
        self.bm25_index = BM25Okapi(tokenised_corpus)

        #printing confirmation that the index is ready
        print(f"BM25 index built successfully.")

    def retrieve(self, claim_text, k=DEFAULT_K):
        """
        Retrieving the top-k documents most relevant to the given claim.

        claim_text: str -- the claim to retrieve evidence for
        k: int -- number of documents to return

        Returns: list of dicts with keys:
            doc_id  (str)   the document identifier
            text    (str)   the document text
            score   (float) the BM25 relevance score
            rank    (int)   rank position (1 = most relevant)
        """

        #tokenising the claim text by whitespace to match the index format
        tokenised_query = claim_text.lower().split()

        #computing BM25 scores for all documents in the corpus
        scores = self.bm25_index.get_scores(tokenised_query)

        #getting the indices of the top-k scoring documents
        top_k_indices = np.argsort(scores)[::-1][:k]

        #building the result list with doc_id, text, score, and rank
        results = []
        for rank, index in enumerate(top_k_indices, start=1):
            results.append({
                "doc_id": self.doc_ids[index],
                "text":   self.doc_texts[index],
                "score":  float(scores[index]),
                "rank":   rank,
            })

        #returning the ranked list of retrieved documents
        return results


# ---------------------------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Implementing dense retrieval using sentence-transformer embeddings.

    All documents in the corpus are encoded into dense vectors upfront.
    At retrieval time, the claim is encoded and cosine similarity is
    computed against all document vectors to find the most relevant ones.

    This captures semantic similarity beyond keyword overlap, which is
    important for scientific claims where exact term matching is unreliable.
    """

    def __init__(self, corpus, device=None):
        """
        Loading the sentence-transformer model and encoding the full corpus.

        corpus: dict {doc_id (str): document_text (str)}
        device: torch.device or None (auto-detects if None)
        """

        #storing the corpus doc ids and texts for index alignment
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        #auto-detecting the best available device if none is specified
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        #printing which device will be used for encoding
        print(f"Loading dense retrieval model: {DENSE_MODEL_NAME}")
        print(f"Using device: {self.device}")

        #loading the sentence-transformer model
        self.model = SentenceTransformer(DENSE_MODEL_NAME, device=self.device)

        #encoding all corpus documents into dense vectors upfront
        print(f"Encoding {len(self.doc_texts)} documents into dense vectors...")
        print(f"(This happens once per corpus -- subsequent retrievals are fast)")

        #encoding in batches for memory efficiency
        self.document_embeddings = self.model.encode(
            self.doc_texts,
            batch_size=ENCODING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        #printing confirmation that encoding is complete
        print(f"Corpus encoded. Embedding matrix shape: {self.document_embeddings.shape}")

    def retrieve(self, claim_text, k=DEFAULT_K):
        """
        Retrieving the top-k documents most semantically similar to the claim.

        claim_text: str -- the claim to retrieve evidence for
        k: int -- number of documents to return

        Returns: list of dicts with keys:
            doc_id  (str)   the document identifier
            text    (str)   the document text
            score   (float) cosine similarity score (0 to 1)
            rank    (int)   rank position (1 = most similar)
        """

        #encoding the claim into a dense vector on the same device as the corpus embeddings
        #normalize_embeddings=True means cosine similarity = dot product
        claim_embedding = self.model.encode(
            claim_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )

        #computing cosine similarity between the claim and all documents
        cosine_similarities = np.dot(self.document_embeddings, claim_embedding)

        #getting the indices of the top-k most similar documents
        top_k_indices = np.argsort(cosine_similarities)[::-1][:k]

        #building the result list with doc_id, text, score, and rank
        results = []
        for rank, index in enumerate(top_k_indices, start=1):
            results.append({
                "doc_id": self.doc_ids[index],
                "text":   self.doc_texts[index],
                "score":  float(cosine_similarities[index]),
                "rank":   rank,
            })

        #returning the ranked list of retrieved documents
        return results


# ---------------------------------------------------------------------------
# Retrieval evaluation helpers
# ---------------------------------------------------------------------------

def compute_recall_at_k(claims, retrieval_results, k):
    """
    Computing Recall@k -- the fraction of claims for which at least one
    gold evidence document appears in the top-k retrieved documents.

    This is the standard retrieval evaluation metric for fact-checking.
    A claim with no annotated evidence (NEI) is excluded from the calculation.

    claims: list of claim dicts (from data.utils)
    retrieval_results: dict {claim_id: list of retrieved doc dicts}
    k: int -- cutoff for evaluation
    """

    #initialising counters for hits and total evaluated claims
    total_evaluated = 0
    total_hits = 0

    #iterating over each claim to check if evidence was retrieved
    for claim in claims:

        #skipping NEI claims that have no annotated evidence documents
        if not claim["evidence_doc_ids"]:
            continue

        #getting the top-k retrieved doc ids for this claim
        retrieved_doc_ids = [
            result["doc_id"]
            for result in retrieval_results[claim["id"]][:k]
        ]

        #checking if any gold evidence document appears in the retrieved set
        gold_doc_ids = set(claim["evidence_doc_ids"])
        if gold_doc_ids & set(retrieved_doc_ids):
            total_hits += 1

        #counting this claim as evaluated
        total_evaluated += 1

    #returning recall as a fraction, handling the edge case of no evaluated claims
    if total_evaluated == 0:
        return 0.0
    return total_hits / total_evaluated


def run_retrieval_on_split(claims, retriever, k=DEFAULT_K):
    """
    Running retrieval for every claim in a split and returning results.

    claims: list of claim dicts
    retriever: BM25Retriever or DenseRetriever instance
    k: int as number of documents to retrieve per claim

    Returns: dict {claim_id (str): list of retrieved doc dicts}
    """

    #initialising the results dictionary
    retrieval_results = {}

    #iterating over each claim and retrieving top-k documents
    for i, claim in enumerate(claims):

        #printing progress every 100 claims so we can track long runs
        if i % 100 == 0:
            print(f"  Retrieving for claim {i+1} / {len(claims)}...")

        #running retrieval for this claim
        retrieved = retriever.retrieve(claim["claim"], k=k)

        #storing the results keyed by claim id
        retrieval_results[claim["id"]] = retrieved

    #returning the complete retrieval results
    return retrieval_results


# ---------------------------------------------------------------------------
# Sanity check and evaluation entry point
# ---------------------------------------------------------------------------

def run_retrieval_evaluation(dataset_name="scifact"):
    """
    Running BM25 and dense retrieval on the validation split of a dataset,
    computing Recall@k for k in {1, 5, 10}, and printing a comparison table.

    This gives us the retrieval quality before any stance reranking is applied.
    dataset_name: 'scifact' or 'scifact_open'
    """

    #printing a header for the evaluation run
    print(f"\n{'=' * 60}")
    print(f"  Retrieval Evaluation -- {dataset_name.upper()}")
    print(f"{'=' * 60}\n")

    #loading the claims and retrieval corpus for the chosen dataset
    if dataset_name == "scifact":
        #SciFact: evaluate on the validation split, retrieve over its ~5K corpus
        eval_claims, corpus = load_scifact(split="validation")
    elif dataset_name == "scifact_open":
        #SciFact-Open: test-only collection, no split argument.
        #load_scifact_open returns (claims, corpus) where corpus is the full 500K set.
        eval_claims, corpus = load_scifact_open(corpus_file="full")
    else:
        #raising an error for unrecognised dataset names
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'scifact' or 'scifact_open'.")

    #printing basic corpus and claim statistics
    print(f"Validation claims : {len(eval_claims)}")
    print(f"Corpus size       : {len(corpus)} documents\n")

    #counting how many claims have annotated evidence (non-NEI)
    claims_with_evidence = [c for c in eval_claims if c["evidence_doc_ids"]]
    print(f"Claims with evidence (non-NEI): {len(claims_with_evidence)}")
    print(f"NEI claims (excluded from Recall@k): {len(eval_claims) - len(claims_with_evidence)}\n")

    # ---------------------------------------------------------------------------
    # BM25 retrieval evaluation
    # ---------------------------------------------------------------------------

    #building the BM25 index over the full corpus
    print("--- BM25 Retrieval ---")
    bm25_retriever = BM25Retriever(corpus)

    #running BM25 retrieval for all validation claims at k=10
    print("\nRunning BM25 retrieval...")
    bm25_results = run_retrieval_on_split(eval_claims, bm25_retriever, k=10)

    #computing Recall@k for k in {1, 5, 10}
    bm25_recall_at_1  = compute_recall_at_k(eval_claims, bm25_results, k=1)
    bm25_recall_at_5  = compute_recall_at_k(eval_claims, bm25_results, k=5)
    bm25_recall_at_10 = compute_recall_at_k(eval_claims, bm25_results, k=10)

    # ---------------------------------------------------------------------------
    # Dense retrieval evaluation
    # ---------------------------------------------------------------------------

    #building the dense retriever by encoding the full corpus
    print("\n--- Dense Retrieval ---")
    dense_retriever = DenseRetriever(corpus)

    #running dense retrieval for all validation claims at k=10
    print("\nRunning dense retrieval...")
    dense_results = run_retrieval_on_split(eval_claims, dense_retriever, k=10)

    #computing Recall@k for k in {1, 5, 10}
    dense_recall_at_1  = compute_recall_at_k(eval_claims, dense_results, k=1)
    dense_recall_at_5  = compute_recall_at_k(eval_claims, dense_results, k=5)
    dense_recall_at_10 = compute_recall_at_k(eval_claims, dense_results, k=10)

    # ---------------------------------------------------------------------------
    # Printing the comparison table
    # ---------------------------------------------------------------------------

    print(f"\n{'=' * 50}")
    print(f"  Retrieval Recall@k -- {dataset_name.upper()} (validation)")
    print(f"{'=' * 50}")
    print(f"  {'Method':<20} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print(f"  {'-'*38}")
    print(f"  {'BM25':<20} {bm25_recall_at_1:>6.3f} {bm25_recall_at_5:>6.3f} {bm25_recall_at_10:>6.3f}")
    print(f"  {'Dense':<20} {dense_recall_at_1:>6.3f} {dense_recall_at_5:>6.3f} {dense_recall_at_10:>6.3f}")
    print(f"{'=' * 50}\n")

    # ---------------------------------------------------------------------------
    # Saving results to disk
    # ---------------------------------------------------------------------------
    import json

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    #saving the Recall@k numbers (the Step 3 thesis result) -- small, human-readable
    recall_summary = {
        "dataset": dataset_name,
        "corpus_size": len(corpus),
        "num_claims": len(eval_claims),
        "num_claims_with_evidence": len(claims_with_evidence),
        "recall_at_k": {
            "bm25":  {"1": bm25_recall_at_1,  "5": bm25_recall_at_5,  "10": bm25_recall_at_10},
            "dense": {"1": dense_recall_at_1, "5": dense_recall_at_5, "10": dense_recall_at_10},
        },
    }
    #recall_path = os.path.join(results_dir, f"retrieval_recall_{dataset_name}.json")
    recall_path = os.path.join(results_dir, f"retrieval_recall_mpnet_{dataset_name}.json")
    with open(recall_path, "w") as f:
        json.dump(recall_summary, f, indent=2)
    print(f"Recall@k summary saved to {recall_path}")

    #saving the retrieved candidates (the INPUT to Step 4's stance reranker) so the
    #expensive retrieval never has to be re-run. We save doc_id + score per candidate.
    def _slim(results):
        #keeping only what Step 4 needs: for each claim, its ranked (doc_id, score) list
        return {
            claim_id: [{"doc_id": r["doc_id"], "score": float(r["score"])} for r in docs]
            for claim_id, docs in results.items()
        }

    candidates = {
        "dataset": dataset_name,
        "bm25":  _slim(bm25_results),
        "dense": _slim(dense_results),
    }
    #candidates_path = os.path.join(results_dir, f"retrieval_candidates_{dataset_name}.json")
    candidates_path = os.path.join(results_dir, f"retrieval_candidates_mpnet_{dataset_name}.json")
    with open(candidates_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Retrieved candidates saved to {candidates_path}")

    #returning results and Recall@k numbers for use in downstream steps and thesis tables
    return {
        "bm25": bm25_results,
        "dense": dense_results,
        "bm25_retriever": bm25_retriever,
        "dense_retriever": dense_retriever,
        "recall_at_k": {
            "bm25":  {1: bm25_recall_at_1,  5: bm25_recall_at_5,  10: bm25_recall_at_10},
            "dense": {1: dense_recall_at_1, 5: dense_recall_at_5, 10: dense_recall_at_10},
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    #parsing command line arguments so we can specify the dataset without editing the file
    parser = argparse.ArgumentParser(description="BM25 and dense retrieval evaluation")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "scifact_open"],
                        help="Dataset to run retrieval on")
    args = parser.parse_args()

    #running retrieval evaluation on the chosen dataset
    run_retrieval_evaluation(dataset_name=args.dataset)
"""
baseline.py: RoBERTa no-retrieval baseline model

Trains and evaluates a RoBERTa classifier on claim text alone,
with no retrieved evidence whatsoever.

This answers the question: what can the model predict using only
its pre-trained knowledge, before any retrieval is introduced?

Every downstream RAG pipeline result is compared against this.

Design decisions (document these in your thesis):
- roberta-base: strong enough to be meaningful, standard for NLP classification
- AdamW with weight_decay=0.01: standard for transformer fine-tuning (Loshchilov & Hutter, 2019)
- Gradient clipping at 1.0: prevents exploding gradients, standard practice
- 10% warmup steps: well-established heuristic for transformer schedulers
- Macro F1 as primary metric: correct for imbalanced 3-class problems
- Learning rate search over {1e-5, 2e-5, 3e-5}: we pick the best on val F1
- Early stopping with patience 2: saves best checkpoint, avoids overfitting on small data
- Shared 3-class head (SUPPORT/CONTRADICT/NEI): SciFact is 3-class, but the secondary
  evaluation dataset SciFact-Open is 2-class (SUPPORT/CONTRADICT, no NEI). We keep one
  unified head so the same SciFact-trained classifier can be scored on both. Metrics are
  computed only over classes actually present in each split, so SciFact-Open's scores are
  not distorted by the unused NEI class. On SciFact-Open the model can still predict NEI,
  but since no gold label is NEI there, such predictions are simply counted as errors.
"""

#importing the operating system module for handling file paths
import os

#importing the system module so we can add the project root to the Python path
import sys

#adding the project root directory to the path so imports from data/ work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#importing PyTorch as the core deep learning framework
import torch

#importing Dataset and DataLoader to handle batching and shuffling of training data
from torch.utils.data import Dataset, DataLoader

#importing the RoBERTa tokenizer that converts text into token ids
from transformers import RobertaTokenizer

#importing the RoBERTa model with a sequence classification head for 3-class prediction
from transformers import RobertaForSequenceClassification

#importing the learning rate scheduler that does linear warmup then decay
from transformers import get_linear_schedule_with_warmup

#importing the AdamW optimiser which is standard for fine-tuning transformer models
from torch.optim import AdamW

#importing metric functions for evaluating model performance
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

#importing Counter to count label distributions in the training data
from collections import Counter

#importing numpy so we can seed it for reproducibility
import numpy as np

#importing random so we can seed Python's RNG (used by random.sample in the LR search)
import random

#importing our unified data loading functions for both datasets
from data.utils import load_scifact, load_scifact_open, LABEL_SUPPORT, LABEL_CONTRADICT, LABEL_NEI

import json
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#defining the name of the pre-trained RoBERTa model we are fine-tuning
MODEL_NAME = "roberta-base"

#setting the maximum number of tokens RoBERTa processes per input claim
#claim-only inputs are short (max 75 tokens), but claim+evidence pairs are long,
#so evidence mode uses a larger max length
MAX_LENGTH_CLAIM_ONLY = 128
MAX_LENGTH_CLAIM_EVIDENCE = 512

#setting value of how many examples are processed together in one forward pass
BATCH_SIZE = 16

#setting the maximum number of epochs -- early stopping may end training earlier
MAX_EPOCHS = 10

#defining how many epochs without improvement before we stop training early
EARLY_STOPPING_PATIENCE = 2

#setting the learning rates we search over -- we pick the best on validation F1
LEARNING_RATES_TO_TRY = [1e-5, 2e-5, 3e-5]

#setting the ordered list of label names used consistently across the project
LABEL_LIST = [LABEL_SUPPORT, LABEL_CONTRADICT, LABEL_NEI]

#creating a dictionary mapping each label string to its integer index for the model
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_LIST)}

#creating a dictionary mapping each integer index back to its label string for reporting
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    """
    Seeding all random number generators so results are reproducible.
    Documented in the thesis: all reported numbers use seed=42 unless stated.
    """
    #seeding Python's built-in random module (used by random.sample in LR search)
    random.seed(seed)

    #seeding NumPy's RNG
    np.random.seed(seed)

    #seeding PyTorch's CPU RNG
    torch.manual_seed(seed)

    #seeding PyTorch's GPU RNG for all CUDA devices, if any
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Token length check
# ---------------------------------------------------------------------------

def check_max_token_length(claims, tokenizer):
    """
    Checking the actual maximum token length across all claims.
    This confirms that MAX_LENGTH_CLAIM_ONLY = 128 does not truncate any claims,
    which is important to document in the thesis.
    """

    #initialising the maximum token count seen so far
    maximum_token_count = 0

    #iterating over every claim to measure its token length
    for claim_dict in claims:

        #skipping claims with empty or None text to avoid tokenizer errors
        if not claim_dict["claim"]:
            continue

        #tokenising this claim without padding or truncation to get true length
        tokens = tokenizer(claim_dict["claim"], truncation=False)

        #updating the maximum if this claim is longer than any seen so far
        token_count = len(tokens["input_ids"])
        if token_count > maximum_token_count:
            maximum_token_count = token_count

    #printing the result so we can verify MAX_LENGTH is safe
    print(f"  Maximum claim token length in this split: {maximum_token_count}")
    if maximum_token_count > MAX_LENGTH_CLAIM_ONLY:
        print(f"  WARNING: some claims exceed MAX_LENGTH_CLAIM_ONLY={MAX_LENGTH_CLAIM_ONLY} and will be truncated!")
    else:
        print(f"  MAX_LENGTH_CLAIM_ONLY={MAX_LENGTH_CLAIM_ONLY} is safe -- no claims will be truncated.")

    #returning the maximum for logging
    return maximum_token_count


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ClaimDataset(Dataset):
    """
    Wrapping a list of claim dicts into a PyTorch Dataset.

    Each claim dict has keys: id, claim, label, evidence_doc_ids.
    This class tokenises the claim text and returns tensors that
    PyTorch DataLoader can batch together for training and evaluation.
    """

    def __init__(self, list_of_claim_dicts, tokenizer, input_mode="claim_only", corpus=None):
        #storing the list of claim dicts so we can index into them
        self.claims = list_of_claim_dicts
        #storing the tokenizer so we can encode each claim at access time
        self.tokenizer = tokenizer
        #input_mode: "claim_only" (baseline) or "claim_evidence" (evidence-aware RAG classifier)
        self.input_mode = input_mode
        #corpus {doc_id: text} -- required for claim_evidence mode to look up gold evidence
        self.corpus = corpus
        #choosing max length based on mode (evidence pairs need more room)
        self.max_length = (MAX_LENGTH_CLAIM_EVIDENCE if input_mode == "claim_evidence"
                           else MAX_LENGTH_CLAIM_ONLY)
        if input_mode == "claim_evidence" and corpus is None:
            raise ValueError("claim_evidence mode requires a corpus to look up evidence text.")

    def __len__(self):
        #returning the total number of claims in this dataset split
        return len(self.claims)

    def __getitem__(self, index):
        #retrieving the claim dict at the given index position
        claim_dict = self.claims[index]

        if self.input_mode == "claim_evidence":
            #looking up the gold evidence text for this claim from the corpus.
            #a claim may cite multiple evidence docs; we concatenate their text.
            evidence_texts = []
            for doc_id in claim_dict["evidence_doc_ids"]:
                doc_text = self.corpus.get(str(doc_id))
                if doc_text:
                    evidence_texts.append(doc_text)
            evidence_text = " ".join(evidence_texts)

            #tokenising claim + gold evidence as a PROPER PAIR (RoBERTa inserts </s></s>).
            #NEI claims have no evidence_doc_ids, so evidence_text is "" -- the model
            #then sees claim + empty evidence, which is the correct signal for "no evidence".
            encoding = self.tokenizer(
                claim_dict["claim"],
                evidence_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            #claim-only tokenisation (the original baseline behaviour)
            encoding = self.tokenizer(
                claim_dict["claim"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        #converting the label string to its corresponding integer id
        label_id = LABEL_TO_ID[claim_dict["label"]]

        #returning a dict of tensors -- squeezing removes the extra batch dimension added by return_tensors="pt"
        return {
            #squeezing the input ids tensor from shape (1, max_length) to (max_length,) to (MAX_LENGTH_CLAIM_ONLY,)
            "input_ids": encoding["input_ids"].squeeze(),

            #squeezing the attention mask tensor from shape (1, max_length) to (max_length,) to (MAX_LENGTH_CLAIM_ONLY,)
            "attention_mask": encoding["attention_mask"].squeeze(),

            #converting the label integer to a long tensor as required by PyTorch loss functions
            "label": torch.tensor(label_id, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimiser, scheduler, device):
    """
    Running one full pass through the training data and updating model weights.
    Returns the average cross-entropy loss across all batches in this epoch.
    """

    #setting the model to training mode so dropout layers are active
    model.train()

    #initialising the running total of loss across all batches
    total_loss = 0.0

    #iterating over each batch from the training dataloader
    for batch in dataloader:

        #moving input ids to the correct device (MPS, CUDA, or CPU)
        input_ids = batch["input_ids"].to(device)

        #moving attention mask to the correct device
        attention_mask = batch["attention_mask"].to(device)

        #moving labels to the correct device
        labels = batch["label"].to(device)

        #clearing gradients accumulated from the previous training step
        optimiser.zero_grad()

        #running the forward pass through RoBERTa to get loss and logits
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        #extracting the cross-entropy loss computed internally by the model
        loss = outputs.loss

        #running backpropagation to compute gradients for all parameters
        loss.backward()

        #clipping gradients to a maximum norm of 1.0 to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #updating model weights using the computed gradients
        optimiser.step()

        #advancing the learning rate scheduler by one step
        scheduler.step()

        #adding this batch loss to the running total
        total_loss += loss.item()

    #returning the average loss across all batches in this epoch
    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device):
    """
    Running inference on a dataset split without updating model weights.
    Returns macro F1, macro precision, macro recall, and a full
    per-class classification report string.
    """

    #setting the model to evaluation mode so dropout is disabled
    model.eval()

    #initialising empty lists to collect all true labels
    all_true_labels = []

    #initialising empty lists to collect all predicted labels
    all_predicted_labels = []

    #disabling gradient computation during evaluation to save memory
    with torch.no_grad():

        #iterating over each batch from the evaluation dataloader
        for batch in dataloader:

            #moving input ids to the correct device
            input_ids = batch["input_ids"].to(device)

            #moving attention mask to the correct device
            attention_mask = batch["attention_mask"].to(device)

            #moving labels to the correct device
            labels = batch["label"].to(device)

            #running the forward pass to get output logits
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            #taking the index of the highest logit as the predicted class
            predictions = torch.argmax(outputs.logits, dim=-1)

            #collecting true labels as a Python list for metric computation
            all_true_labels.extend(labels.cpu().numpy().tolist())

            #collecting predicted labels as a Python list for metric computation
            all_predicted_labels.extend(predictions.cpu().numpy().tolist())

    #determining which label ids are actually present in this dataset split.
    #comes before the metric calls so they can scope to these labels.
    present_label_ids = sorted(set(all_true_labels))
    present_label_names = [ID_TO_LABEL[label_id] for label_id in present_label_ids]

    #computing macro-averaged F1 over only the classes present in the gold labels
    macro_f1 = f1_score(
        all_true_labels,
        all_predicted_labels,
        labels=present_label_ids,
        average="macro",
        zero_division=0,
    )

    #computing macro-averaged precision over only the classes present in the gold labels
    macro_precision = precision_score(
        all_true_labels,
        all_predicted_labels,
        labels=present_label_ids,
        average="macro",
        zero_division=0,
    )

    #computing macro-averaged recall over only the classes present in the gold labels
    macro_recall = recall_score(
        all_true_labels,
        all_predicted_labels,
        labels=present_label_ids,
        average="macro",
        zero_division=0,
    )

    # Generating a full per-class breakdown report using only the labels present
    report = classification_report(
        all_true_labels,
        all_predicted_labels,
        labels=present_label_ids,
        target_names=present_label_names,
        zero_division=0,
    )

    #returning all metrics and the detailed report
    return macro_f1, macro_precision, macro_recall, report


# ---------------------------------------------------------------------------
# Learning rate search -- trains with each LR and picks the best on val F1
# ---------------------------------------------------------------------------

def find_best_learning_rate(train_claims, eval_claims, tokenizer, device, input_mode="claim_only", corpus=None):
    """
    Searching over LEARNING_RATES_TO_TRY by training for 3 epochs each
    and returning the learning rate that gives the highest validation F1.
    This is a simple but defensible hyperparameter selection approach.
    """

    #printing a header for the learning rate search
    print("\n--- Learning rate search ---")

    #limiting the LR search to 2000 examples maximum to keep compute manageable on large datasets
    MAX_SEARCH_EXAMPLES = 2000
    if len(train_claims) > MAX_SEARCH_EXAMPLES:
        #sampling a random subset for the LR search only -- full data is used for actual training
        search_claims = random.sample(train_claims, MAX_SEARCH_EXAMPLES)
        print(f"  (Using {MAX_SEARCH_EXAMPLES} random examples for LR search on large dataset)")
    else:
        #using all training examples for the LR search on small datasets like SciFact
        search_claims = train_claims

    #initialising tracking variables for the best result found so far
    best_learning_rate = None
    best_val_f1_found = -1.0

    #wrapping search_claims (not all train_claims) in dataset and dataloader for this search
    search_dataset = ClaimDataset(search_claims, tokenizer, input_mode=input_mode, corpus=corpus)
    val_dataset = ClaimDataset(eval_claims, tokenizer, input_mode=input_mode, corpus=corpus)
    search_dataloader = DataLoader(search_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #iterating over each candidate learning rate
    for candidate_lr in LEARNING_RATES_TO_TRY:

        #reseeding before each trial so every learning rate starts from the SAME
        #random initialisation and data-shuffle path -- so the only thing differing
        #between trials is the learning rate itself (fair, defensible comparison)
        #seeding everything for reproducibility before any randomness happens
        set_seed(42)

        #printing which learning rate we are currently trying
        print(f"\n  Trying learning rate: {candidate_lr}")

        #loading a fresh copy of the model for each learning rate trial
        trial_model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL_LIST),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        ).to(device)

        #setting up the AdamW optimiser with this candidate learning rate
        trial_optimiser = AdamW(
            trial_model.parameters(),
            lr=candidate_lr,
            weight_decay=0.01,
        )

        #computing training steps for 3 trial epochs using the search dataloader
        trial_total_steps = len(search_dataloader) * 3
        trial_warmup_steps = int(0.1 * trial_total_steps)

        #setting up the learning rate scheduler for this trial
        trial_scheduler = get_linear_schedule_with_warmup(
            trial_optimiser,
            num_warmup_steps=trial_warmup_steps,
            num_training_steps=trial_total_steps,
        )

        #training for 3 epochs to get a representative validation F1
        for epoch_number in range(1, 4):
            #running one epoch of training on the search subset
            train_one_epoch(trial_model, search_dataloader, trial_optimiser, trial_scheduler, device)

        #evaluating on the full validation set after 3 trial epochs
        trial_val_f1, _, _, _ = evaluate(trial_model, val_dataloader, device)

        #printing the result for this learning rate
        print(f"  Val F1 at lr={candidate_lr}: {trial_val_f1:.4f}")

        #updating the best learning rate if this one performed better
        if trial_val_f1 > best_val_f1_found:
            best_val_f1_found = trial_val_f1
            best_learning_rate = candidate_lr

    #printing the chosen learning rate
    print(f"\n  Best learning rate: {best_learning_rate} (val F1 = {best_val_f1_found:.4f})")

    #returning the best learning rate for use in the full training run
    return best_learning_rate

# ---------------------------------------------------------------------------
# Main training and evaluation pipeline
# ---------------------------------------------------------------------------

def run_baseline(dataset_name="scifact", input_mode="claim_only", seed=42):
    """
    Loading SciFact, searching for the best learning rate, fine-tuning RoBERTa
    with early stopping on claim text only, evaluating on the validation split,
    and saving the best checkpoint.

    dataset_name: 'scifact' (SciFact-Open is evaluated separately, see
    evaluate_on_scifact_open, because it has no training split).
    """
    #seeding everything for reproducibility before any randomness happens
    set_seed(seed)

    #printing a clear header so output is easy to read in the terminal
    print(f"\n{'=' * 60}")
    mode_label = "No-Retrieval Baseline" if input_mode == "claim_only" else "Claim+Evidence Classifier"
    print(f"  RoBERTa {mode_label}  --  {dataset_name.upper()}")
    print(f"{'=' * 60}\n")

    #selecting MPS for Apple Silicon Macs, CUDA for NVIDIA GPUs, or CPU as fallback
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #printing which device will be used for training
    print(f"Using device: {device}\n")

    #loading the correct dataset based on the dataset_name argument
    if dataset_name == "scifact":
        #loading SciFact training claims with full corpus for evidence lookup in claim+evidence mode
        train_claims, corpus = load_scifact(split="train")

        #loading SciFact validation claims for evaluation after each epoch
        eval_claims, _ = load_scifact(split="validation")

        #SciFact's official test labels are withheld (blind leaderboard), so there is no
        #held-out test set; validation is the final reported split (see thesis)

    else:
        #raising an error if an unrecognised dataset name is given
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'scifact'.")

    #printing the number of training and validation claims
    print(f"Train claims : {len(train_claims)}")
    print(f"Val claims   : {len(eval_claims)}")

    #printing the label distribution to detect class imbalance early
    train_label_counts = Counter(c["label"] for c in train_claims)
    print(f"Train label distribution: {dict(train_label_counts)}\n")

    #loading the RoBERTa tokenizer from HuggingFace
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    #checking that MAX_LENGTH_CLAIM_ONLY is safe for the actual claim lengths in this dataset
    print("\nChecking claim token lengths...")
    if input_mode == "claim_only":
        check_max_token_length(train_claims, tokenizer)

    #searching for the best learning rate across our three candidates
    best_lr = find_best_learning_rate(train_claims, eval_claims, tokenizer, device, input_mode=input_mode, corpus=corpus)

    #loading a fresh model for the full training run with the best learning rate
    print(f"\n\nStarting full training run with best learning rate: {best_lr}\n")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    #moving the model to the selected device
    model = model.to(device)

    #wrapping the train and validation claims in dataset and dataloader objects
    train_dataset = ClaimDataset(train_claims, tokenizer, input_mode=input_mode, corpus=corpus)
    val_dataset = ClaimDataset(eval_claims, tokenizer, input_mode=input_mode, corpus=corpus)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #setting up AdamW optimiser with the best learning rate found above
    optimiser = AdamW(model.parameters(), lr=best_lr, weight_decay=0.01)
 
    '''
    The scheduler is planned over MAX_EPOCHS. If early stopping triggers earlier,
    training simply ends before the schedule completes, which is standard practice.
    '''
    #tying total scheduler steps to MAX_EPOCHS so the LR never decays to zero mid-training;
    #early stopping simply ends before the schedule completes, which is standard practice
    total_training_steps = len(train_dataloader) * MAX_EPOCHS

    #computing the number of warmup steps as 10% of total training steps
    warmup_steps = int(0.1 * total_training_steps)

    #setting up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # ---------------------------------------------------------------------------
    # Training loop with early stopping and best model saving
    # ---------------------------------------------------------------------------

    #initialising the best validation F1 seen so far for early stopping
    best_val_f1 = -1.0

    #initialising best precision/recall so they can be saved to JSON at the end
    best_val_precision = 0.0
    best_val_recall = 0.0

    #initialising the best validation report as an empty string -- updated when best F1 improves
    best_val_report = ""

    #initialising the counter tracking how many epochs have passed without improvement
    epochs_without_improvement = 0

    #constructing the save path for the best model checkpoint
    #naming the checkpoint by mode so claim_only (Model 1) and claim_evidence (Model 2)
    #to avoid overwriting each other
    model_tag = "baseline" if input_mode == "claim_only" else "evidence"
    save_directory = os.path.join(
        os.path.dirname(__file__), "saved_models", f"{model_tag}_{dataset_name}"
    )

    #creating the save directory if it does not already exist
    os.makedirs(save_directory, exist_ok=True)

    #printing a separator before training begins
    print("Starting training...\n")

    #iterating over each epoch up to MAX_EPOCHS
    for epoch_number in range(1, MAX_EPOCHS + 1):

        #running one full epoch of training and getting the average loss
        average_train_loss = train_one_epoch(
            model, train_dataloader, optimiser, scheduler, device
        )

        #evaluating the model on the validation set after this epoch
        val_f1, val_precision, val_recall, val_report = evaluate(
            model, val_dataloader, device
        )

        #printing a summary of this epoch's results
        print(f"Epoch {epoch_number} / {MAX_EPOCHS}")
        print(f"  Train loss      : {average_train_loss:.4f}")
        print(f"  Val macro F1    : {val_f1:.4f}")
        print(f"  Val precision   : {val_precision:.4f}")
        print(f"  Val recall      : {val_recall:.4f}")

        #checking whether this epoch produced a new best validation F1
        if val_f1 > best_val_f1:
            # Updating the best F1 score
            best_val_f1 = val_f1

            #storing precision and recall from this best epoch too (for the JSON results)
            best_val_precision = val_precision
            best_val_recall = val_recall

            #storing the classification report from this best epoch
            best_val_report = val_report

            #resetting the no-improvement counter
            epochs_without_improvement = 0

            #saving the model and tokenizer as the new best checkpoint
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
            print(f"  New best model saved (val F1 = {best_val_f1:.4f})")

        else:
            #incrementing the no-improvement counter
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

            #stopping training early if patience is exceeded
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch_number} epochs.")
                break

        #printing a blank line between epochs for readability
        print()

    #reloading the best checkpoint from disk so we evaluate the ACTUAL saved model,
    #not the in-memory one -- this verifies the saved checkpoint reproduces the reported metric
    best_model = RobertaForSequenceClassification.from_pretrained(save_directory).to(device)
    best_tokenizer = RobertaTokenizer.from_pretrained(save_directory)

    #re-evaluating the reloaded checkpoint on the validation set as a correctness check
    reloaded_f1, reloaded_precision, reloaded_recall, reloaded_report = evaluate(
        best_model, val_dataloader, device
    )

    #reporting the reloaded checkpoint's validation result (this is the final reported result)
    print("\nValidation classification report (reloaded best checkpoint):")
    print(reloaded_report)
    print(f"Reloaded checkpoint macro F1 : {reloaded_f1:.4f}")

    #sanity check: the reloaded model should reproduce the best F1 seen during training.
    #a mismatch would indicate a save/reload bug, so we warn if they differ noticeably.
    if abs(reloaded_f1 - best_val_f1) > 1e-4:
        print(f"  WARNING: reloaded F1 ({reloaded_f1:.4f}) differs from training-time "
              f"best F1 ({best_val_f1:.4f}) -- possible save/reload issue.")
    else:
        print(f"  Confirmed: reloaded checkpoint reproduces the reported F1 ({best_val_f1:.4f}).")

    #SciFact has no public test labels, so the validation result above is the final reported result
    print("\n(No held-out test set for SciFact, as validation is the final reported result.)")

    #printing the remaining summary
    print(f"\nBest learning rate used  : {best_lr}")
    print(f"Model saved to           : {save_directory}")

    #saving the SciFact baseline results as structured JSON for later comparison
    results = {
        "dataset": dataset_name,
        "split_reported": "validation",
        "best_learning_rate": best_lr,
        "macro_f1": reloaded_f1,
        "precision": reloaded_precision,
        "recall": reloaded_recall,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH_CLAIM_EVIDENCE if input_mode == "claim_evidence" else MAX_LENGTH_CLAIM_ONLY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "seed": seed,
        "input_mode": input_mode,
    }
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"baseline_{dataset_name}_{input_mode}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to results/baseline_{dataset_name}_{input_mode}.json")

    #returning the best model and tokenizer for optional further use
    return best_model, best_tokenizer

# ---------------------------------------------------------------------------
# SciFact-Open evaluation -- zero-shot generalisation of the SciFact baseline
# ---------------------------------------------------------------------------

def evaluate_on_scifact_open():
    """
    Evaluating the SciFact-trained baseline on SciFact-Open WITHOUT retraining.
    SciFact-Open is a test-only collection (no train/val split), so this is a
    zero-shot generalisation reference: the same claim-only model trained on
    SciFact is scored on SciFact-Open's claims.

    SciFact-Open has no NEI class (SUPPORT/CONTRADICT only). Any NEI the model
    predicts is necessarily wrong here; metrics are computed only over the labels
    actually present, so scores reflect SciFact-Open's two real classes.
    """
    #seeding for reproducibility (evaluation is deterministic, but kept for consistency)
    #seeding everything for reproducibility before any randomness happens
    set_seed(42)

    #selecting the device the same way run_baseline does
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    #locating the SciFact baseline checkpoint saved by run_baseline("scifact")
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "saved_models", "baseline_scifact"
    )
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"No SciFact baseline found at {checkpoint_dir}. "
            "Run run_baseline('scifact') first to train and save it."
        )

    #loading the trained SciFact model and its tokenizer from disk
    print(f"Loading SciFact-trained baseline from: {checkpoint_dir}")
    model = RobertaForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint_dir)

    #loading SciFact-Open claims (test-only collection)
    open_claims, _ = load_scifact_open()
    print(f"SciFact-Open claims: {len(open_claims)}")
    print(f"Label distribution: {dict(Counter(c['label'] for c in open_claims))}\n")

    #wrapping in dataset/dataloader and evaluating once
    open_dataset = ClaimDataset(open_claims, tokenizer)
    open_dataloader = DataLoader(open_dataset, batch_size=BATCH_SIZE, shuffle=False)
    open_f1, open_precision, open_recall, open_report = evaluate(model, open_dataloader, device)

    print("SciFact-Open zero-shot classification report (SciFact-trained baseline):")
    print(open_report)
    print(f"SciFact-Open macro F1     : {open_f1:.4f}")
    print(f"SciFact-Open precision    : {open_precision:.4f}")
    print(f"SciFact-Open recall       : {open_recall:.4f}")

    #saving SciFact-Open results as JSON
    results = {
        "dataset": "scifact_open",
        "evaluation": "zero_shot_from_scifact_baseline",
        "macro_f1": open_f1,
        "precision": open_precision,
        "recall": open_recall,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH_CLAIM_ONLY,
        "batch_size": BATCH_SIZE,
        "trained_on": "scifact",
        "seed": seed,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "baseline_scifact_open.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/baseline_scifact_open.json")

    return open_f1, open_precision, open_recall, open_report

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    #parsing command line arguments so we can specify the dataset without editing the file
    parser = argparse.ArgumentParser(description="RoBERTa no-retrieval baseline")
    parser.add_argument("--dataset", default="scifact",
                        choices=["scifact", "scifact_open"],
                        help="'scifact' trains the baseline; 'scifact_open' evaluates "
                             "the trained SciFact baseline on SciFact-Open (no training)")
    parser.add_argument("--input_mode", default="claim_only",
                        choices=["claim_only", "claim_evidence"],
                        help="claim_only = Model 1 (baseline); claim_evidence = Model 2 (evidence-aware)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility / multi-seed variance check")
    args = parser.parse_args()

    if args.dataset == "scifact_open":
        #evaluation-only: uses the already-trained SciFact baseline
        evaluate_on_scifact_open()
    else:
        run_baseline(dataset_name=args.dataset, input_mode=args.input_mode, seed=args.seed)


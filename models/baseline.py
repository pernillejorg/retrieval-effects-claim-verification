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

#importing our unified data loading functions for both datasets
from data.utils import load_scifact, load_sciclaimhunt, LABEL_SUPPORT, LABEL_CONTRADICT, LABEL_NEI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#defining the name of the pre-trained RoBERTa model we are fine-tuning
MODEL_NAME = "roberta-base"

#setting the maximum number of tokens RoBERTa processes per input claim
# (verified against actual claim lengths before training -- see check_max_token_length)
MAX_LENGTH = 128

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
# Token length check
# ---------------------------------------------------------------------------

def check_max_token_length(claims, tokenizer):
    """
    Checking the actual maximum token length across all claims.
    This confirms that MAX_LENGTH = 128 does not truncate any claims,
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
    if maximum_token_count > MAX_LENGTH:
        print(f"  WARNING: some claims exceed MAX_LENGTH={MAX_LENGTH} and will be truncated!")
    else:
        print(f"  MAX_LENGTH={MAX_LENGTH} is safe -- no claims will be truncated.")

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

    def __init__(self, list_of_claim_dicts, tokenizer):
        #storing the list of claim dicts so we can index into them
        self.claims = list_of_claim_dicts

        #storing the tokenizer so we can encode each claim at access time
        self.tokenizer = tokenizer

    def __len__(self):
        #returning the total number of claims in this dataset split
        return len(self.claims)

    def __getitem__(self, index):
        #retrieving the claim dict at the given index position
        claim_dict = self.claims[index]

        #tokenising the claim text into input ids and attention mask tensors
        encoding = self.tokenizer(
            claim_dict["claim"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        #converting the label string to its corresponding integer id
        label_id = LABEL_TO_ID[claim_dict["label"]]

        #returning a dict of tensors -- squeezing removes the extra batch dimension added by return_tensors="pt"
        return {
            #squeezing the input ids tensor from shape (1, MAX_LENGTH) to (MAX_LENGTH,)
            "input_ids": encoding["input_ids"].squeeze(),

            #squeezing the attention mask tensor from shape (1, MAX_LENGTH) to (MAX_LENGTH,)
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

    #computing macro-averaged F1 score across all three classes
    macro_f1 = f1_score(all_true_labels, all_predicted_labels, average="macro")

    #computing macro-averaged precision score across all three classes
    macro_precision = precision_score(all_true_labels, all_predicted_labels, average="macro")

    #computing macro-averaged recall score across all three classes
    macro_recall = recall_score(all_true_labels, all_predicted_labels, average="macro")

    #generating a full per-class breakdown report for detailed analysis
    report = classification_report(
        all_true_labels,
        all_predicted_labels,
        target_names=LABEL_LIST,
    )

    #returning all metrics and the detailed report
    return macro_f1, macro_precision, macro_recall, report


# ---------------------------------------------------------------------------
# Learning rate search -- trains with each LR and picks the best on val F1
# ---------------------------------------------------------------------------

def find_best_learning_rate(train_claims, val_claims, tokenizer, device):
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
        import random
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
    search_dataset = ClaimDataset(search_claims, tokenizer)
    val_dataset = ClaimDataset(val_claims, tokenizer)
    search_dataloader = DataLoader(search_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #iterating over each candidate learning rate
    for candidate_lr in LEARNING_RATES_TO_TRY:

        #printing which learning rate we are currently trying
        print(f"\n  Trying learning rate: {candidate_lr}")

        #loading a fresh copy of the model for each learning rate trial
        trial_model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL_LIST),
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

def run_baseline(dataset_name="scifact"):
    """
    Loading the dataset, searching for the best learning rate,
    fine-tuning RoBERTa with early stopping on claim text only,
    evaluating on the validation split, and saving the best checkpoint.

    dataset_name: 'scifact' or 'sciclaimhunt'
    """

    #printing a clear header so output is easy to read in the terminal
    print(f"\n{'=' * 60}")
    print(f"  RoBERTa No-Retrieval Baseline  --  {dataset_name.upper()}")
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
        #loading SciFact training claims
        train_claims, _ = load_scifact(split="train")

        #loading SciFact validation claims for evaluation after each epoch
        val_claims, _ = load_scifact(split="validation")

    elif dataset_name == "sciclaimhunt":
        #loading SciClaimHunt training claims
        train_claims, _ = load_sciclaimhunt(split="train")

        #loading SciClaimHunt validation claims
        val_claims, _ = load_sciclaimhunt(split="val")

    else:
        #raising an error if an unrecognised dataset name is given
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'scifact' or 'sciclaimhunt'.")

    #printing the number of training and validation claims
    print(f"Train claims : {len(train_claims)}")
    print(f"Val claims   : {len(val_claims)}")

    #printing the label distribution to detect class imbalance early
    train_label_counts = Counter(c["label"] for c in train_claims)
    print(f"Train label distribution: {dict(train_label_counts)}\n")

    #loading the RoBERTa tokenizer from HuggingFace
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    #checking that MAX_LENGTH is safe for the actual claim lengths in this dataset
    print("\nChecking claim token lengths...")
    check_max_token_length(train_claims, tokenizer)

    #searching for the best learning rate across our three candidates
    best_lr = find_best_learning_rate(train_claims, val_claims, tokenizer, device)

    #loading a fresh model for the full training run with the best learning rate
    print(f"\n\nStarting full training run with best learning rate: {best_lr}\n")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
    )

    #moving the model to the selected device
    model = model.to(device)

    #wrapping the train and validation claims in dataset and dataloader objects
    train_dataset = ClaimDataset(train_claims, tokenizer)
    val_dataset = ClaimDataset(val_claims, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #setting up AdamW optimiser with the best learning rate found above
    optimiser = AdamW(model.parameters(), lr=best_lr, weight_decay=0.01)

    '''
    Using a conservative epoch estimate for the scheduler rather than MAX_EPOCHS,
    since early stopping typically terminates training well before the maximum.
    This avoids the scheduler decaying too slowly due to an overly large step count.
    Empirically, 3-5 epochs is typical for RoBERTa on small scientific datasets (consistent with Wadden et al., 2020 on SciFact).
    '''
 
    SCHEDULER_EPOCH_ESTIMATE = 5
    total_training_steps = len(train_dataloader) * SCHEDULER_EPOCH_ESTIMATE

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

    #initialising the best validation report as an empty string -- updated when best F1 improves
    best_val_report = ""

    #initialising the counter tracking how many epochs have passed without improvement
    epochs_without_improvement = 0

    #constructing the save path for the best model checkpoint
    save_directory = os.path.join(
        os.path.dirname(__file__), "saved_models", f"baseline_{dataset_name}"
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

    #printing the final classification report from the best checkpoint epoch
    print("\nFinal validation classification report (best checkpoint):")
    print(best_val_report)

    #printing a summary of the final best result
    print(f"\nBest validation macro F1 : {best_val_f1:.4f}")
    print(f"Best learning rate used  : {best_lr}")
    print(f"Model saved to           : {save_directory}")

    #returning the model and tokenizer for optional further use
    return model, tokenizer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    #parsing command line arguments so we can specify the dataset without editing the file
    parser = argparse.ArgumentParser(description="RoBERTa no-retrieval baseline")
    parser.add_argument("--dataset", default="scifact", choices=["scifact", "sciclaimhunt"],
                        help="Dataset to run baseline on")
    args = parser.parse_args()

    #running the baseline on the chosen dataset
    run_baseline(dataset_name=args.dataset)
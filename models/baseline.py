"""
baseline.py -- RoBERTa no-retrieval baseline model

Trains and evaluates a RoBERTa classifier on claim text alone,
with no retrieved evidence whatsoever.

This answers the question: what can the model predict using only
its pre-trained knowledge, before any retrieval is introduced?

Every downstream RAG pipeline result is compared against this.
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

'''
Configuration
'''

#defining the name of the pre-trained RoBERTa model we are fine-tuning
MODEL_NAME = "roberta-base"

#defining the maximum number of tokens RoBERTa processes per input claim
MAX_LENGTH = 128

#setting how many examples are processed together in one forward pass
BATCH_SIZE = 16

#setting how many complete passes through the training data we perform
NUM_EPOCHS = 3

#the learning rate used by the AdamW optimiser
LEARNING_RATE = 2e-5

#ordered list of label names used consistently across the project
LABEL_LIST = [LABEL_SUPPORT, LABEL_CONTRADICT, LABEL_NEI]

#creating a dictionary mapping each label string to its integer index for the model
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_LIST)}

#creating a dictionary mapping each integer index back to its label string for reporting
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}



class ClaimDataset(Dataset):
    """
    Setting a list of claim dicts into a PyTorch Dataset.

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

        #returning a dict of tensors -- squeezing removes the extra batch dimension
        return {
            #squeezing the input ids tensor from shape (1, MAX_LENGTH) to (MAX_LENGTH,)
            "input_ids": encoding["input_ids"].squeeze(),

            #squeezing the attention mask tensor from shape (1, MAX_LENGTH) to (MAX_LENGTH,)
            "attention_mask": encoding["attention_mask"].squeeze(),

            #converting the label integer to a long tensor as required by PyTorch loss functions
            "label": torch.tensor(label_id, dtype=torch.long),
        }

'''
Training function
'''

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

'''
Evaluation function
'''

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

'''
Main training and evaluation pipeline
'''

def run_baseline(dataset_name="scifact"):
    """
    Loading the dataset, fine-tuning RoBERTa on claim text only,
    evaluating on the validation split, and saving the trained model.

    dataset_name: 'scifact' or 'sciclaimhunt'
    """

    #printing a clear header so output is easy to read
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
        #loading SciFact training claims (labels only, no retrieval corpus needed here)
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

    #loading the RoBERTa model with a 3-class classification head
    print(f"Loading model: {MODEL_NAME} with {len(LABEL_LIST)} output classes\n")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
    )

    #moving the model to the selected device
    model = model.to(device)

    #wrapping the train claims list in our ClaimDataset class
    train_dataset = ClaimDataset(train_claims, tokenizer)

    #wrapping the validation claims list in our ClaimDataset class
    val_dataset = ClaimDataset(val_claims, tokenizer)

    #creating a DataLoader for training with shuffling enabled
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #creating a DataLoader for validation without shuffling
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #setting up AdamW optimiser with a small weight decay for regularisation
    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    #computing the total number of training steps across all epochs
    total_training_steps = len(train_dataloader) * NUM_EPOCHS

    #computing the number of warmup steps as 10% of total training steps
    warmup_steps = int(0.1 * total_training_steps)

    #setting up a linear warmup then linear decay learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    '''
    Training loop where running for NUM_EPOCHS epochs
    '''

    #printing a separator before training begins
    print("Starting training...\n")

    #iterating over each epoch from 1 to NUM_EPOCHS inclusive
    for epoch_number in range(1, NUM_EPOCHS + 1):

        #running one full epoch of training and getting the average loss
        average_train_loss = train_one_epoch(
            model, train_dataloader, optimiser, scheduler, device
        )

        #evaluating the model on the validation set after this epoch
        val_f1, val_precision, val_recall, val_report = evaluate(
            model, val_dataloader, device
        )

        #printing a summary of this epoch's training and validation results
        print(f"Epoch {epoch_number} / {NUM_EPOCHS}")
        print(f"  Train loss      : {average_train_loss:.4f}")
        print(f"  Val macro F1    : {val_f1:.4f}")
        print(f"  Val precision   : {val_precision:.4f}")
        print(f"  Val recall      : {val_recall:.4f}")
        print()

    #printing the full per-class classification report after all epochs
    print("Final validation classification report:")
    print(val_report)

    '''
    Saving the trained model
    '''

    #constructing the save path inside models/saved_models/
    save_directory = os.path.join(
        os.path.dirname(__file__), "saved_models", f"baseline_{dataset_name}"
    )

    #creating the save directory if it does not already exist
    os.makedirs(save_directory, exist_ok=True)

    #saving the fine-tuned model weights and configuration
    model.save_pretrained(save_directory)

    #saving the tokenizer so it can be reloaded with the model later
    tokenizer.save_pretrained(save_directory)

    #printing confirmation of where the model was saved
    print(f"\nModel saved to: {save_directory}")

    #returning the trained model and tokenizer for optional further use
    return model, tokenizer


'''
Entry point for running the baseline experiment directly from the command line.
'''

if __name__ == "__main__":
    #running the baseline experiment on SciFact first
    run_baseline(dataset_name="scifact")

    #run_baseline(dataset_name="sciclaimhunt")  #uncomment to run on SciClaimHunt after SciFact
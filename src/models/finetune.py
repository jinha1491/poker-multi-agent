import json
import torch
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import chromadb
from collections import Counter

# map GTO actions to class indices
LABEL_MAP = {
    "fold": 0,
    "call": 1,
    "check": 2,
    "raise": 3,
    "bet": 4,
}

NUM_CLASSES = 5


class PokerTextDataset(Dataset):
    """
    Dataset that feeds raw poker scenario text directly to DistilBERT.
    No manual tokenization needed — DistilBERT's tokenizer handles it.
    """
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 256):
        self.labels = torch.tensor(labels, dtype=torch.long)

        # tokenize all texts at once
        # padding=True — pad shorter sequences
        # truncation=True — cut sequences longer than max_length
        # return_tensors="pt" — return PyTorch tensors
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }


def load_data_from_chromadb(limit: int = 50000, batch_size: int = 5000):
    """
    Pull raw scenario texts and GTO labels directly from ChromaDB.
    We use raw text this time — no manual tokenization.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="poker_strategy")

    print(f"Loading {limit} scenarios from ChromaDB...")

    texts = []
    labels = []
    offset = 0

    while offset < limit:
        current_batch = min(batch_size, limit - offset)

        results = collection.get(
            limit=current_batch,
            offset=offset,
            include=["documents", "metadatas"]
        )

        if not results["documents"]:
            break

        for doc, meta in zip(results["documents"], results["metadatas"]):
            raw_label = meta.get("optimal_action", "")
            label_text = raw_label.lower().strip().split()[0] if raw_label else ""

            # skip unknown labels
            if label_text not in LABEL_MAP:
                continue

            texts.append(doc)
            labels.append(LABEL_MAP[label_text])

        offset += current_batch
        print(f"  Loaded {offset} scenarios...")

    print(f"Total valid samples: {len(texts)}")
    return texts, labels


def compute_class_weights(labels: list) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(NUM_CLASSES):
        count = counts.get(i, 1)
        weights.append(total / (NUM_CLASSES * count))
    return torch.tensor(weights, dtype=torch.float)


def finetune():
    wandb.init(
        project="poker-distilbert",
        config={
            "model": "distilbert-base-uncased",
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "max_length": 256,
            "train_split": 0.8,
            "data_limit": 50000,
        }
    )
    cfg = wandb.config

    # load pretrained DistilBERT tokenizer
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # load data
    texts, labels = load_data_from_chromadb(limit=cfg.data_limit)

    # build dataset
    print("Tokenizing dataset...")
    dataset = PokerTextDataset(texts, labels, tokenizer, max_length=cfg.max_length)

    # train/val split
    train_size = int(cfg.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train: {train_size} | Val: {val_size}")

    # load pretrained DistilBERT with classification head
    # num_labels=5 adds a linear layer mapping 768 → 5 classes
    import os

    print("Loading DistilBERT model...")
    if os.path.exists("src/models/poker_distilbert"):
    	print("Resuming from saved checkpoint...")
    	model = DistilBertForSequenceClassification.from_pretrained(
        	"src/models/poker_distilbert"
    )
    else:
    	model = DistilBertForSequenceClassification.from_pretrained(
        	"distilbert-base-uncased",
        	num_labels=NUM_CLASSES
    )
    # use GPU if available, otherwise CPU
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
	)
    print(f"Training on: {device}")
    model = model.to(device)

    # class weights for imbalanced data
    class_weights = compute_class_weights(labels).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # AdamW with lower learning rate — important for fine-tuning
    # too high LR destroys pretrained weights (catastrophic forgetting)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    # linear warmup scheduler — standard for transformer fine-tuning
    # gradually increases LR at start, then linearly decays
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    wandb.watch(model, log="all")

    best_val_acc = 0.0

    print("\nStarting fine-tuning...")
    for epoch in range(cfg.epochs):
        # training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_batch = batch["labels"].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # calculate loss with class weights
            loss = criterion(logits, label_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == label_batch).sum().item()
            train_total += label_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label_batch = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, label_batch)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == label_batch).sum().item()
                val_total += label_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save the full model for later use
            model.save_pretrained("src/models/poker_distilbert")
            tokenizer.save_pretrained("src/models/poker_distilbert")
            print(f"  New best model saved (val_acc: {val_acc:.4f})")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\nFine-tuning complete. Best val accuracy: {best_val_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    finetune()

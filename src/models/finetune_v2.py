import os
import torch
import wandb
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)

LABEL_MAP = {"fold": 0, "call": 1, "check": 2, "raise": 3, "bet": 4}
NUM_CLASSES = 5
SAVE_PATH = "src/models/poker_distilbert_v2"


class PokerTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.encodings = tokenizer(texts, padding=True, truncation=True,
                                   max_length=max_length, return_tensors="pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_shuffled_sample(n, seed=42):
    # v1 took the first 50K rows from ChromaDB, which turned out to be 0% preflop
    # because the dataset is stored in blocks. Shuffling the full train split
    # first gives a sample that matches the real street mix (~11% preflop).
    ds = load_dataset("RZ412/PokerBench", split="train").shuffle(seed=seed)

    texts, labels = [], []
    for row in ds:
        raw = row["output"].lower().strip()
        action = raw.split()[0] if raw else ""
        if action in LABEL_MAP:
            texts.append(row["instruction"])
            labels.append(LABEL_MAP[action])
        if len(texts) >= n:
            break

    preflop = sum(1 for t in texts if "The flop comes" not in t)
    print(f"Sampled {len(texts)} hands, {100 * preflop / len(texts):.1f}% preflop")
    return texts, labels


def compute_class_weights(labels):
    counts = Counter(labels)
    total = len(labels)
    return torch.tensor([total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
                        dtype=torch.float)


def finetune():
    wandb.init(project="poker-distilbert", config={
        "version": "v2-shuffled-sample",
        "model": "distilbert-base-uncased",
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "max_length": 256,
        "train_split": 0.8,
        "data_limit": 50000,
        "seed": 42,
    })
    cfg = wandb.config
    torch.manual_seed(cfg.seed)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    texts, labels = load_shuffled_sample(cfg.data_limit, seed=cfg.seed)

    print("Tokenizing...")
    dataset = PokerTextDataset(texts, labels, tokenizer, cfg.max_length)
    train_size = int(cfg.train_split * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size],
                                    generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # start from the original pretrained weights, not v1, so the only
    # difference between v1 and v2 is how the training data was sampled
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_CLASSES)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Training on: {device}")

    criterion = torch.nn.CrossEntropyLoss(weight=compute_class_weights(labels).to(device))
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    best_val_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                val_loss += criterion(logits, y).item()
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(SAVE_PATH, exist_ok=True)
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"  New best model saved (val_acc: {val_acc:.4f})")

        wandb.log({"epoch": epoch + 1,
                   "train_loss": train_loss / len(train_loader), "train_acc": train_acc,
                   "val_loss": val_loss / len(val_loader), "val_acc": val_acc,
                   "learning_rate": scheduler.get_last_lr()[0]})

        print(f"Epoch {epoch + 1}/{cfg.epochs} | Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss / len(val_loader):.4f} | "
              f"Val Acc: {val_acc:.4f}")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    finetune()

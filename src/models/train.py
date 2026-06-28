import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import wandb
from src.models.transformer import PokerTransformer


class PokerDataset(Dataset):
    """
    PyTorch Dataset — wraps our sequences and labels.
    
    Why we need this:
    PyTorch's DataLoader needs a Dataset object to efficiently
    batch, shuffle, and load data during training.
    We implement __len__ and __getitem__ — the two methods
    PyTorch requires for any custom dataset.
    """
    def __init__(self, sequences: list, labels: list):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def compute_class_weights(labels: list) -> torch.Tensor:
    """
    Give higher weight to underrepresented classes.
    
    Why:
    raise (15%) and bet (10%) appear less than fold/call/check (25% each).
    Without weighting, the model learns to ignore rare classes.
    Weight = total_samples / (num_classes * class_count)
    Rarer class → higher weight → model pays more attention to it.
    """
    counts = Counter(labels)
    total = len(labels)
    num_classes = 5  # fold, call, check, raise, bet
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)  # avoid division by zero
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float)


def train(config=None):
    # initialize weights and biases run
    wandb.init(
        project="poker-transformer",
        config={
            "epochs": 20,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "train_split": 0.8,
        }
    )
    cfg = wandb.config

    # load sequences from disk
    print("Loading sequences...")
    with open("src/data/sequences.json", "r") as f:
        data = json.load(f)

    sequences = data["sequences"]
    labels = data["labels"]

    # filter out UNK labels (7) — we only train on known actions
    filtered = [(s, l) for s, l in zip(sequences, labels) if l < 5]
    sequences = [x[0] for x in filtered]
    labels = [x[1] for x in filtered]
    print(f"Training on {len(sequences)} sequences after filtering UNK")

    # build dataset and split into train/val
    dataset = PokerDataset(sequences, labels)
    train_size = int(cfg.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train: {train_size} | Val: {val_size}")

    # initialize model
    model = PokerTransformer(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    )

    # class weights to handle imbalance
    class_weights = compute_class_weights(labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW optimizer — standard for transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # learning rate scheduler — reduces LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # watch model with wandb — logs gradients and parameters
    wandb.watch(model, log="all")

    best_val_acc = 0.0

    print("\nStarting training...")
    for epoch in range(cfg.epochs):
        # --- training phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for sequences_batch, labels_batch in train_loader:
            # create padding mask — True where token is PAD (6)
            padding_mask = (sequences_batch == 6)

            # forward pass
            outputs = model(sequences_batch, padding_mask)

            # calculate loss
            loss = criterion(outputs, labels_batch)

            # backward pass — compute gradients
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping — prevents exploding gradients
            # common practice when training transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # update weights
            optimizer.step()

            # track metrics
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # --- validation phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # no gradients needed for validation
            for sequences_batch, labels_batch in val_loader:
                padding_mask = (sequences_batch == 6)
                outputs = model(sequences_batch, padding_mask)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # update learning rate scheduler
        scheduler.step(avg_val_loss)

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "src/models/best_model.pt")
            print(f"  New best model saved (val_acc: {val_acc:.4f})")

        # log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    train()

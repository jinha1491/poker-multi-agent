import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

LABEL_MAP = {"fold": 0, "call": 1, "check": 2, "raise": 3, "bet": 4}
LABEL_NAMES = ["fold", "call", "check", "raise", "bet"]
MODEL_PATH = "src/models/poker_distilbert"


def load_test_set():
    ds = load_dataset("RZ412/PokerBench", split="test")

    texts, labels = [], []
    skipped = Counter()

    for row in ds:
        raw = row["output"].lower().strip()
        # "bet 18" -> "bet", same cleanup as training
        action = raw.split()[0] if raw else ""
        if action not in LABEL_MAP:
            skipped[action] += 1
            continue
        texts.append(row["instruction"])
        labels.append(LABEL_MAP[action])

    print(f"Test hands kept: {len(texts)}")
    if skipped:
        print(f"Skipped (unknown labels): {dict(skipped)}")
    return texts, labels


def predict(model, tokenizer, texts, device, batch_size=32):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        if (i // batch_size) % 50 == 0:
            print(f"  {i + len(batch)}/{len(texts)}")
    return preds


def plot_confusion(cm, path="src/models/confusion_matrix.png"):
    # normalize by row so each cell = % of that true label
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), LABEL_NAMES)
    ax.set_yticks(range(5), LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("DistilBERT on PokerBench test set")

    for r in range(5):
        for c in range(5):
            color = "white" if cm_pct[r, c] > 0.5 else "black"
            ax.text(c, r, f"{cm_pct[r, c]:.2f}", ha="center", va="center", color=color)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Confusion matrix saved to {path}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

    texts, labels = load_test_set()

    # baseline: always guess the most common label
    majority = Counter(labels).most_common(1)[0][0]
    baseline_acc = sum(1 for l in labels if l == majority) / len(labels)

    print("\nRunning predictions...")
    preds = predict(model, tokenizer, texts, device)

    acc = accuracy_score(labels, preds)
    print(f"\nMajority-class baseline: {baseline_acc:.4f} (always '{LABEL_NAMES[majority]}')")
    print(f"DistilBERT test accuracy: {acc:.4f}\n")

    print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))

    cm = confusion_matrix(labels, preds)
    plot_confusion(cm)

    results = {
        "test_accuracy": acc,
        "majority_baseline": baseline_acc,
        "num_test_hands": len(labels),
        "report": classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }
    with open("src/models/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to src/models/test_results.json")


if __name__ == "__main__":
    main()

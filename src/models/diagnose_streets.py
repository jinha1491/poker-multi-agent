import chromadb
from collections import Counter, defaultdict
from src.models.evaluate import load_test_set, predict, MODEL_PATH
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

STREETS = ["preflop", "flop", "turn", "river"]


def street_of(text: str) -> str:
    # check the latest street first, since a river hand also mentions flop and turn
    if "The river comes" in text:
        return "river"
    if "The turn comes" in text:
        return "turn"
    if "The flop comes" in text:
        return "flop"
    return "preflop"


def training_slice_streets(limit=50000, batch_size=5000):
    # same 50K slice the DistilBERT fine-tune pulled from ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="poker_strategy")

    counts = Counter()
    offset = 0
    while offset < limit:
        res = collection.get(limit=min(batch_size, limit - offset),
                             offset=offset, include=["documents"])
        if not res["documents"]:
            break
        for doc in res["documents"]:
            counts[street_of(doc)] += 1
        offset += batch_size
    return counts


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Checking street mix of the 50K training slice...")
    train_counts = training_slice_streets()
    train_total = sum(train_counts.values())

    print("Checking street mix of the full 563K train split...")
    full_train = load_dataset("RZ412/PokerBench", split="train")
    full_counts = Counter(street_of(t) for t in full_train["instruction"])
    full_total = sum(full_counts.values())

    print("Running model on test set...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    texts, labels = load_test_set()
    preds = predict(model, tokenizer, texts, device)

    correct = defaultdict(int)
    total = defaultdict(int)
    for text, y, p in zip(texts, labels, preds):
        s = street_of(text)
        total[s] += 1
        correct[s] += int(y == p)

    print("\n" + "=" * 72)
    print(f"{'street':<10}{'50K slice':>14}{'full train':>14}{'test hands':>12}{'test acc':>12}")
    print("-" * 72)
    for s in STREETS:
        slice_pct = 100 * train_counts[s] / train_total
        full_pct = 100 * full_counts[s] / full_total
        acc = correct[s] / total[s] if total[s] else float("nan")
        print(f"{s:<10}{slice_pct:>13.1f}%{full_pct:>13.1f}%{total[s]:>12}{acc:>12.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

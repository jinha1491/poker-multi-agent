import re
import json
import chromadb
from collections import Counter

ACTION_TOKENS = {
    "fold": 0,
    "call": 1,
    "check": 2,
    "raise": 3,
    "bet": 4,
    "allin": 5,
    "PAD": 6,
    "UNK": 7,
}

MAX_SEQ_LEN = 20


def extract_actions(text: str) -> list[str]:
    """Pull betting actions out of a raw PokerBench scenario string."""
    text = text.lower()
    keywords = ["fold", "call", "check", "raise", "bet", "allin", "all-in", "all in"]
    actions = []
    words = re.findall(r'\b\w+\b', text)
    for word in words:
        if word in keywords:
            if word in ["all-in", "all in"]:
                actions.append("allin")
            else:
                actions.append(word)
    return actions


def tokenize(actions: list[str]) -> list[int]:
    """Convert action strings into integer tokens."""
    return [ACTION_TOKENS.get(action, ACTION_TOKENS["UNK"]) for action in actions]


def pad_or_truncate(tokens: list[int], max_len: int) -> list[int]:
    """Make every sequence exactly max_len long."""
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [ACTION_TOKENS["PAD"]] * (max_len - len(tokens))


def load_sequences_from_chromadb(limit: int = 10000):
    """Pull scenarios from ChromaDB and convert to tokenized sequences."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="poker_strategy")

    print(f"Pulling {limit} scenarios from ChromaDB...")

    results = collection.get(
        limit=limit,
        include=["documents", "metadatas"]
    )

    sequences = []
    labels = []

    for doc, meta in zip(results["documents"], results["metadatas"]):
        actions = extract_actions(doc)

        if len(actions) == 0:
            continue

        tokens = tokenize(actions)
        tokens = pad_or_truncate(tokens, MAX_SEQ_LEN)

        # take only first word to handle sizing like "bet 18" → "bet"
        raw_label = meta.get("optimal_action", "")
        label_text = raw_label.lower().strip().split()[0] if raw_label else ""
        label = ACTION_TOKENS.get(label_text, ACTION_TOKENS["UNK"])

        sequences.append(tokens)
        labels.append(label)

    print(f"Processed {len(sequences)} valid sequences")
    return sequences, labels


def analyze_distribution(labels: list[int]):
    """Check class balance across actions."""
    reverse_tokens = {v: k for k, v in ACTION_TOKENS.items()}
    counts = Counter(labels)
    print("\nLabel distribution:")
    for token, count in sorted(counts.items()):
        action = reverse_tokens.get(token, "unknown")
        pct = count / len(labels) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    sequences, labels = load_sequences_from_chromadb(limit=10000)

    print(f"\nSample sequence (tokens): {sequences[0]}")
    print(f"Sample label: {labels[0]}")
    print(f"Sequence shape: {len(sequences)} x {len(sequences[0])}")

    analyze_distribution(labels)

    print("\nSaving sequences to disk...")
    with open("src/data/sequences.json", "w") as f:
        json.dump({"sequences": sequences, "labels": labels}, f)
    print("Saved to src/data/sequences.json")

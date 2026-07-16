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
    # position tokens
    "utg": 8,
    "mp": 9,
    "co": 10,
    "btn": 11,
    "sb": 12,
    "bb": 13,
    # street tokens
    "preflop": 14,
    "flop": 15,
    "turn": 16,
    "river": 17,
    # stack depth tokens
    "short": 18,
    "medium": 19,
    "deep": 20,
}

VOCAB_SIZE = 21
MAX_SEQ_LEN = 30  # longer now to fit richer sequences


def extract_street(text: str) -> list[int]:
    """Extract street tokens from scenario text."""
    text = text.lower()
    tokens = []
    if "preflop" in text or "pre-flop" in text:
        tokens.append(ACTION_TOKENS["preflop"])
    if "flop" in text and "preflop" not in text[:text.find("flop")]:
        tokens.append(ACTION_TOKENS["flop"])
    if "turn" in text:
        tokens.append(ACTION_TOKENS["turn"])
    if "river" in text:
        tokens.append(ACTION_TOKENS["river"])
    return tokens


def extract_position(text: str) -> list[int]:
    """Extract position tokens from scenario text."""
    text = text.lower()
    tokens = []
    position_map = {
        "under the gun": "utg",
        "utg": "utg",
        "middle position": "mp",
        "mp": "mp",
        "cutoff": "co",
        "co": "co",
        "button": "btn",
        "btn": "btn",
        "small blind": "sb",
        "sb": "sb",
        "big blind": "bb",
        "bb": "bb",
    }
    for phrase, token_key in position_map.items():
        if phrase in text:
            tokens.append(ACTION_TOKENS[token_key])
    return tokens[:3]  # cap at 3 positions to keep sequences compact


def extract_stack_depth(text: str) -> list[int]:
    """
    Classify stack depth from scenario text.
    Short: under 20bb, Medium: 20-60bb, Deep: 60bb+
    """
    text = text.lower()
    # look for bb amounts in text
    bb_matches = re.findall(r'(\d+)\s*bb', text)
    if bb_matches:
        stack = int(bb_matches[0])
        if stack < 20:
            return [ACTION_TOKENS["short"]]
        elif stack < 60:
            return [ACTION_TOKENS["medium"]]
        else:
            return [ACTION_TOKENS["deep"]]
    return []


def extract_actions(text: str) -> list[int]:
    """Extract action tokens from scenario text."""
    text = text.lower()
    keywords = ["fold", "call", "check", "raise", "bet", "allin", "all-in"]
    tokens = []
    words = re.findall(r'\b\w+\b', text)
    for word in words:
        if word in keywords:
            if word == "all-in":
                tokens.append(ACTION_TOKENS["allin"])
            else:
                tokens.append(ACTION_TOKENS[word])
    return tokens


def build_rich_sequence(text: str) -> list[int]:
    """
    Combine all feature types into one rich sequence.
    Order: [street tokens] + [position tokens] + [stack tokens] + [action tokens]
    """
    streets = extract_street(text)
    positions = extract_position(text)
    stacks = extract_stack_depth(text)
    actions = extract_actions(text)

    sequence = streets + positions + stacks + actions
    return sequence if sequence else [ACTION_TOKENS["UNK"]]


def pad_or_truncate(tokens: list[int], max_len: int) -> list[int]:
    """Make every sequence exactly max_len long."""
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [ACTION_TOKENS["PAD"]] * (max_len - len(tokens))


def load_sequences_from_chromadb(limit: int = 100000, batch_size: int = 5000):
    """Pull scenarios from ChromaDB in batches and build rich sequences."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="poker_strategy")

    print(f"Pulling {limit} scenarios from ChromaDB in batches of {batch_size}...")

    sequences = []
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
            sequence = build_rich_sequence(doc)

            if len(sequence) == 0:
                continue

            tokens = pad_or_truncate(sequence, MAX_SEQ_LEN)

            raw_label = meta.get("optimal_action", "")
            label_text = raw_label.lower().strip().split()[0] if raw_label else ""
            label = ACTION_TOKENS.get(label_text, ACTION_TOKENS["UNK"])

            sequences.append(tokens)
            labels.append(label)

        offset += current_batch
        print(f"  Processed {offset} scenarios...")

    print(f"Total valid sequences: {len(sequences)}")
    return sequences, labels


def analyze_distribution(labels: list[int]):
    """Check class balance."""
    reverse_tokens = {v: k for k, v in ACTION_TOKENS.items()}
    counts = Counter(labels)
    print("\nLabel distribution:")
    for token, count in sorted(counts.items()):
        action = reverse_tokens.get(token, "unknown")
        pct = count / len(labels) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    sequences, labels = load_sequences_from_chromadb(limit=100000)

    print(f"\nSample sequence (tokens): {sequences[0]}")
    print(f"Sample label: {labels[0]}")
    print(f"Sequence shape: {len(sequences)} x {len(sequences[0])}")

    analyze_distribution(labels)

    print("\nSaving sequences to disk...")
    with open("src/data/sequences.json", "w") as f:
        json.dump({"sequences": sequences, "labels": labels}, f)
    print("Saved to src/data/sequences.json")

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import median

from datasets import load_dataset
from src.models.predictor import ActionPredictor
from src.agents.supervisor import build_graph

LABELS = ["fold", "call", "check", "raise", "bet"]
N_HANDS = 300
SEED = 42


def street_of(text):
    if "The river comes" in text:
        return "river"
    if "The turn comes" in text:
        return "turn"
    if "The flop comes" in text:
        return "flop"
    return "preflop"


def stratified_sample(texts, n, seed):
    # keep each street's share the same as in the full test set
    by_street = defaultdict(list)
    for i, t in enumerate(texts):
        by_street[street_of(t)].append(i)
    rng = random.Random(seed)
    picked = []
    for idxs in by_street.values():
        k = round(n * len(idxs) / len(texts))
        picked += rng.sample(idxs, k)
    return sorted(picked)


def to_label(text):
    # the strategy agent writes free text like "Raise" or "raise to 9bb",
    # so take whichever action word appears first
    t = (text or "").lower()
    hits = [(t.find(a), a) for a in LABELS if a in t]
    return min(hits)[1] if hits else None


def run_llm(graph, text):
    start = time.perf_counter()
    try:
        out = graph.invoke({
            "hand_situation": text,
            "opponent_action": "See the action history in the hand above.",
            "hand_analysis": {}, "opponent_model": {}, "strategy": {}, "final_explanation": "",
        })
        raw = out["strategy"].get("action", "")
    except Exception as e:
        raw = f"ERROR: {e}"
    return raw, time.perf_counter() - start


def main():
    ds = load_dataset("RZ412/PokerBench", split="test")
    texts, outputs = ds["instruction"], ds["output"]
    idxs = stratified_sample(texts, N_HANDS, SEED)

    hands = [{"idx": i, "street": street_of(texts[i]),
              "truth": outputs[i].lower().split()[0], "text": texts[i]} for i in idxs]
    counts = defaultdict(int)
    for h in hands:
        counts[h["street"]] += 1
    print(f"Sampled {len(hands)} hands: {dict(counts)}")

    print("\nRunning DistilBERT (CPU, one hand at a time)...")
    predictor = ActionPredictor()
    for h in hands:
        start = time.perf_counter()
        h["bert_pred"] = predictor.predict(h["text"])["action"]
        h["bert_latency"] = time.perf_counter() - start

    print("Running LLM agent pipeline (4 GPT-4o-mini calls per hand)...")
    graph = build_graph()
    wall_start = time.perf_counter()
    # a few hands in parallel to keep the run short; each hand's latency is still timed on its own
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda h: run_llm(graph, h["text"]), hands))
    llm_wall = time.perf_counter() - wall_start

    for h, (raw, lat) in zip(hands, results):
        h["llm_raw"] = raw
        h["llm_pred"] = to_label(raw)
        h["llm_latency"] = lat

    def accuracy(rows, key):
        return sum(r[key] == r["truth"] for r in rows) / len(rows)

    print("\n" + "=" * 64)
    print(f"{'':<12}{'DistilBERT v2':>18}{'LLM pipeline':>18}{'hands':>10}")
    print("-" * 64)
    print(f"{'overall':<12}{accuracy(hands, 'bert_pred'):>18.4f}{accuracy(hands, 'llm_pred'):>18.4f}{len(hands):>10}")
    for s in ["preflop", "flop", "turn", "river"]:
        rows = [h for h in hands if h["street"] == s]
        if rows:
            print(f"{s:<12}{accuracy(rows, 'bert_pred'):>18.4f}{accuracy(rows, 'llm_pred'):>18.4f}{len(rows):>10}")
    print("-" * 64)
    print(f"{'median latency':<16}{median(h['bert_latency'] for h in hands):>13.3f}s"
          f"{median(h['llm_latency'] for h in hands):>17.2f}s")
    unparsed = sum(1 for h in hands if h["llm_pred"] is None)
    errors = sum(1 for h in hands if str(h["llm_raw"]).startswith("ERROR"))
    print(f"LLM answers with no recognizable action: {unparsed} (API errors: {errors})")
    print(f"LLM pipeline wall time: {llm_wall / 60:.1f} min")
    print("=" * 64)

    with open("src/models/baseline_comparison.json", "w") as f:
        json.dump([{k: v for k, v in h.items() if k != "text"} for h in hands], f, indent=2)
    print("Per-hand results saved to src/models/baseline_comparison.json")


if __name__ == "__main__":
    main()

import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

LABEL_NAMES = ["fold", "call", "check", "raise", "bet"]

# phrases every PokerBench prompt contains; free-form text usually won't
FORMAT_MARKERS = ["your position is", "your holding is", "now it is your turn"]


class ActionPredictor:
    """Serves the fine-tuned DistilBERT that predicts the solver's action."""

    def __init__(self, model_path=os.getenv("DISTILBERT_PATH", "src/models/poker_distilbert_v2")):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, hand_situation: str) -> dict:
        # same 256-token limit the model was trained and evaluated with
        enc = self.tokenizer(hand_situation, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0]

        idx = int(torch.argmax(probs))
        text = hand_situation.lower()
        return {
            "action": LABEL_NAMES[idx],
            "confidence": round(probs[idx].item(), 4),
            "distribution": {a: round(probs[i].item(), 4) for i, a in enumerate(LABEL_NAMES)},
            "in_distribution": all(m in text for m in FORMAT_MARKERS),
        }


if __name__ == "__main__":
    from datasets import load_dataset

    predictor = ActionPredictor()
    ds = load_dataset("RZ412/PokerBench", split="test")

    # one postflop and one preflop hand from the test set, plus off-format text
    for i in [0, 10999]:
        row = ds[i]
        r = predictor.predict(row["instruction"])
        print(f"Test row {i}: predicted {r['action']} ({r['confidence']}) | solver says: {row['output']} "
              f"| in_distribution: {r['in_distribution']}")

    r = predictor.predict("I have pocket aces, what should I do?")
    print(f"Free-form text: predicted {r['action']} ({r['confidence']}) | in_distribution: {r['in_distribution']}")

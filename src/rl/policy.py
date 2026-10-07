import re
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

ACTIONS = ["fold", "call", "check", "raise", "bet"]
POSITIONS = ["utg", "mp", "co", "btn", "sb", "bb"]
STACK_DEPTHS = ["short", "medium", "deep"]
STRENGTH_PHRASES = ["a weak hand", "a marginal hand", "a medium strength hand",
                    "a strong hand", "a premium hand"]
FEATURE_DIM = len(STRENGTH_PHRASES) + len(POSITIONS) + len(STACK_DEPTHS)  # 14


def state_features(hand_strength, position, stack_depth):
    # one-hot encode the three facts that fully describe a state
    f = torch.zeros(FEATURE_DIM)
    f[hand_strength] = 1.0
    f[5 + POSITIONS.index(position)] = 1.0
    f[11 + STACK_DEPTHS.index(stack_depth)] = 1.0
    return f


def parse_state(text):
    # recover the structured state from the environment's text template
    t = text.lower()
    strength = next((i for i, p in enumerate(STRENGTH_PHRASES) if p in t), None)
    pos = re.search(r"in the (\w+) position", t)
    stack = re.search(r"with (\w+) stacks", t)
    if (strength is None or not pos or not stack
            or pos.group(1) not in POSITIONS or stack.group(1) not in STACK_DEPTHS):
        raise ValueError(
            "Expected format: 'You are in the BTN position with deep stacks. You have a premium hand.'")
    return {"hand_strength": strength, "position": pos.group(1), "stack_depth": stack.group(1)}


class PokerPolicy(nn.Module):
    """
    Policy network for the RL agent.

    Input = frozen DistilBERT [CLS] embedding + 14 one-hot state features.
    The features were added after finding the embeddings barely separated
    hand strengths (weak vs strong cosine similarity 0.988), which left
    training dependent on luck to escape a "check everything" optimum.
    """

    ACTIONS = ACTIONS

    def __init__(self, distilbert_path="src/models/poker_distilbert_v2",
                 hidden_dim=128, use_embedding=True):
        super().__init__()
        self.use_embedding = use_embedding

        if use_embedding:
            self.tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path)
            self.encoder = DistilBertModel.from_pretrained(distilbert_path)
            for param in self.encoder.parameters():
                param.requires_grad = False
            input_dim = 768 + FEATURE_DIM
        else:
            input_dim = FEATURE_DIM

        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(ACTIONS)),
        )

    def encode_state(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def build_input(self, texts, features):
        if self.use_embedding:
            return torch.cat([self.encode_state(texts), features], dim=1)
        return features

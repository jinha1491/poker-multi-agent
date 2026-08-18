import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


class PokerPolicy(nn.Module):
    """
    Policy network for the RL agent.

    Architecture:
    Text state -> DistilBERT (frozen) -> embedding ->
    Policy MLP (trainable) -> action probabilities

    We reuse the fine-tuned DistilBERT as a frozen feature extractor.
    It already understands poker language deeply from supervised
    fine-tuning — we don't want to destroy that by training it further
    with noisy RL gradients. Only the small policy head learns during
    RL training.
    """

    ACTIONS = ["fold", "call", "check", "raise", "bet"]

    def __init__(self, distilbert_path: str = "src/models/poker_distilbert", hidden_dim: int = 128):
        super().__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path)
        self.encoder = DistilBertModel.from_pretrained(distilbert_path)

        for param in self.encoder.parameters():
            param.requires_grad = False

        distilbert_hidden = 768

        self.policy_head = nn.Sequential(
            nn.Linear(distilbert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.ACTIONS))
        )

    def encode_state(self, texts: list) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding

    def forward(self, texts: list) -> torch.Tensor:
        embeddings = self.encode_state(texts)
        logits = self.policy_head(embeddings)
        return logits

    def get_action(self, text: str):
        logits = self.forward([text])
        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        action = self.ACTIONS[action_idx.item()]
        return action, log_prob, probs.detach().numpy()[0]


if __name__ == "__main__":
    print("Loading policy network...")
    policy = PokerPolicy()

    test_text = "You are in the BTN position with deep stacks. You have a premium hand."
    action, log_prob, probs = policy.get_action(test_text)

    print(f"\nState: {test_text}")
    print(f"Sampled action: {action}")
    print(f"Log probability: {log_prob.item():.4f}")
    print("\nFull distribution:")
    for a, p in zip(policy.ACTIONS, probs):
        print(f"  {a}: {p:.4f}")

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    print(f"\nTrainable parameters (policy head): {trainable:,}")
    print(f"Frozen parameters (DistilBERT): {frozen:,}")

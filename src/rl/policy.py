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

        # load the fine-tuned DistilBERT encoder (not the classification head)
        self.tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path)
        self.encoder = DistilBertModel.from_pretrained(distilbert_path)

        # freeze all DistilBERT parameters — no gradients flow through it
        for param in self.encoder.parameters():
            param.requires_grad = False

        # DistilBERT's hidden size is 768
        distilbert_hidden = 768

        # policy head: small MLP mapping embedding -> action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(distilbert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.ACTIONS))
        )

    def encode_state(self, texts: list) -> torch.Tensor:
        """
        Convert a batch of state descriptions (text) into DistilBERT
        embeddings. We use the [CLS] token representation — DistilBERT's
        standard way of summarizing a whole sequence into one vector.
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )

        # no_grad because DistilBERT is frozen — saves memory and compute
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # [CLS] token is the first token of the sequence output
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding

    def forward(self, texts: list) -> torch.Tensor:
        """
        Full forward pass: text -> embedding -> action probabilities.
        Returns raw logits (before softmax) — we apply softmax separately
        so we can also compute log-probabilities for the policy gradient.
        """
        embeddings = self.encode_state(texts)
        logits = self.policy_head(embeddings)
        return logits

    def get_action(self, text: str):
        """
        Sample an action from the policy for a single state.
        Returns the chosen action, its log-probability (needed for
        policy gradient updates), and the full probability distribution.
        """
        logits = self.forward([text])
        probs = torch.softmax(logits, dim=-1)

        # sample from the categorical distribution over actions
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()

        # log probability of the sampled action — this is what
        # the policy gradient update uses
        log_prob = dist.log_prob(action_idx)

        action = self.ACTIONS[action_idx.item()]
        return action, log_prob, probs.detach().numpy()[0]


if __name__ == "__main__":
    print("Loading policy network...")
    policy = PokerPolicy()

    # sanity check
    test_text = "You are in the BTN position with deep stacks. You have a premium hand."
    action, log_prob, probs = policy.get_action(test_text)

    print(f"\nState: {test_text}")
    print(f"Sampled action: {action}")
    print(f"Log probability: {log_prob.item():.4f}")
    print("\nFull distribution:")
    for a, p in zip(policy.ACTIONS, probs):
        print(f"  {a}: {p:.4f}")

    # count trainable vs frozen parameters
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    print(f"\nTrainable parameters (policy head): {trainable:,}")
    print(f"Frozen parameters (DistilBERT): {frozen:,}")

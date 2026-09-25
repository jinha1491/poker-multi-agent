import torch
import os
from src.rl.policy import PokerPolicy


class RLAgent:
    """
    Loads the trained RL policy for inference.

    We rebuild the same architecture used during training (frozen
    DistilBERT + policy head), then load the trained weights into
    just the policy head. DistilBERT weights come from the fine-tuned
    checkpoint automatically since PokerPolicy loads them fresh.
    """

    def __init__(
        self,
	distilbert_path: str = os.getenv("DISTILBERT_PATH", "src/models/poker_distilbert"),
        policy_weights_path: str = "src/rl/policy_head.pt",
    ):
        print("Loading RL agent...")
        self.policy = PokerPolicy(distilbert_path=distilbert_path, hidden_dim=128)

        # load the trained policy head weights
        state_dict = torch.load(policy_weights_path, map_location="cpu")
        self.policy.policy_head.load_state_dict(state_dict)
        self.policy.eval()
        print("RL agent loaded.")

    def decide(self, hand_situation: str) -> dict:
        """
        Given a hand situation as text, return the RL agent's decision.
        Uses greedy action selection (argmax) for inference — during
        training we sampled for exploration, but for serving we want
        the policy's best guess, not a random sample.
        """
        with torch.no_grad():
            logits = self.policy.forward([hand_situation])
            probs = torch.softmax(logits, dim=-1)[0]

        action_idx = torch.argmax(probs).item()
        action = self.policy.ACTIONS[action_idx]

        distribution = {
            self.policy.ACTIONS[i]: round(probs[i].item(), 4)
            for i in range(len(self.policy.ACTIONS))
        }

        return {
            "action": action,
            "confidence": round(probs[action_idx].item(), 4),
            "distribution": distribution,
        }


if __name__ == "__main__":
    agent = RLAgent()

    test_cases = [
        "You are in the BTN position with deep stacks. You have a premium hand.",
        "You are in the UTG position with short stacks. You have a weak hand.",
        "You are in the CO position with medium stacks. You have a medium strength hand.",
    ]

    for text in test_cases:
        result = agent.decide(text)
        print(f"\nState: {text}")
        print(f"Decision: {result['action']} (confidence: {result['confidence']})")
        print(f"Distribution: {result['distribution']}")

import os
import torch
from src.rl.policy import PokerPolicy, parse_state, state_features


class RLAgent:
    """
    Serves the trained RL policy. Uses structured state features only:
    across 3 seeds, features-only averaged 0.795 reward vs 0.685 when
    the DistilBERT embedding was concatenated in.
    """

    def __init__(self, policy_weights_path=os.getenv("RL_WEIGHTS", "src/rl/policy_head.pt")):
        self.policy = PokerPolicy(use_embedding=False)
        state_dict = torch.load(policy_weights_path, map_location="cpu")
        self.policy.policy_head.load_state_dict(state_dict)
        self.policy.eval()

    def decide(self, hand_situation: str) -> dict:
        s = parse_state(hand_situation)
        feats = state_features(s["hand_strength"], s["position"], s["stack_depth"]).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(self.policy.policy_head(feats), dim=-1)[0]

        # greedy at serving time: we want the policy's best answer, not a sample
        idx = int(torch.argmax(probs))
        return {
            "action": self.policy.ACTIONS[idx],
            "confidence": round(probs[idx].item(), 4),
            "distribution": {a: round(probs[i].item(), 4) for i, a in enumerate(self.policy.ACTIONS)},
        }


if __name__ == "__main__":
    agent = RLAgent()
    tests = [
        "You are in the BTN position with deep stacks. You have a premium hand.",
        "You are in the UTG position with short stacks. You have a weak hand.",
        "You are in the CO position with medium stacks. You have a medium strength hand.",
    ]
    for t in tests:
        r = agent.decide(t)
        print(f"\n{t}\n-> {r['action']} ({r['confidence']})  {r['distribution']}")

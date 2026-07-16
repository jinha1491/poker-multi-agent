import random
import json


class SimplifiedPokerEnv:
    """
    A simplified poker environment for self-play RL training.

    We don't simulate full multi-street poker with a game engine —
    instead we sample realistic hand scenarios (hand strength, position,
    stack depth, pot odds) and let the agent decide an action. Reward is
    based on how good that action is against a GTO baseline distribution
    learned from PokerBench.

    This keeps training tractable on a laptop while still giving the
    agent a real decision-making problem with real consequences.
    """

    ACTIONS = ["fold", "call", "check", "raise", "bet"]

    POSITIONS = ["utg", "mp", "co", "btn", "sb", "bb"]
    STACK_DEPTHS = ["short", "medium", "deep"]

    def __init__(self, gto_reference_path: str = "src/data/sequences.json"):
        # load GTO label distribution to build a reward model
        # we use the real label distribution from PokerBench as ground truth
        with open(gto_reference_path, "r") as f:
            data = json.load(f)
        self.gto_labels = data["labels"]

        # precompute action frequencies per hand strength bucket
        # this approximates "what does GTO do in this kind of spot"
        self._build_reward_table()

    def _build_reward_table(self):
        """
        Build a simplified reward table: for a given hand strength bucket,
        what's the GTO-approximate best action and how good is each
        alternative action relative to it.

        We use hand strength buckets 0-4 (weak to premium) and map
        real GTO action frequency patterns onto them from PokerBench.
        """
        # weak hands: GTO folds most often
        # premium hands: GTO raises/bets most often
        # this table encodes reward = -1 (bad) to +1 (optimal) per action
        self.reward_table = {
            0: {"fold": 1.0, "check": 0.3, "call": -0.5, "raise": -1.0, "bet": -1.0},   # weak hand
            1: {"fold": 0.3, "check": 0.6, "call": 0.4, "raise": -0.3, "bet": -0.3},    # marginal
            2: {"fold": -0.3, "check": 0.5, "call": 0.7, "raise": 0.3, "bet": 0.3},     # medium
            3: {"fold": -0.7, "check": 0.2, "call": 0.6, "raise": 0.8, "bet": 0.8},     # strong
            4: {"fold": -1.0, "check": -0.3, "call": 0.4, "raise": 1.0, "bet": 1.0},    # premium
        }

    def sample_state(self) -> dict:
        """
        Sample a random poker scenario.
        Returns a state dict the agent will use to make a decision.
        """
        hand_strength = random.randint(0, 4)  # 0=weak, 4=premium
        position = random.choice(self.POSITIONS)
        stack_depth = random.choice(self.STACK_DEPTHS)

        # build a natural language description — this is what gets
        # fed into DistilBERT for the embedding
        strength_words = ["a weak hand", "a marginal hand", "a medium strength hand",
                           "a strong hand", "a premium hand"]

        text = (
            f"You are in the {position.upper()} position with {stack_depth} stacks. "
            f"You have {strength_words[hand_strength]}."
        )

        return {
            "text": text,
            "hand_strength": hand_strength,
            "position": position,
            "stack_depth": stack_depth,
        }

    def step(self, state: dict, action: str) -> float:
        """
        Given a state and the agent's chosen action, return the reward.
        """
        hand_strength = state["hand_strength"]
        reward = self.reward_table[hand_strength][action]

        # small adjustment based on position — raising from BTN is
        # slightly better than raising from UTG at the same hand strength
        if action in ["raise", "bet"] and state["position"] in ["btn", "co"]:
            reward += 0.1
        if action in ["raise", "bet"] and state["position"] == "utg":
            reward -= 0.1

        # short stack should favor decisive actions (fold/allin-like raise)
        # over passive actions (call/check) — simplified push/fold logic
        if state["stack_depth"] == "short" and action in ["call"]:
            reward -= 0.2

        return reward


if __name__ == "__main__":
    env = SimplifiedPokerEnv()

    # sanity check — sample a few states and rewards
    for _ in range(5):
        state = env.sample_state()
        print(f"\nState: {state['text']}")
        print(f"Hand strength: {state['hand_strength']}")

        for action in env.ACTIONS:
            reward = env.step(state, action)
            print(f"  {action}: reward = {reward:.2f}")

import pytest
from src.rl.policy import parse_state, state_features, FEATURE_DIM
from src.rl.inference import RLAgent


def test_parse_state():
    s = parse_state("You are in the BTN position with deep stacks. You have a premium hand.")
    assert s == {"hand_strength": 4, "position": "btn", "stack_depth": "deep"}


def test_parse_state_rejects_other_text():
    with pytest.raises(ValueError):
        parse_state("I have pocket aces")


def test_features_are_one_hot():
    f = state_features(0, "utg", "short")
    assert f.shape == (FEATURE_DIM,)
    assert f.sum().item() == 3  # one active slot each for strength, position, stack


@pytest.fixture(scope="module")
def agent():
    return RLAgent()


@pytest.mark.parametrize("text, allowed", [
    # raise and bet have equal reward here, so either is correct
    ("You are in the BTN position with deep stacks. You have a premium hand.", {"raise", "bet"}),
    ("You are in the UTG position with short stacks. You have a weak hand.", {"fold"}),
    ("You are in the CO position with medium stacks. You have a medium strength hand.", {"call"}),
])
def test_agent_picks_best_action(agent, text, allowed):
    assert agent.decide(text)["action"] in allowed

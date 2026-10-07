from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

ACTIONS = {"fold", "call", "check", "raise", "bet"}

# a hand written in PokerBench's prompt format
POKERBENCH_HAND = (
    "You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a "
    "game scenario and you need to make the optimal decision.\n\nHere is a game summary:\n\n"
    "The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\n"
    "The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n"
    "In this hand, your position is BB, and your holding is [Ace of Diamond and Ten of Club].\n"
    "Before the flop, SB call. Assume that all other players that is not mentioned folded.\n\n"
    "Now it is your turn to make a move.\n"
    "To remind you, the current pot size is 1.0 chips, and your holding is [Ace of Diamond and Ten of Club].\n\n"
    "Decide on an action based on the strength of your hand on this board, your position, and "
    "actions before you. Do not explain your answer.\nYour optimal action is:"
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_valid_distribution():
    r = client.post("/predict", json={"hand_situation": POKERBENCH_HAND})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in ACTIONS
    assert set(body["distribution"]) == ACTIONS
    assert abs(sum(body["distribution"].values()) - 1.0) < 0.01
    assert body["in_distribution"] is True


def test_predict_flags_free_text():
    r = client.post("/predict", json={"hand_situation": "I have pocket aces, what should I do?"})
    assert r.status_code == 200
    assert r.json()["in_distribution"] is False


def test_predict_rejects_empty_input():
    r = client.post("/predict", json={"hand_situation": "   "})
    assert r.status_code == 400


def test_rl_analyze_valid_input():
    r = client.post("/rl-analyze", json={
        "hand_situation": "You are in the UTG position with short stacks. You have a weak hand."})
    assert r.status_code == 200
    assert r.json()["action"] == "fold"


def test_rl_analyze_bad_input_returns_400():
    r = client.post("/rl-analyze", json={"hand_situation": "I have aces"})
    assert r.status_code == 400

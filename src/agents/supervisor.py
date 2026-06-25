from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.agents.hand_analysis import analyze_hand
from src.agents.opponent_modeling import model_opponent
from src.agents.strategy import determine_strategy
from src.agents.explainability import explain_decision


# shared state object passed between all agents
class PokerState(TypedDict):
    hand_situation: str
    opponent_action: str
    hand_analysis: dict
    opponent_model: dict
    strategy: dict
    final_explanation: str


# agent nodes
def hand_analysis_node(state: PokerState) -> PokerState:
    result = analyze_hand(state["hand_situation"])
    return {"hand_analysis": result}


def opponent_modeling_node(state: PokerState) -> PokerState:
    result = model_opponent(state["hand_situation"], state["opponent_action"])
    return {"opponent_model": result}


def strategy_node(state: PokerState) -> PokerState:
    result = determine_strategy(
        state["hand_situation"],
        state["hand_analysis"],
        state["opponent_model"]
    )
    return {"strategy": result}


def explainability_node(state: PokerState) -> PokerState:
    result = explain_decision(
        state["hand_situation"],
        state["hand_analysis"],
        state["opponent_model"],
        state["strategy"]
    )
    return {"final_explanation": result}


# build the graph
def build_graph():
    graph = StateGraph(PokerState)

    graph.add_node("hand_analysis", hand_analysis_node)
    graph.add_node("opponent_modeling", opponent_modeling_node)
    graph.add_node("strategy", strategy_node)
    graph.add_node("explainability", explainability_node)

    graph.set_entry_point("hand_analysis")
    graph.add_edge("hand_analysis", "opponent_modeling")
    graph.add_edge("opponent_modeling", "strategy")
    graph.add_edge("strategy", "explainability")
    graph.add_edge("explainability", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "hand_situation": "You have AKs on the button. UTG raises to 3bb, everyone else folds to you.",
        "opponent_action": "UTG raises to 3bb",
        "hand_analysis": {},
        "opponent_model": {},
        "strategy": {},
        "final_explanation": ""
    })

    print("\n--- HAND ANALYSIS ---")
    print(result["hand_analysis"])
    print("\n--- OPPONENT MODEL ---")
    print(result["opponent_model"])
    print("\n--- STRATEGY ---")
    print(result["strategy"])
    print("\n--- FINAL EXPLANATION ---")
    print(result["final_explanation"])

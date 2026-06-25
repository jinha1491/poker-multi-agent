import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="poker_strategy")


def determine_strategy(hand_situation: str, hand_analysis: dict, opponent_model: dict) -> dict:
    results = collection.query(
        query_texts=[hand_situation],
        n_results=5
    )

    retrieved_context = "\n\n".join(
        f"Scenario: {doc}\nGTO Action: {meta['optimal_action']}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    prompt = f"""You are a poker strategy specialist. Determine the optimal action based on the analysis below.

Hand situation:
{hand_situation}

Hand analysis:
{hand_analysis}

Opponent model:
{opponent_model}

GTO reference scenarios with their optimal actions:
{retrieved_context}

Based on all of the above, provide:
1. Recommended action (fold/call/raise)
2. Sizing if raising (e.g. 2.5x, 3x, pot)
3. GTO frequency breakdown (e.g. raise 70%, call 30%)
4. Exploitative adjustment based on opponent model

Respond in this exact format:
ACTION: <value>
SIZING: <value>
GTO_FREQUENCY: <value>
EXPLOITATIVE_ADJUSTMENT: <value>
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    output = response.choices[0].message.content

    strategy = {}
    for line in output.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            strategy[key.strip().lower()] = value.strip()

    return strategy


if __name__ == "__main__":
    test_hand = "You have AKs on the button. UTG raises to 3bb, everyone else folds to you."
    test_hand_analysis = {
        "hand_strength": "premium",
        "position": "favorable",
        "equity": "65%"
    }
    test_opponent_model = {
        "player_type": "tight-aggressive",
        "range_estimate": "AA, KK, QQ, JJ, AK, AQs",
        "exploitative_note": "apply pressure with bluffs"
    }
    result = determine_strategy(test_hand, test_hand_analysis, test_opponent_model)
    print(result)

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="poker_strategy")


def model_opponent(hand_situation: str, opponent_action: str) -> dict:
    query = f"{hand_situation} Opponent action: {opponent_action}"

    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    retrieved_context = "\n\n".join(results["documents"][0])

    prompt = f"""You are a poker opponent modeling specialist. Analyze the opponent's likely range and tendencies based on their action.

Hand situation:
{hand_situation}

Opponent action:
{opponent_action}

Similar GTO scenarios for reference:
{retrieved_context}

Based on the opponent's action and similar scenarios, provide:
1. Player type (tight-passive/tight-aggressive/loose-passive/loose-aggressive)
2. Estimated range (what hands they likely have)
3. Exploitative note (how to adjust against this type of player)

Respond in this exact format:
PLAYER_TYPE: <value>
RANGE_ESTIMATE: <value>
EXPLOITATIVE_NOTE: <value>
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    output = response.choices[0].message.content

    analysis = {}
    for line in output.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            analysis[key.strip().lower()] = value.strip()

    return analysis


if __name__ == "__main__":
    test_hand = "You have AKs on the button."
    test_action = "UTG raises to 3bb"
    result = model_opponent(test_hand, test_action)
    print(result)

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="poker_strategy")


def analyze_hand(hand_situation: str) -> dict:
    results = collection.query(
        query_texts=[hand_situation],
        n_results=5
    )

    retrieved_context = "\n\n".join(results["documents"][0])

    prompt = f"""You are a poker hand analysis specialist. Analyze the following hand situation.

Hand situation:
{hand_situation}

Similar GTO scenarios for reference:
{retrieved_context}

Based on the hand situation and similar scenarios, provide:
1. Hand strength (weak/medium/strong/premium)
2. Position assessment (favorable/neutral/unfavorable)
3. Estimated equity (rough percentage)
4. Key factors affecting this decision

Respond in this exact format:
HAND_STRENGTH: <value>
POSITION: <value>
EQUITY: <value>
KEY_FACTORS: <value>
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
    test_hand = "You have AKs on the button. UTG raises to 3bb, everyone else folds to you."
    result = analyze_hand(test_hand)
    print(result)

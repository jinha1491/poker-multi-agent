from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def explain_decision(
    hand_situation: str,
    hand_analysis: dict,
    opponent_model: dict,
    strategy: dict
) -> str:

    prompt = f"""You are a poker coach explaining a decision to a student. 
Use clear, concise language. Focus on the reasoning, not just the conclusion.

Hand situation:
{hand_situation}

Hand analysis:
{hand_analysis}

Opponent model:
{opponent_model}

Recommended strategy:
{strategy}

Write a clear explanation of why this is the correct play. Cover:
1. Why this hand is strong or weak in this spot
2. What the opponent's action tells us about their range
3. Why the recommended action is correct from a GTO perspective
4. Any exploitative adjustments worth making

Keep it under 150 words. Write like a coach, not a textbook.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content


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
    test_strategy = {
        "action": "raise",
        "sizing": "3x",
        "gto_frequency": "raise 80%, call 20%",
        "exploitative_adjustment": "apply pressure with larger raise"
    }
    result = explain_decision(test_hand, test_hand_analysis, test_opponent_model, test_strategy)
    print(result)

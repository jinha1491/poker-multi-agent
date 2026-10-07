from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.inference import RLAgent
from src.models.predictor import ActionPredictor

# /analyze calls OpenAI on every request, so public deployments can switch it off
ENABLE_ANALYZE = os.getenv("ENABLE_ANALYZE", "true").lower() == "true"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rl_agent = RLAgent()
predictor = ActionPredictor()

graph = None
if ENABLE_ANALYZE:
    from agents.supervisor import build_graph
    graph = build_graph()


class HandRequest(BaseModel):
    hand_situation: str
    opponent_action: str


class RLRequest(BaseModel):
    hand_situation: str


class PredictRequest(BaseModel):
    hand_situation: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rl-analyze")
def rl_analyze(request: RLRequest):
    try:
        return rl_agent.decide(request.hand_situation)
    except ValueError as e:
        # input didn't match the expected format, so it's a client error, not a crash
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
def predict(request: PredictRequest):
    """Fine-tuned DistilBERT: predicts the solver's action for a PokerBench-format hand."""
    if not request.hand_situation.strip():
        raise HTTPException(status_code=400, detail="hand_situation is empty")
    return predictor.predict(request.hand_situation)


@app.post("/analyze")
def analyze(request: HandRequest):
    if graph is None:
        raise HTTPException(status_code=503,
                            detail="/analyze is turned off in this deployment. Run the project locally to use it.")

    result = graph.invoke({
        "hand_situation": request.hand_situation,
        "opponent_action": request.opponent_action,
        "hand_analysis": {},
        "opponent_model": {},
        "strategy": {},
        "final_explanation": ""
    })

    def stream_response():
        yield f"data: {json.dumps({'type': 'hand_analysis', 'data': result['hand_analysis']})}\n\n"
        yield f"data: {json.dumps({'type': 'opponent_model', 'data': result['opponent_model']})}\n\n"
        yield f"data: {json.dumps({'type': 'strategy', 'data': result['strategy']})}\n\n"

        words = result["final_explanation"].split()
        explanation_so_far = ""
        for word in words:
            explanation_so_far += word + " "
            yield f"data: {json.dumps({'type': 'explanation', 'data': explanation_so_far.strip()})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

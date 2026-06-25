from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.supervisor import build_graph

app = FastAPI()

# allow React frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


class HandRequest(BaseModel):
    hand_situation: str
    opponent_action: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: HandRequest):
    result = graph.invoke({
        "hand_situation": request.hand_situation,
        "opponent_action": request.opponent_action,
        "hand_analysis": {},
        "opponent_model": {},
        "strategy": {},
        "final_explanation": ""
    })

    def stream_response():
        # stream each agent output one by one
        yield f"data: {json.dumps({'type': 'hand_analysis', 'data': result['hand_analysis']})}\n\n"
        yield f"data: {json.dumps({'type': 'opponent_model', 'data': result['opponent_model']})}\n\n"
        yield f"data: {json.dumps({'type': 'strategy', 'data': result['strategy']})}\n\n"

        # stream explanation word by word
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

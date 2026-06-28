import { useState } from "react";
import axios from "axios";
import "./App.css";

interface AgentOutputs {
  hand_analysis: Record<string, string>;
  opponent_model: Record<string, string>;
  strategy: Record<string, string>;
  explanation: string;
}

function App() {
  const [handSituation, setHandSituation] = useState("");
  const [opponentAction, setOpponentAction] = useState("");
  const [loading, setLoading] = useState(false);
  const [outputs, setOutputs] = useState<AgentOutputs>({
    hand_analysis: {},
    opponent_model: {},
    strategy: {},
    explanation: "",
  });

  const analyze = async () => {
    setLoading(true);
    setOutputs({ hand_analysis: {}, opponent_model: {}, strategy: {}, explanation: "" });

    const response = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hand_situation: handSituation, opponent_action: opponentAction }),
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n").filter((line) => line.startsWith("data: "));

      for (const line of lines) {
        const data = line.replace("data: ", "");
        if (data === "[DONE]") {
          setLoading(false);
          break;
        }

        try {
          const parsed = JSON.parse(data);
          if (parsed.type === "hand_analysis") {
            setOutputs((prev) => ({ ...prev, hand_analysis: parsed.data }));
          } else if (parsed.type === "opponent_model") {
            setOutputs((prev) => ({ ...prev, opponent_model: parsed.data }));
          } else if (parsed.type === "strategy") {
            setOutputs((prev) => ({ ...prev, strategy: parsed.data }));
          } else if (parsed.type === "explanation") {
            setOutputs((prev) => ({ ...prev, explanation: parsed.data }));
          }
        } catch {}
      }
    }
  };

  return (
    <div className="app">
      <h1>Poker Strategy Advisor</h1>
      <p className="subtitle">Multi-agent GTO analysis powered by AI</p>

      <div className="input-section">
        <textarea
          placeholder="Describe your hand situation (e.g. You have AKs on the button. UTG raises to 3bb.)"
          value={handSituation}
          onChange={(e) => setHandSituation(e.target.value)}
          rows={3}
        />
        <textarea
          placeholder="Describe opponent action (e.g. UTG raises to 3bb)"
          value={opponentAction}
          onChange={(e) => setOpponentAction(e.target.value)}
          rows={2}
        />
        <button onClick={analyze} disabled={loading || !handSituation || !opponentAction}>
          {loading ? "Analyzing..." : "Analyze Hand"}
        </button>
      </div>

      {outputs.hand_analysis && Object.keys(outputs.hand_analysis).length > 0 && (
        <div className="results">
          <div className="agent-card">
            <h2>Hand Analysis</h2>
            {Object.entries(outputs.hand_analysis).map(([key, value]) => (
              <p key={key}><strong>{key.replace(/_/g, " ")}:</strong> {value}</p>
            ))}
          </div>

          <div className="agent-card">
            <h2>Opponent Model</h2>
            {Object.entries(outputs.opponent_model).map(([key, value]) => (
              <p key={key}><strong>{key.replace(/_/g, " ")}:</strong> {value}</p>
            ))}
          </div>

          <div className="agent-card">
            <h2>Strategy</h2>
            {Object.entries(outputs.strategy).map(([key, value]) => (
              <p key={key}><strong>{key.replace(/_/g, " ")}:</strong> {value}</p>
            ))}
          </div>

          {outputs.explanation && (
            <div className="agent-card explanation">
              <h2>Coach Explanation</h2>
              <p>{outputs.explanation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;

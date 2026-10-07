# Agentic AI Component & Tool Orchestration

## 1. Role of Agentic AI

In contrast to standalone predictive algorithms (e.g. running an XGBoost script), **Agentic AI** serves as the intelligent reasoning and orchestration layer that:
1. Receives complex natural-language queries from farmers.
2. Decides dynamically which external tools or ML inference routines need to be invoked.
3. Chains multi-step operations (e.g., retrieving coordinates -> querying Open-Meteo -> feeding weather into Yield Model -> interpreting output).
4. Explains model predictions in understandable agronomic terms with scientific integrity.

---

## 2. Agent Tool Inventory

The agent is provided with five native schema tools declared in [`backend/app/services/agent_service.py`](file:///backend/app/services/agent_service.py):

```json
[
  {
    "name": "get_weather",
    "description": "Fetch current weather conditions and annual climate estimates for a given latitude/longitude."
  },
  {
    "name": "predict_yield",
    "description": "Predict crop yield (tonnes/ha) for given crop, location, soil, and weather."
  },
  {
    "name": "recommend_crop",
    "description": "Recommend the most suitable crop given soil (N-P-K-pH) and climatic conditions."
  },
  {
    "name": "compare_crops",
    "description": "Compare two or more crops under identical soil and climatic constraints."
  },
  {
    "name": "get_known_options",
    "description": "Returns lists of supported crops, states, and seasons in the trained model."
  }
]
```

---

## 3. Strict Grounding Principles
- **No Hallucination**: The agent is programmatically restricted from creating arbitrary numeric yield estimates or inventing weather parameters.
- **Evidence-Based Explanations**: All advisory statements are grounded in model outputs and validated agronomic relationships.
- **Graceful Error Handling**: If an API or model parameter is missing, the agent clearly states the missing requirements rather than substituting arbitrary defaults.

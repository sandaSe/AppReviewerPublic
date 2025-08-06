from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import AzureChatOpenAI
from functools import partial
from dotenv import load_dotenv
import os
import json
from typing import Dict, List, Any

load_dotenv()

# ---- 1. Define the shared state ----
class ReviewState(TypedDict):
    review_text: str
    rating: int
    initial_tags: Dict[str, float]
    active_agents: List[str]
    agent_outputs: Dict[str, Any]
    aggregator_result: Dict[str, Any]
    final_decision: List[str]
    notes: str
    prediction: Dict[str, Any]  


# ---- 2. LLM Setup ----
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_API_KEY")
azure_deployment = os.getenv("AZURE_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_API_VERSION")

llm = AzureChatOpenAI(
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_key=azure_api_key,
)


# ---- 3. Agent Functions ----
def classifier_agent(state: ReviewState) -> ReviewState:
    review = state["review_text"]
    rating = state.get("rating", "N/A")

    prompt = f"""
                    You are an expert in analyzing mobile app user reviews. Your task is to classify each review into ONE of the following four categories, using both the content of the review and the user’s star rating (1 to 5):

                    Review: "{review}"
                    User Rating: {rating}

                    Here are the categories:

                    1. **Bug**
                    - The review describes a technical problem, crash, malfunction, or unexpected app behavior.
                    - Look for mentions of errors, failures, glitches, or something not working as expected.

                    2. **Feature**
                    - The review requests, suggests, or implies a need for a new feature or an improvement to existing functionality.
                    - May contain phrases like “should add”, “why not have”, “I wish”, “missing”, “need option to…”

                    3. **UserExperience**
                    - The review provides detailed feedback on usability, interface, layout, design, or how a feature made them feel.
                    - DO NOT select this if the review is just requesting a feature or reporting a bug.

                    4. **Rating**
                    - The review is very brief and mainly reflects an overall impression without specific details.

                    Instructions:
                    1. Decide the best matching category.
                    2. Justify your choice briefly.
                    3. Give confidence score (0.0 to 1.0).

                    Respond ONLY in this JSON format:
                    {{"label": "<Bug/Feature/UserExperience/Rating>", "confidence": float, "reason": "short explanation"}}
                    """

    response = llm.invoke(prompt)
    try:
        prediction = json.loads(response.content)
    except Exception:
        prediction = {"label": None, "confidence": 0.0, "reason": "Failed to parse response"}

    state["prediction"] = prediction
    return state

def uncertainty_agent(state: ReviewState) -> ReviewState:
    if "agent_trace" not in state:
        state["agent_trace"] = []

    pred = state.get("prediction", {})
    confidence = pred.get("confidence", 0)
    label = pred.get("label", "Unknown")

    if confidence < 0.7 and state.get("uncertainty_count", 0) < 3:
        review = state["review_text"]
        rating = state.get("rating", "N/A")

        prompt = f"""
                        You are an uncertainty resolution agent reviewing a low-confidence classification.

                        Previous Classification: "{label}" (confidence: {confidence})
                        Reason: "{pred.get('reason', '')}"

                        Review: "{review}"
                        User Rating: {rating}

                        Instructions:
                        1. Rethink your decision independently.
                        2. Provide your label, explanation, and confidence.

                        Respond in JSON ONLY:
                        {{"label": "<Bug|Feature|UserExperience|Rating>", "confidence": float, "reason": "brief justification"}}
                        """

        state["uncertainty_count"] = state.get("uncertainty_count", 0) + 1
        response = llm.invoke(prompt)
        try:
            new_pred = json.loads(response.content)
        except Exception:
            new_pred = pred  # fallback to previous
        state["prediction"] = new_pred
        state["notes"] = "Uncertainty agent revised prediction."
        state["agent_trace"].append({
            "agent": "uncertainty_agent",
            "label": new_pred.get("label"),
            "confidence": new_pred.get("confidence"),
            "reason": new_pred.get("reason")
        })
    else:
        state["notes"] = "Uncertainty agent skipped."

    return state


def optimizer_agent(state: ReviewState) -> ReviewState:
    if "agent_trace" not in state:
        state["agent_trace"] = []
    # If uncertainty agent lowered confidence, try rerouting or refining prompt logic (this is a stub)
    pred = state.get("prediction", {})
    confidence = pred.get("confidence", 0)
   # label = pred.get("label", "Unknown")

    if confidence < 0.7:
        # For example, tweak prompt or add instructions for re-classification
        review = state["review_text"]
        rating = state.get("rating", "N/A")

        trace = state.get("agent_trace", [])
        trace_str = json.dumps(trace, indent=2)

        prompt = f"""
        You are an optimizer agent. Review the following trace of decisions and provide a final classification.

        Agent trace: {trace_str}

        Review: "{review}"
        Rating: {rating}

        Respond in JSON ONLY:
        {{"label": "<Bug|Feature|UserExperience|Rating>", "confidence": float, "reason": "refined explanation"}}
        """
        response = llm.invoke(prompt)
        try:
            final_pred = json.loads(response.content)
        except Exception:
            final_pred = pred

        state["prediction"] = final_pred
        state["notes"] += " Optimizer applied memory refinement."
        state.setdefault("agent_trace", []).append({
            "agent": "optimizer_agent",
            "label": final_pred.get("label"),
            "confidence": final_pred.get("confidence"),
            "reason": final_pred.get("reason")
        })
    else:
        state["notes"] += " Optimizer skipped due to high confidence."

    return state


def label_selector(state: ReviewState) -> ReviewState:
    pred = state.get("prediction", {})
    state["final_decision"] = [pred.get("label")] if pred.get("label") else []
    return state


# ---- 4. Routing Logic ----
def should_check_uncertainty(state: ReviewState) -> bool:
    pred = state.get("prediction", {})
    confidence = pred.get("confidence", 0)
    label = pred.get("label", "")

    # Smart routing conditions
    if 0.4 <= confidence <= 0.85:
        return True
    if label in {"Rating", "UserExperience"} and confidence < 0.8:
        return True
    if label == "Feature" and confidence < 0.7:
        return True
    return False


def route_after_classifier(state: ReviewState) -> str:
    return "uncertainty_agent" if should_check_uncertainty(state) else "label_selector"

def route_after_uncertainty(state: ReviewState) -> str:
    return "optimizer_agent"

def route_after_optimizer(state: ReviewState) -> str:
    return "label_selector"


# ---- 5. Build LangGraph ----
builder = StateGraph(ReviewState)

builder.add_node("classifier_agent", classifier_agent)
builder.add_node("uncertainty_agent", uncertainty_agent)
builder.add_node("optimizer_agent", optimizer_agent)
builder.add_node("label_selector", label_selector)

builder.set_entry_point("classifier_agent")

# Add conditional edges with routing functions
builder.add_conditional_edges("classifier_agent", route_after_classifier)
builder.add_edge("uncertainty_agent", "optimizer_agent")
builder.add_edge("optimizer_agent", "label_selector")
builder.add_edge("label_selector", END)

graph = builder.compile()


# ---- 6. Run batch processing ----
if __name__ == "__main__":
    input_path = ""
    output_path = ""

    with open(input_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    valid_reviews = [item for item in reviews if item.get("comment") or item.get("rating")]
    print(f"Total valid reviews: {len(valid_reviews)}")

    outputs = []

    for idx, item in enumerate(valid_reviews, 1):
        comment = str(item.get("comment", "")).strip()
        rating = item.get("rating", None)

        state = {
            "review_text": comment,
            "rating": rating,
            "initial_tags": {},
            "active_agents": [],
            "agent_outputs": {},
            "aggregator_result": {},
            "final_decision": [],
            "notes": "",
            "prediction": {},
            "uncertainty_count": 0,
            "agent_trace": []
        }

        result = graph.invoke(state)

        if "agent_trace" not in result:
            result["agent_trace"] = []

        final_label = result.get("final_decision", [])
        label_val = final_label[0] if final_label else None

        result["agent_trace"].insert(0, {
            "agent": "classifier_agent",
            "label": label_val,
            "confidence": result.get("prediction", {}).get("confidence", 0),
            "reason": "Initial classification"
        })


        processed = {
            "review_text": result.get("review_text"),
            "rating": result.get("rating"),
            "final_decision": result.get("final_decision", []),
            "notes": result.get("notes", ""),
            "prediction": result.get("prediction", {}),
            "true_label": item.get("reviewAnnotatorLabel"),
            "uncertainty_count": result.get("uncertainty_count", 0),
            "agent_trace": result.get("agent_trace", [])
        }

        outputs.append(processed)
        print(f"Processed {idx}/{len(valid_reviews)} reviews...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(outputs)} processed reviews to {output_path}")

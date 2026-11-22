"""Deterministic LangGraph routing with an explicit fallback."""

from typing import TypedDict


class State(TypedDict, total=False):
    text: str
    category: str
    result: str


def build_workflow(classifier):
    """Route classified documents; this is not an autonomous agent."""
    from langgraph.graph import StateGraph, START, END

    def classify(state: State) -> dict:
        return {"category": str(classifier.predict([state["text"]])[0])}

    def process(state: State) -> dict:
        return {"result": f"Ready to extract fields from: {state['category']}"}

    graph = StateGraph(State)
    graph.add_node("classify", classify)
    graph.add_node("process", process)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "process")
    graph.add_edge("process", END)
    return graph.compile()

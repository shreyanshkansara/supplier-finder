"""The sourcing workflow as a LangGraph StateGraph.

    parse -> completeness --missing--> clarify (PAUSE) --> completeness
                         \\--complete--> retrieve -> verify -> rank -> report

The result is a list of vendors that can supply the requirement (with contact details) and the
near misses. The one pause is for the buyer to answer clarifying questions.
"""

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import AgentDeps, Nodes
from app.agent.state import GraphState


def build_graph(deps: AgentDeps) -> StateGraph:
    n = Nodes(deps)
    g = StateGraph(GraphState)
    for name in ("parse", "completeness", "clarify", "retrieve", "verify", "rank", "report"):
        g.add_node(name, getattr(n, name))

    def after_completeness(s: dict) -> str:
        if s["missing"] and s.get("clarify_rounds", 0) < deps.max_clarify_rounds:
            return "clarify"
        return "retrieve"

    g.add_edge(START, "parse")
    g.add_edge("parse", "completeness")
    g.add_conditional_edges("completeness", after_completeness, ["clarify", "retrieve"])
    g.add_edge("clarify", "completeness")
    g.add_edge("retrieve", "verify")
    g.add_edge("verify", "rank")
    g.add_edge("rank", "report")
    g.add_edge("report", END)
    return g

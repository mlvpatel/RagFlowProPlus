"""The agentic RAG graph: a bounded, self-correcting LangGraph state machine.

Flow: retrieve, then grade. If the grade is weak, rewrite the query and retrieve
again up to a bounded number of attempts, then optionally fall back to web
search, then generate. After generating, self-check the answer for grounding and
regenerate once if it is not grounded. Every branch is bounded, so the loop
always terminates, which is the cost guard from the enterprise design.
"""

import logging

from langgraph.graph import END, START, StateGraph

from src.agent import nodes
from src.agent.state import AgentState
from src.core.config import settings
from src.core.langchain_utils import _make_llm, _reformulate_query, _to_lc_messages

logger = logging.getLogger(__name__)


def _decide_after_grade(state) -> str:
    if state.get("grade") == "relevant":
        return "generate"
    if state.get("attempts", 0) < settings.agent_max_retrieval_attempts:
        return "rewrite"
    if settings.agent_enable_web and not state.get("used_web"):
        return "web_search"
    return "generate"


def _decide_after_check(state) -> str:
    # Regenerate once only when the answer is ungrounded AND the evidence was
    # weak. When the grade was relevant, trust the strong retrieval and stop, so
    # a small local model's noisy self-check does not cause needless regeneration.
    if (
        not state.get("grounded", True)
        and state.get("grade") == "weak"
        and state.get("generations", 0) < 2
    ):
        return "generate"
    return END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade_documents)
    graph.add_node("rewrite", nodes.rewrite_query)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("generate", nodes.generate)
    graph.add_node("self_check", nodes.self_check)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _decide_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", "self_check")
    graph.add_conditional_edges(
        "self_check", _decide_after_check, {"generate": "generate", END: END}
    )
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(model: str, question: str, chat_history=None) -> dict:
    """Run the agentic RAG once and return the answer, context, and trace.

    When there is chat history the question is first reformulated into a
    standalone query, so conversational follow ups still work.
    """
    history = _to_lc_messages(chat_history)
    query = (
        _reformulate_query(_make_llm(model, temperature=0), question, history)
        if history
        else question
    )
    initial = {
        "model": model,
        "question": question,
        "chat_history": history,
        "query": query,
        "attempts": 0,
        "generations": 0,
        "steps": [],
    }
    final = get_graph().invoke(
        initial, config={"recursion_limit": settings.agent_max_steps}
    )
    return {
        "answer": final.get("answer", ""),
        "context": final.get("documents", []),
        "steps": final.get("steps", []),
        "confidence": final.get("confidence"),
        "used_web": final.get("used_web", False),
    }

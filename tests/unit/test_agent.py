"""Unit tests for the agentic graph control flow.

These test the routing logic and parsing in isolation, with no model or database,
so they run in CI without Ollama or Postgres.
"""

from langgraph.graph import END

from src.agent import graph, nodes
from src.core.config import settings


def test_parse_json_extracts_object():
    assert nodes._parse_json('noise {"a": 1} tail', {}) == {"a": 1}
    assert nodes._parse_json("not json at all", {"x": 2}) == {"x": 2}


def test_decide_after_grade_relevant_generates():
    assert graph._decide_after_grade({"grade": "relevant"}) == "generate"


def test_decide_after_grade_weak_rewrites_when_attempts_left():
    assert graph._decide_after_grade({"grade": "weak", "attempts": 0}) == "rewrite"


def test_decide_after_grade_weak_generates_when_exhausted():
    state = {"grade": "weak", "attempts": settings.agent_max_retrieval_attempts}
    assert graph._decide_after_grade(state) == "generate"


def test_decide_after_check_ends_when_grounded():
    assert graph._decide_after_check({"grounded": True}) == END


def test_decide_after_check_regenerates_when_ungrounded_and_weak():
    state = {"grounded": False, "grade": "weak", "generations": 1}
    assert graph._decide_after_check(state) == "generate"


def test_decide_after_check_trusts_strong_retrieval():
    state = {"grounded": False, "grade": "relevant", "generations": 1}
    assert graph._decide_after_check(state) == END


def test_decide_after_check_stops_after_two_generations():
    state = {"grounded": False, "grade": "weak", "generations": 2}
    assert graph._decide_after_check(state) == END


def test_graph_compiles():
    assert graph.build_graph() is not None


class _RecorderLLM:
    """Plain fake chat model that records the last human message it saw."""

    def __init__(self, reply: str):
        self.reply = reply
        self.last_prompt = None

    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage

        self.last_prompt = messages[-1].content
        return AIMessage(content=self.reply)


def test_nodes_operate_on_the_standalone_query(monkeypatch):
    """With history, run_agent seeds state["query"] with the rewrite; every
    node that reads the question must use it, or the pronoun leaks back in."""
    from langchain_core.documents import Document

    import src.agent.nodes as nodes

    seen = {}

    class _Retriever:
        def invoke(self, q):
            seen["retrieve"] = q
            return [Document(page_content="ctx")]

    monkeypatch.setattr(nodes, "get_final_retriever", lambda: _Retriever())
    state = {
        "model": "m",
        "question": "and its price?",
        "query": "price of Nimbus Pro?",
    }
    out = nodes.retrieve(state)
    assert seen["retrieve"] == "price of Nimbus Pro?"

    grader = _RecorderLLM('{"relevant": true, "confidence": 0.9, "reason": "ok"}')
    monkeypatch.setattr(nodes, "_make_llm", lambda *a, **k: grader)
    nodes.grade_documents({**state, "documents": out["documents"]})
    assert "price of Nimbus Pro?" in grader.last_prompt
    assert "and its price?" not in grader.last_prompt

    answerer = _RecorderLLM("the price is 42")
    monkeypatch.setattr(nodes, "_make_llm", lambda *a, **k: answerer)
    nodes.generate({**state, "documents": out["documents"]})
    assert answerer.last_prompt == "price of Nimbus Pro?"


def test_rewrite_iterates_from_the_current_query(monkeypatch):
    import src.agent.nodes as nodes

    rewriter = _RecorderLLM("better query")
    monkeypatch.setattr(nodes, "_make_llm", lambda *a, **k: rewriter)
    out = nodes.rewrite_query(
        {"model": "m", "question": "raw", "query": "first rewrite"}
    )
    assert rewriter.last_prompt == "first rewrite"
    assert out["query"] == "better query"

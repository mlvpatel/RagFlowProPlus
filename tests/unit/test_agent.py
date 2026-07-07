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

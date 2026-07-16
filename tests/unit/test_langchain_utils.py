"""Unit tests for the chain utilities (fake chat models, no network)."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

import src.core.langchain_utils as lu


def test_make_llm_routes_local_models_to_ollama(monkeypatch):
    monkeypatch.setattr(lu.settings, "ollama_base_url", "http://localhost:11434")
    llm = lu._make_llm("llama3.2:3b")
    assert llm.__class__.__name__ == "ChatOllama"


def test_to_lc_messages_maps_roles():
    history = [
        {"role": "human", "content": "q"},
        {"role": "ai", "content": "a"},
    ]
    msgs = lu._to_lc_messages(history)
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert lu._to_lc_messages(None) == []


def test_reformulate_query_skips_without_history():
    fake_llm = FakeListChatModel(responses=["should not be used"])
    assert lu._reformulate_query(fake_llm, "hello", []) == "hello"


def test_reformulate_query_rewrites_with_history():
    fake_llm = FakeListChatModel(responses=["standalone question"])
    history = lu._to_lc_messages([{"role": "human", "content": "earlier"}])
    assert (
        lu._reformulate_query(fake_llm, "follow up", history) == "standalone question"
    )

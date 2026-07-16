"""
RAG chain for rag-agentic-2025, built directly on langchain-core (LCEL).

Building on langchain-core rather than the legacy langchain.chains helpers
keeps this stable across langchain major versions. The helpers here feed the
agent graph: model routing, history conversion, standalone-question rewriting,
and the retriever stack. The LLM provider is chosen from the model name, so
the same code serves OpenAI, Anthropic, and local Ollama models.
"""

import logging
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.core.config import settings
from src.embeddings.vectorstore_utils import get_query_embeddings
from src.retrieval.retrievers import HybridRetriever, ReRankingRetriever

logger = logging.getLogger(__name__)

CONTEXTUALIZE_SYSTEM = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

_contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _make_llm(model: str, temperature: float | None = None):
    """Return a chat model for the given model name, provider chosen by name.

    The agent passes temperature 0 so grading and self checking are
    deterministic and reproducible rather than varying run to run.
    """
    name = model.lower()
    if any(
        tag in name for tag in ("llama", "qwen", "deepseek", "mistral", "gemma", "phi")
    ):
        from langchain_ollama import ChatOllama

        kwargs = {"model": model, "base_url": settings.ollama_base_url}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOllama(**kwargs)
    if "claude" in name:
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)
    from langchain_openai import ChatOpenAI

    kwargs = {"model": model, "api_key": settings.openai_api_key}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def get_final_retriever():
    """The hybrid retriever, wrapped in the cross encoder reranker when enabled."""
    base = HybridRetriever(embeddings=get_query_embeddings(), k=settings.top_k)
    if settings.use_reranker:
        return ReRankingRetriever(base_retriever=base, top_n=settings.reranker_top_n)
    return base


def warm_reranker() -> None:
    """Load the reranker at startup so the first chat request is not slow.

    Never fatal: if the reranker cannot load (for example torch is not
    installed in this environment), log a warning and continue.
    """
    if not settings.use_reranker:
        return
    try:
        from src.retrieval.retrievers import get_reranker

        get_reranker()
    except Exception as exc:
        logger.warning("Reranker warm-up skipped: %s", exc)


def _to_lc_messages(chat_history) -> List[Any]:
    """Convert stored {role, content} dicts into langchain message objects."""
    messages: List[Any] = []
    for turn in chat_history or []:
        if turn.get("role") in ("ai", "assistant"):
            messages.append(AIMessage(content=turn["content"]))
        else:
            messages.append(HumanMessage(content=turn["content"]))
    return messages


def _reformulate_query(llm, user_input: str, history: List[Any]) -> str:
    """Rewrite the question to be standalone, skipped when there is no history."""
    if not history:
        return user_input
    chain = _contextualize_prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input, "chat_history": history})


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

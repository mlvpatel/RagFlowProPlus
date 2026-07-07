"""Nodes of the agentic RAG graph.

Each node is one Thought, Action, Observation step from the planning diagram:
retrieve (action), grade (observation and plan validation), rewrite (replan),
web_search (backtrack to an outside source), generate (answer), and self_check
(verify grounding). Every node appends a compact entry to the trace.
"""

import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.tools import web_search_docs
from src.core.config import settings
from src.core.langchain_utils import _format_context, _make_llm, get_final_retriever

logger = logging.getLogger(__name__)

_GRADER_SYSTEM = (
    "You grade whether retrieved context can answer a question. "
    'Return ONLY compact JSON: {"relevant": true or false, '
    '"confidence": a number from 0 to 1, "reason": "short"}. '
    "Set relevant to true only if the context clearly contains the answer."
)

_REWRITE_SYSTEM = (
    "Rewrite the user question into a better search query for a document "
    "retriever. Keep it concise, add key entities and synonyms. Return ONLY "
    "the rewritten query, nothing else."
)

_ANSWER_SYSTEM = (
    "You are a careful assistant. Answer the question using ONLY the provided "
    "context. If the context does not contain the answer, say you do not have "
    "that information. Do not invent facts.\n\nContext:\n{context}"
)

_CHECK_SYSTEM = (
    "You verify that an answer is grounded in the provided context. "
    'Return ONLY compact JSON: {"grounded": true or false, "reason": "short"}. '
    "Set grounded to false if the answer states facts the context does not support."
)


def _parse_json(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        snippet = text[text.index("{") : text.rindex("}") + 1]
        return json.loads(snippet)
    except Exception:
        return default


def retrieve(state):
    query = state.get("query") or state["question"]
    docs = get_final_retriever().invoke(query)
    attempts = state.get("attempts", 0) + 1
    step = {"node": "retrieve", "query": query, "documents": len(docs)}
    return {"documents": docs, "attempts": attempts, "steps": [step]}


def grade_documents(state):
    llm = _make_llm(state["model"], temperature=0)
    context = _format_context(state.get("documents") or [])[:4000]
    prompt = f"Question: {state['question']}\n\nContext:\n{context}"
    raw = llm.invoke(
        [SystemMessage(content=_GRADER_SYSTEM), HumanMessage(content=prompt)]
    ).content
    data = _parse_json(
        raw, {"relevant": bool(state.get("documents")), "confidence": 0.5}
    )
    confidence = float(data.get("confidence", 0.5))
    relevant = (
        bool(data.get("relevant")) and confidence >= settings.agent_confidence_threshold
    )
    step = {
        "node": "grade",
        "confidence": round(confidence, 2),
        "relevant": relevant,
        "reason": data.get("reason", ""),
    }
    return {
        "confidence": confidence,
        "grade": "relevant" if relevant else "weak",
        "steps": [step],
    }


def rewrite_query(state):
    llm = _make_llm(state["model"], temperature=0)
    new_query = llm.invoke(
        [
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=state["question"]),
        ]
    ).content.strip()
    step = {"node": "rewrite", "query": new_query}
    return {"query": new_query, "steps": [step]}


def web_search(state):
    docs = web_search_docs(state["question"])
    merged = (state.get("documents") or []) + docs
    step = {"node": "web_search", "documents": len(docs)}
    return {"documents": merged, "used_web": True, "steps": [step]}


def generate(state):
    llm = _make_llm(state["model"], temperature=0)
    context = _format_context(state.get("documents") or [])
    system = _ANSWER_SYSTEM.format(context=context)
    answer = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=state["question"])]
    ).content
    generations = state.get("generations", 0) + 1
    step = {"node": "generate", "generations": generations}
    return {"answer": answer, "generations": generations, "steps": [step]}


def self_check(state):
    llm = _make_llm(state["model"], temperature=0)
    context = _format_context(state.get("documents") or [])[:4000]
    prompt = f"Answer: {state.get('answer', '')}\n\nContext:\n{context}"
    raw = llm.invoke(
        [SystemMessage(content=_CHECK_SYSTEM), HumanMessage(content=prompt)]
    ).content
    data = _parse_json(raw, {"grounded": True})
    grounded = bool(data.get("grounded", True))
    step = {
        "node": "self_check",
        "grounded": grounded,
        "reason": data.get("reason", ""),
    }
    return {"grounded": grounded, "steps": [step]}

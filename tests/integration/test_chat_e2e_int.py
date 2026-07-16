"""End to end RAG test through Ollama: embeddings, pgvector retrieval, generation.

This proves the full chat pipeline works for real, with local models, no paid
key. It is skipped automatically when Ollama is not running.
"""

import urllib.request

import psycopg
import pytest
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

import src.core.config as config_mod
import src.embeddings.vectorstore_utils as vs

FILE_ID = 987654
FACTS = [
    "The capital of France is Paris.",
    "rag-agentic-2025 uses pgvector on Postgres for hybrid retrieval.",
    "Redis is the Celery broker used for background indexing.",
]


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _cleanup():
    with psycopg.connect(config_mod.settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM langchain_pg_embedding e "
                "USING langchain_pg_collection c "
                "WHERE e.collection_id = c.uuid AND c.name = %s "
                "AND e.cmetadata->>'file_id' = %s",
                (vs.COLLECTION_NAME, str(FILE_ID)),
            )


@pytest.mark.skipif(not _ollama_running(), reason="ollama server not running")
def test_end_to_end_rag_answer_is_grounded(pg_available, monkeypatch):
    monkeypatch.setattr(config_mod.settings, "embedding_provider", "ollama")
    monkeypatch.setattr(config_mod.settings, "use_reranker", False)
    monkeypatch.setattr(config_mod.settings, "top_k", 3)
    monkeypatch.setattr(vs, "_document_embeddings", None)
    monkeypatch.setattr(vs, "_query_embeddings", None)
    monkeypatch.setattr(vs, "_store", None)

    embeddings = OllamaEmbeddings(
        model=config_mod.settings.ollama_embedding_model,
        base_url=config_mod.settings.ollama_base_url,
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=vs.COLLECTION_NAME,
        connection=vs._sqlalchemy_url(),
        use_jsonb=True,
    )
    _cleanup()
    store.add_documents(
        [Document(page_content=f, metadata={"file_id": FILE_ID}) for f in FACTS]
    )

    try:
        from src.agent.graph import run_agent

        result = run_agent("llama3.2:3b", "What is the capital of France?")
    finally:
        _cleanup()

    assert result["context"], "expected retrieved context"
    assert any(s["node"] == "grade" for s in result["steps"])
    assert "paris" in result["answer"].lower()

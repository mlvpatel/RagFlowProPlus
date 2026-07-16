"""
Hybrid retrieval for rag-agentic-2025.

Dense search (pgvector cosine distance) and sparse search (Postgres full
text) are fused with Reciprocal Rank Fusion inside a single SQL query, so
there is no per query index rebuild in Python. The rag-advanced-2023 rung
rebuilt a BM25 index over the whole corpus on every query; this replaces that.

A cross encoder reranker (bge-reranker-v2-m3) then reorders the fused
candidates. The reranker is loaded lazily and can be injected, so tests
run without downloading the model or importing torch.
"""

import logging
from typing import Any, List

import psycopg
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from src.core.config import settings

logger = logging.getLogger(__name__)

_HYBRID_SQL = """
WITH q AS (
    SELECT c.uuid AS cid
    FROM langchain_pg_collection c
    WHERE c.name = %(collection)s
),
dense AS (
    SELECT e.id, e.document, e.cmetadata,
           row_number() OVER (ORDER BY e.embedding <=> %(qvec)s::vector) AS rank_dense
    FROM langchain_pg_embedding e, q
    WHERE e.collection_id = q.cid
    ORDER BY e.embedding <=> %(qvec)s::vector
    LIMIT %(pool)s
),
sparse AS (
    SELECT e.id, e.document, e.cmetadata,
           row_number() OVER (
               ORDER BY ts_rank(to_tsvector('english', e.document),
                                plainto_tsquery('english', %(query)s)) DESC
           ) AS rank_sparse
    FROM langchain_pg_embedding e, q
    WHERE e.collection_id = q.cid
      AND to_tsvector('english', e.document) @@ plainto_tsquery('english', %(query)s)
    LIMIT %(pool)s
)
SELECT
    coalesce(d.id, s.id)               AS id,
    coalesce(d.document, s.document)   AS document,
    coalesce(d.cmetadata, s.cmetadata) AS cmetadata,
    coalesce(1.0 / (%(rrf_k)s + d.rank_dense), 0)
      + coalesce(1.0 / (%(rrf_k)s + s.rank_sparse), 0) AS score
FROM dense d
FULL OUTER JOIN sparse s ON d.id = s.id
ORDER BY score DESC
LIMIT %(k)s
"""


def _vector_literal(vec: List[float]) -> str:
    """Format an embedding as the pgvector text input form, [a,b,c]."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


class HybridRetriever(BaseRetriever):
    """Fuse dense and sparse retrieval with RRF, computed in one SQL query."""

    embeddings: Any
    connection: str = settings.database_url
    collection: str = "rag_agentic_documents"
    k: int = settings.top_k
    pool: int = 20
    rrf_k: int = 60

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        qvec = _vector_literal(self.embeddings.embed_query(query))
        with psycopg.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    _HYBRID_SQL,
                    {
                        "collection": self.collection,
                        "qvec": qvec,
                        "query": query,
                        "pool": self.pool,
                        "rrf_k": self.rrf_k,
                        "k": self.k,
                    },
                )
                rows = cur.fetchall()

        docs: List[Document] = []
        for _id, document, cmetadata, score in rows:
            meta = dict(cmetadata) if cmetadata else {}
            meta["score"] = float(score)
            docs.append(Document(page_content=document, metadata=meta))
        return docs


_reranker = None


def get_reranker():
    """Lazily load the bge-reranker-v2-m3 cross encoder (downloads on first use)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s (first call only)", settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


class ReRankingRetriever(BaseRetriever):
    """Rerank a base retriever's candidates with a cross encoder, keep top_n."""

    base_retriever: Any
    top_n: int = settings.reranker_top_n
    cross_encoder: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return docs
        encoder = self.cross_encoder or get_reranker()
        scores = encoder.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]

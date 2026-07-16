"""Integration test: the hybrid SQL retrieval against live Postgres and pgvector.

A deterministic fake embedder keeps dense scores reproducible without an API
key. The sparse full text side is what should surface the matching document,
which proves the dense plus sparse RRF fusion is working end to end.
"""

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_postgres import PGVector

from src.core.config import settings
from src.embeddings.vectorstore_utils import _sqlalchemy_url
from src.retrieval.retrievers import HybridRetriever

COLLECTION = "rag_agentic_test_hybrid"


def _cleanup():
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM langchain_pg_embedding e "
                "USING langchain_pg_collection c "
                "WHERE e.collection_id = c.uuid AND c.name = %s",
                (COLLECTION,),
            )


def test_hybrid_retrieval_surfaces_the_matching_document(pg_available):
    fake = DeterministicFakeEmbedding(size=settings.embedding_dims)
    store = PGVector(
        embeddings=fake,
        collection_name=COLLECTION,
        connection=_sqlalchemy_url(),
        use_jsonb=True,
    )
    _cleanup()
    store.add_documents(
        [
            Document(page_content="the cat sat on the mat", metadata={"file_id": 1}),
            Document(
                page_content="quantum computing uses qubits", metadata={"file_id": 2}
            ),
            Document(
                page_content="the dog ran across the park", metadata={"file_id": 3}
            ),
        ]
    )

    retriever = HybridRetriever(embeddings=fake, collection=COLLECTION, k=3)
    out = retriever.invoke("cat mat")

    assert out, "expected at least one result"
    assert (
        "cat" in out[0].page_content
    ), "the sparse match should rank the cat document first"

    _cleanup()

"""
Vector store and embedding utilities for rag-agentic-2025.

Google gemini-embedding-001 embeddings stored in pgvector on Postgres.
This module keeps the public interface of the chroma_utils layer used by the earlier
rungs (load_and_split_document, index_document, delete_doc) while
swapping ChromaDB for pgvector, so the rest of the app keeps working.

Everything heavy is lazy. Importing this module opens no database
connection and loads no embedding client, which keeps unit tests fast
and lets processes import it without credentials.
"""

import datetime
import logging
import os
from typing import List

import psycopg
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_agentic_documents"

_text_splitter = None
_document_embeddings = None
_query_embeddings = None
_store = None


def _sqlalchemy_url() -> str:
    """PGVector runs on a SQLAlchemy engine, which needs the psycopg driver name."""
    url = settings.database_url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    return _text_splitter


def _build_embeddings(task_type: str):
    """Construct an embedder for the given task type.

    Provider is chosen by settings.embedding_provider. Google
    gemini-embedding-001 is the production default. Ollama (a local, no cost
    embedder such as nomic-embed-text) is available for offline development
    and verification; it ignores the task type.
    """
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        task_type=task_type,
    )


def get_document_embeddings():
    """Embedder used when indexing documents (retrieval_document task type)."""
    global _document_embeddings
    if _document_embeddings is None:
        _document_embeddings = _build_embeddings("retrieval_document")
    return _document_embeddings


def get_query_embeddings():
    """Embedder used when embedding a user query (retrieval_query task type)."""
    global _query_embeddings
    if _query_embeddings is None:
        _query_embeddings = _build_embeddings("retrieval_query")
    return _query_embeddings


def get_store():
    """Return the lazily constructed PGVector store, backed by document embeddings."""
    global _store
    if _store is None:
        from langchain_postgres import PGVector

        _store = PGVector(
            embeddings=get_document_embeddings(),
            collection_name=COLLECTION_NAME,
            connection=_sqlalchemy_url(),
            use_jsonb=True,
        )
    return _store


def load_and_split_document(file_path: str) -> List[Document]:
    """Load a document by file type and split it into chunks."""
    if file_path.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".html"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader

        loader = UnstructuredHTMLLoader(file_path)
    elif file_path.endswith((".txt", ".md")):
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return _get_text_splitter().split_documents(documents)


def index_document(file_path: str, file_id: int) -> bool:
    """Embed and store a document's chunks in pgvector, tagged with file_id."""
    try:
        splits = load_and_split_document(file_path)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for split in splits:
            split.metadata["file_id"] = file_id
            split.metadata["filename"] = os.path.basename(file_path)
            split.metadata["indexed_at"] = timestamp
        get_store().add_documents(splits)
        logger.info(f"Indexed {len(splits)} chunks for file_id={file_id}")
        return True
    except Exception as e:
        logger.error(f"Error indexing document {file_path}: {e}")
        return False


def delete_doc(file_id: int) -> bool:
    """Delete all chunks for a file_id from pgvector.

    langchain_postgres does not expose a metadata filtered delete, so this
    runs a direct SQL delete against the collection's embedding rows,
    matching the file_id stored in the chunk metadata (cmetadata jsonb).
    """
    try:
        with psycopg.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding e
                    USING langchain_pg_collection c
                    WHERE e.collection_id = c.uuid
                      AND c.name = %s
                      AND e.cmetadata->>'file_id' = %s
                    """,
                    (COLLECTION_NAME, str(file_id)),
                )
        logger.info(f"Deleted chunks for file_id={file_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file_id {file_id}: {e}")
        return False

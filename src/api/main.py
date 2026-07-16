"""rag-agentic-2025 API: agentic RAG chat plus document management.

The chat endpoint runs the self-correcting agent and returns the answer along
with its reasoning trace, so callers and the UI can see the retrieve, grade,
rewrite, and self-check steps the agent took.
"""

import os
import shutil
import uuid
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import (
    APIRouter,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.agent.graph import run_agent
from src.api.db_utils import (
    delete_document_record,
    get_all_documents,
    get_chat_history,
    init_db,
    insert_application_logs,
)
from src.api.pydantic_models import (
    DeleteFileRequest,
    DocumentInfo,
    QueryInput,
    QueryResponse,
)
from src.api.security import limiter, sanitize_question
from src.core.config import settings
from src.core.langchain_utils import warm_reranker
from src.core.logging_config import configure_logging, logger
from src.embeddings.vectorstore_utils import delete_doc
from src.worker.celery_app import celery_app
from src.worker.tasks import process_document

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".html", ".txt", ".md"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        init_db()
    except Exception as exc:
        logger.warning("init_db skipped at startup: %s", exc)
    warm_reranker()
    yield


app = FastAPI(title="rag-agentic-2025 API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

v1 = APIRouter(prefix="/v1")


@app.get("/health")
def health():
    return {"status": "ok"}


@v1.post("/chat", response_model=QueryResponse)
@limiter.limit("60/minute")
def chat(request: Request, query: QueryInput):
    """Answer with the agentic RAG and return the answer plus its reasoning trace."""
    session_id = query.session_id or str(uuid.uuid4())
    question = sanitize_question(query.question)
    history = get_chat_history(session_id)
    result = run_agent(query.model, question, history)
    answer = result["answer"]
    try:
        insert_application_logs(session_id, question, answer, query.model)
    except Exception as exc:
        logger.error("Failed to persist chat turn: %s", exc)
    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=query.model,
        steps=result["steps"],
        confidence=result.get("confidence"),
        used_web=result.get("used_web", False),
    )


@v1.post("/upload-doc")
@limiter.limit("10/minute")
async def upload_doc(request: Request, file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    upload_dir = os.getenv("UPLOAD_DIR", "data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, os.path.basename(file.filename))
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    task = process_document.delay(path, file.filename)
    return {"task_id": task.id, "status": "processing", "filename": file.filename}


@v1.get("/list-docs", response_model=list[DocumentInfo])
def list_docs():
    return get_all_documents()


@v1.post("/delete-doc")
def delete_document(req: DeleteFileRequest):
    vector_ok = delete_doc(req.file_id)
    record_ok = delete_document_record(req.file_id)
    if not (vector_ok and record_ok):
        raise HTTPException(status_code=500, detail="Delete failed")
    return {"status": "deleted", "file_id": req.file_id}


@v1.get("/task/{task_id}")
def task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": result.status}


app.include_router(v1)

"""
api.py — FastAPI REST endpoint for the Jenkins RAG assistant.

Usage:
    uvicorn api:app --reload
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_core import load_pipeline, query_rag


# ── Pipeline (loaded once at startup) ────────────────────────────────────────

pipeline = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once when the server starts."""
    vectorstore, chunks, bm25 = load_pipeline()
    pipeline["vectorstore"] = vectorstore
    pipeline["chunks"] = chunks
    pipeline["bm25"] = bm25
    yield
    pipeline.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Jenkins RAG API",
    description="Ask questions about Jenkins documentation. Answers are grounded in indexed sources only.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    fallback: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check that the API and pipeline are ready."""
    ready = bool(pipeline.get("vectorstore"))
    return {"status": "ok" if ready else "loading", "pipeline_ready": ready}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """Ask a question about Jenkins documentation."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    answer, sources = query_rag(
        question,
        vectorstore=pipeline.get("vectorstore"),
        chunks=pipeline.get("chunks"),
        bm25=pipeline.get("bm25"),
    )

    return AnswerResponse(
        question=question,
        answer=answer,
        sources=sources,
        fallback=not bool(sources),
    )

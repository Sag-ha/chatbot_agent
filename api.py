# =============================================================================
# src/api.py — FastAPI web application
#
# This is the entry point for the web server.
# It exposes two endpoints:
#   GET  /health   — Quick check that the server is up (used by Cloud Run)
#   POST /ask      — Submit an insurance question, get an answer
#
# Lifespan management:
#   The database connection and RAG chain are expensive to create.
#   We create them ONCE when the server starts (not on every request).
#   This is called the "application lifespan" pattern.
# =============================================================================

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from .rag_chain import AgentResponse, ask_insurance_question, build_rag_chain
from .vector_store import get_cloud_sql_engine, get_vector_store, init_vector_table

# ── Logging setup ──────────────────────────────────────────────────────────────
# Cloud Run captures stdout/stderr and shows it in Cloud Logging
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── App state (shared across requests) ────────────────────────────────────────
# We store the chain and retriever here so we don't recreate them per request.
class AppState:
    rag_chain = None
    retriever = None
    is_ready = False


app_state = AppState()


# ── Application lifespan ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE at startup and ONCE at shutdown.
    We use it to connect to the database and build the RAG chain.
    The 'yield' separates startup (above) from shutdown (below).
    """
    logger.info("=== Insurance Agent starting up ===")

    try:
        # 1. Connect to Cloud SQL
        engine = await get_cloud_sql_engine()

        # 2. Make sure the vector table exists (safe to call on every startup)
        await init_vector_table(engine)

        # 3. Load the vector store (points to the Cloud SQL table)
        vector_store = await get_vector_store(engine)

        # 4. Build the Gemini-powered RAG chain
        app_state.rag_chain, app_state.retriever = build_rag_chain(vector_store)
        app_state.is_ready = True

        logger.info("=== Insurance Agent is READY to serve requests ===")

    except Exception as e:
        logger.error(f"FATAL: Failed to initialise the app: {e}", exc_info=True)
        # App will start but /health will return 503, alerting Cloud Run
        app_state.is_ready = False

    yield  # ← Server is running here, handling requests

    # Shutdown cleanup (runs when Cloud Run stops the container)
    logger.info("=== Insurance Agent shutting down ===")


# ── FastAPI app instance ───────────────────────────────────────────────────────
app = FastAPI(
    title="Virtual Insurance Agent",
    description="AI-powered insurance Q&A backed by Gemini 2.5 Flash + Cloud SQL",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend apps (or Postman) to call this API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, replace with your frontend domain
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    """What the user sends in the request body."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The insurance question to ask",
        example="What is the difference between term and whole life insurance?",
    )


class QuestionResponse(BaseModel):
    """What the API sends back."""
    answer: str
    source_documents: list[dict]


class HealthResponse(BaseModel):
    status: str
    model: str
    ready: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Cloud Run pings this to decide if the container is alive.
    Returns 200 if ready, 503 if initialisation failed.
    """
    if not app_state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready yet. Check startup logs.",
        )
    return HealthResponse(
        status="healthy",
        model=settings.GEMINI_MODEL,
        ready=True,
    )


@app.post("/ask", response_model=QuestionResponse, tags=["Insurance Agent"])
async def ask_question(request: QuestionRequest):
    """
    Ask the insurance agent a question.

    The agent will:
    1. Search the knowledge base for relevant document chunks.
    2. Feed those chunks + your question to Gemini 2.5 Flash.
    3. Return a grounded answer with source citations.
    """
    # Guard: don't serve requests if startup failed
    if not app_state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Try again in a moment.",
        )

    try:
        response: AgentResponse = await ask_insurance_question(
            question=request.question,
            chain=app_state.rag_chain,
            retriever=app_state.retriever,
        )
        return QuestionResponse(
            answer=response.answer,
            source_documents=response.source_documents,
        )

    except Exception as e:
        logger.error(f"Error handling question '{request.question}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )


# ── Local development entry point ──────────────────────────────────────────────
# DO NOT run this file directly.
# Instead, run from the project ROOT directory:
#
#   uvicorn src.api:app --reload --port 8080
#
# Or simply use the provided main.py:
#
#   python main.py

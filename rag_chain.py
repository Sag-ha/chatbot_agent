# =============================================================================
# src/rag_chain.py — The core RAG (Retrieval-Augmented Generation) logic
#
# How RAG works in 3 steps:
#   1. RETRIEVE  — User asks a question → search vector DB for relevant chunks
#   2. AUGMENT   — Combine retrieved chunks with the question into a prompt
#   3. GENERATE  — Send the prompt to Gemini → get a grounded answer
#
# This ensures Gemini answers ONLY from your insurance documents, not from
# general internet knowledge (which could be wrong or outdated).
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Any, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_google_cloud_sql_pg import CloudSQLVectorStore
from langchain_google_vertexai import ChatVertexAI

from config import settings

logger = logging.getLogger(__name__)


# ── Prompt Template ────────────────────────────────────────────────────────────
# This is the exact instruction we send to Gemini every time a user asks a question.
# {context} will be filled with retrieved document chunks.
# {question} will be filled with the user's question.

INSURANCE_AGENT_PROMPT = ChatPromptTemplate.from_template(
    """You are a professional and helpful insurance agent assistant.
Your job is to answer insurance-related questions using ONLY the provided context below.

CONTEXT (extracted from official insurance documents):
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer clearly and concisely (under 150 words).
- Base your answer ONLY on the context provided above.
- If the context does not contain enough information, say:
  "I don't have enough information about this topic in my knowledge base. 
   Please consult a licensed insurance advisor."
- Do NOT make up facts, policies, or numbers.
- Use simple language — avoid overly technical jargon.
"""
)


# ── Response dataclass ─────────────────────────────────────────────────────────
@dataclass
class AgentResponse:
    """Structured response returned by the RAG chain."""
    answer: str                  # The final answer text from Gemini
    source_documents: list[dict] # Which document chunks were used (for transparency)


# ── LLM factory ───────────────────────────────────────────────────────────────
def get_llm() -> ChatVertexAI:
    """
    Return a configured Gemini 2.5 Flash model via Vertex AI.

    Why Gemini 2.5 Flash?
      - Much faster and cheaper than Gemini Pro
      - Still very capable for Q&A tasks
      - Supports long context windows (good for multiple retrieved chunks)
    """
    return ChatVertexAI(
        model=settings.GEMINI_MODEL,            # "gemini-2.5-flash"
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_REGION,
        temperature=settings.TEMPERATURE,       # 0.1 = factual, not creative
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    )


# ── Chain builder ─────────────────────────────────────────────────────────────
def build_rag_chain(
    vector_store: CloudSQLVectorStore,
) -> Tuple[RunnableSerializable, Any]:
    """
    Build the full RAG chain using LangChain Expression Language (LCEL).

    LCEL uses the pipe operator (|) to chain steps together, like a pipeline:
        retriever | format_docs | prompt | llm | output_parser

    Returns:
        A tuple of (chain, retriever) so we can run the chain AND get source docs.
    """

    # Step 1: Retriever — finds the most relevant document chunks.
    # We use plain "similarity" search (top-k) because it is guaranteed to be
    # supported by CloudSQLVectorStore.  The SIMILARITY_THRESHOLD filter is applied
    # manually in ask_insurance_question() below.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.RETRIEVER_K,   # Return top-k chunks (default 4)
        },
    )

    # Step 2: Format retrieved chunks into a single string for the prompt
    def format_docs(docs) -> str:
        """Join all retrieved chunks with separators so Gemini can read them."""
        if not docs:
            return "No relevant information found in the knowledge base."
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Step 3: LLM
    llm = get_llm()

    # Step 4: Chain everything together using LCEL
    #   {"context": ..., "question": ...}  →  prompt  →  llm  →  plain text
    chain = (
        {
            "context": retriever | format_docs,   # Retrieve docs, then format
            "question": RunnablePassthrough(),    # Pass question through unchanged
        }
        | INSURANCE_AGENT_PROMPT
        | llm
        | StrOutputParser()  # Converts LLM message object → plain string
    )

    logger.info("RAG chain built successfully.")
    return chain, retriever


# ── Main query function ────────────────────────────────────────────────────────
async def ask_insurance_question(
    question: str,
    chain: RunnableSerializable,
    retriever,
) -> AgentResponse:
    """
    Ask the insurance agent a question and get a grounded answer.

    Args:
        question:  The user's insurance question (plain text).
        chain:     The RAG chain returned by build_rag_chain().
        retriever: The retriever (needed to fetch source docs separately).

    Returns:
        AgentResponse with the answer and which documents were used.
    """
    logger.info(f"Processing question: '{question[:80]}...' " if len(question) > 80 else f"Processing question: '{question}'")

    # Run the RAG chain to get the answer
    answer = await chain.ainvoke(question)

    # Also fetch source documents so we can show the user where the answer came from
    source_docs = await retriever.ainvoke(question)
    sources = [
        {
            "page_content_preview": doc.page_content[:200] + "…",
            "source_file": doc.metadata.get("source", "unknown"),
            "page_number": doc.metadata.get("page", "unknown"),
            "gcs_path": doc.metadata.get("gcs_path", ""),
        }
        for doc in source_docs
    ]

    logger.info(f"Answer generated. Used {len(sources)} source chunk(s).")

    return AgentResponse(answer=answer, source_documents=sources)

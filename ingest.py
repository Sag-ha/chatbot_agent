# =============================================================================
# scripts/ingest.py — One-time document ingestion script
#
# PURPOSE:
#   This script loads your insurance PDF documents into the vector database.
#   You only need to run it:
#     - Once when you first set up the project
#     - Whenever you add new PDF documents to the GCS bucket
#
# USAGE (from the project root directory):
#   python scripts/ingest.py
#
# WHAT IT DOES:
#   1. Downloads all PDFs from your GCS bucket
#   2. Splits them into small text chunks
#   3. Converts each chunk to a numeric embedding using Vertex AI
#   4. Saves all embeddings to Cloud SQL (PostgreSQL + pgvector)
#
# After this script completes successfully, your API can answer questions.
# =============================================================================

import asyncio
import logging
import sys
import os

# Add the project root to Python's path so we can import config and src modules.
# This is needed because this script lives inside the /scripts subfolder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import settings
from src.document_loader import download_and_load_pdfs, split_into_chunks
from src.vector_store import (
    add_documents_to_store,
    get_cloud_sql_engine,
    get_vector_store,
    init_vector_table,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Main ingestion logic ───────────────────────────────────────────────────────
async def ingest() -> None:
    """
    Full ingestion pipeline: GCS → chunks → embeddings → Cloud SQL.
    This is an async function because database operations use async I/O.
    """
    logger.info("=" * 60)
    logger.info("  INSURANCE AGENT — Document Ingestion")
    logger.info("=" * 60)
    logger.info(f"  GCS Bucket   : gs://{settings.GCS_BUCKET_NAME}/{settings.GCS_DOCS_PREFIX}")
    logger.info(f"  Cloud SQL    : {settings.CLOUD_SQL_INSTANCE}")
    logger.info(f"  Table        : {settings.VECTOR_TABLE_NAME}")
    logger.info(f"  Embedding    : {settings.EMBEDDING_MODEL}")
    logger.info("=" * 60)

    # ── Step 1: Download and read all PDFs from GCS ────────────────────────────
    logger.info("\n[1/4] Loading PDF documents from GCS …")
    raw_documents = download_and_load_pdfs()
    logger.info(f"      Loaded {len(raw_documents)} pages total.")

    # ── Step 2: Split pages into small overlapping chunks ─────────────────────
    logger.info("\n[2/4] Splitting pages into text chunks …")
    chunks = split_into_chunks(raw_documents)
    logger.info(f"      Created {len(chunks)} chunks "
                f"(chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}).")

    # ── Step 3: Connect to Cloud SQL and prepare the vector table ─────────────
    logger.info("\n[3/4] Connecting to Cloud SQL and preparing vector table …")
    engine = await get_cloud_sql_engine()
    await init_vector_table(engine)      # Creates table if it doesn't exist
    vector_store = await get_vector_store(engine)
    logger.info("      Vector table is ready.")

    # ── Step 4: Embed all chunks and insert into the database ─────────────────
    logger.info("\n[4/4] Embedding chunks and inserting into Cloud SQL …")
    logger.info("      (This may take several minutes for large document sets)")
    await add_documents_to_store(vector_store, chunks)

    # ── Done ───────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  INGESTION COMPLETE!")
    logger.info(f"  {len(chunks)} chunks are now stored in Cloud SQL.")
    logger.info("  You can now start the API server with:  python main.py")
    logger.info("=" * 60)


# ── Script entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(ingest())
    except KeyboardInterrupt:
        logger.info("\nIngestion cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nIngestion FAILED: {e}", exc_info=True)
        logger.error("\nCommon causes:")
        logger.error("  - .env file missing or has wrong values")
        logger.error("  - GCS bucket or PDF folder path is wrong")
        logger.error("  - Cloud SQL instance name is wrong")
        logger.error("  - You are not authenticated: run 'gcloud auth application-default login'")
        sys.exit(1)

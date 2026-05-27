# =============================================================================
# src/vector_store.py — Manage the vector database on Cloud SQL (PostgreSQL)
#
# What is a vector store?
#   Each text chunk from the PDFs is converted to a list of ~768 numbers
#   (called an "embedding"). The vector store saves these numbers and lets us
#   search for chunks that are *semantically* close to a user's question.
#
# We use:
#   - Cloud SQL (PostgreSQL) as the database — a managed DB on GCP
#   - pgvector extension — adds vector search to PostgreSQL
#   - langchain-google-cloud-sql-pg — official Google package that handles
#     the secure Cloud SQL connection for us (no manual IP whitelisting needed)
# =============================================================================

import logging

from langchain_core.documents import Document
from langchain_google_cloud_sql_pg import CloudSQLEngine, CloudSQLVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

from config import settings

logger = logging.getLogger(__name__)


def get_embedding_model() -> VertexAIEmbeddings:
    """
    Return the Vertex AI embedding model.

    text-embedding-004 converts a sentence → 768 numbers that capture meaning.
    Similar sentences get similar numbers, which makes similarity search possible.
    """
    return VertexAIEmbeddings(
        model_name=settings.EMBEDDING_MODEL,    # "text-embedding-004"
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_REGION,
    )


async def get_cloud_sql_engine() -> CloudSQLEngine:
    """
    Create a Cloud SQL connection engine.

    CloudSQLEngine handles:
      - IAM authentication (no hardcoded passwords needed when on GCP)
      - Connection pooling (reuses DB connections for efficiency)
      - Automatic SSL (secure connection)

    The 'instance' parameter format is: "project-id:region:instance-name"
    """
    logger.info(f"Connecting to Cloud SQL instance: {settings.CLOUD_SQL_INSTANCE}")

    # Parse the instance connection name: "project:region:instance"
    parts = settings.CLOUD_SQL_INSTANCE.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"CLOUD_SQL_INSTANCE must be in format 'project:region:instance', "
            f"got: '{settings.CLOUD_SQL_INSTANCE}'"
        )
    project_id, region, instance_name = parts

    engine = await CloudSQLEngine.afrom_instance(
        project_id=project_id,
        region=region,
        instance=instance_name,
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASS,
    )

    logger.info("Cloud SQL engine created successfully.")
    return engine


async def init_vector_table(engine: CloudSQLEngine) -> None:
    """
    Create the vector table in PostgreSQL (if it doesn't already exist).

    This is idempotent — safe to call every time the app starts.
    It enables the pgvector extension and creates the table with the
    right columns: id, content, embedding (vector of 768 floats), metadata.
    """
    logger.info(f"Initialising vector table '{settings.VECTOR_TABLE_NAME}' …")

    await engine.ainit_vectorstore_table(
        table_name=settings.VECTOR_TABLE_NAME,
        vector_size=settings.EMBEDDING_DIMENSIONS,  # 768 for text-embedding-004
        overwrite_existing=False,  # Don't wipe existing data on restart!
    )

    logger.info("Vector table ready.")


async def get_vector_store(engine: CloudSQLEngine) -> CloudSQLVectorStore:
    """
    Return a LangChain vector store backed by Cloud SQL.

    This object lets us:
      - .aadd_documents(chunks)  →  embed chunks and save them to DB
      - .as_retriever(...)       →  search for relevant chunks at query time
    """
    embeddings = get_embedding_model()

    vector_store = await CloudSQLVectorStore.create(
        engine=engine,
        table_name=settings.VECTOR_TABLE_NAME,
        embedding_service=embeddings,
        # Metadata (source file, page number, gcs_path) is stored automatically
        # as a JSON blob in the "langchain_metadata" column — no extra config needed.
    )

    return vector_store


async def add_documents_to_store(
    vector_store: CloudSQLVectorStore,
    chunks: list[Document],
    batch_size: int = 50,
) -> None:
    """
    Embed all chunks and save them to the vector database.

    We process in small batches to:
      - Avoid hitting the Vertex AI rate limit
      - Show progress for large document sets

    Args:
        vector_store: The CloudSQLVectorStore instance.
        chunks:       List of text chunks from document_loader.split_into_chunks().
        batch_size:   How many chunks to embed and insert per API call.
    """
    total = len(chunks)
    logger.info(f"Adding {total} chunks to vector store in batches of {batch_size} …")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        await vector_store.aadd_documents(batch)
        logger.info(f"  Inserted batch {i // batch_size + 1} / {-(-total // batch_size)}")

    logger.info("All chunks inserted into vector store.")

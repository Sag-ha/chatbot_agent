# =============================================================================
# config.py — Central place for all project settings
#
# All sensitive values (passwords, project IDs) are read from a .env file.
# We use pydantic-settings so that missing variables raise a clear error
# instead of crashing deep inside the code.
# =============================================================================

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Every setting the app needs, with types and defaults.
    Values are loaded from the .env file (or real environment variables).
    """

    # ------------------------------------------------------------------
    # GCP Project
    # ------------------------------------------------------------------
    GCP_PROJECT_ID: str           # e.g. "my-insurance-project-123"
    GCP_REGION: str = "us-central1"  # Change if you want a different region

    # ------------------------------------------------------------------
    # Vertex AI  — LLM & Embeddings
    # We use Gemini 2.5 Flash (fast + cheap) for the chatbot.
    # text-embedding-004 is Google's best embedding model right now.
    # ------------------------------------------------------------------
    GEMINI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "text-embedding-004"

    # LLM generation controls
    TEMPERATURE: float = 0.1       # Low = more factual, less creative
    MAX_OUTPUT_TOKENS: int = 512   # Max words in one answer

    # ------------------------------------------------------------------
    # Cloud SQL (PostgreSQL + pgvector)  — Vector Database
    # Cloud SQL instance connection name format:
    #   "your-project:us-central1:your-instance-name"
    # ------------------------------------------------------------------
    CLOUD_SQL_INSTANCE: str        # e.g. "my-project:us-central1:insurance-db"
    DB_NAME: str = "insurance_db"
    DB_USER: str = "insurance_user"
    DB_PASS: str

    # Name of the table that will store embeddings inside PostgreSQL
    VECTOR_TABLE_NAME: str = "insurance_embeddings"

    # Dimensions for text-embedding-004 (768 is fixed by Google)
    EMBEDDING_DIMENSIONS: int = 768

    # ------------------------------------------------------------------
    # Cloud Storage  — Where the PDF insurance documents live
    # ------------------------------------------------------------------
    GCS_BUCKET_NAME: str          # e.g. "my-insurance-docs-bucket"
    GCS_DOCS_PREFIX: str = "insurance-docs/"  # Folder path inside the bucket

    # ------------------------------------------------------------------
    # RAG (Retrieval-Augmented Generation) tuning
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = 500          # Characters per text chunk
    CHUNK_OVERLAP: int = 100       # Overlap between chunks (preserves context)
    RETRIEVER_K: int = 4           # How many chunks to retrieve per query
    SIMILARITY_THRESHOLD: float = 0.5  # Min relevance score (0–1) to include a chunk

    # ------------------------------------------------------------------
    # API Server
    # ------------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080           # Cloud Run uses 8080 by default
    LOG_LEVEL: str = "info"

    class Config:
        # Reads values from a .env file in the project root
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single shared instance — import this everywhere instead of creating new ones
settings = Settings()

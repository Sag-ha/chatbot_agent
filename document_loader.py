# =============================================================================
# src/document_loader.py — Load PDF documents from Google Cloud Storage
#
# What this file does:
#   1. Connects to your GCS bucket
#   2. Downloads every PDF found inside the given folder (prefix)
#   3. Reads the text from each PDF page
#   4. Splits the text into small overlapping chunks
#      (small chunks = better search matches later)
# =============================================================================

import logging
import os
import tempfile

from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import settings

# Standard Python logger — messages will show up in Cloud Run logs
logger = logging.getLogger(__name__)


def download_and_load_pdfs() -> list[Document]:
    """
    Download every PDF from the GCS bucket and load all pages as Documents.

    Returns:
        A flat list of LangChain Document objects (one per PDF page).
        Each Document has:
          - page_content  : the raw text from that page
          - metadata      : {"source": "filename.pdf", "page": 0, ...}
    """
    logger.info(
        f"Connecting to GCS bucket '{settings.GCS_BUCKET_NAME}', "
        f"prefix '{settings.GCS_DOCS_PREFIX}'"
    )

    # Authenticate using Application Default Credentials (ADC).
    # On Cloud Run / GCE this works automatically.
    # Locally, run:  gcloud auth application-default login
    gcs_client = storage.Client(project=settings.GCP_PROJECT_ID)
    bucket = gcs_client.bucket(settings.GCS_BUCKET_NAME)

    # List all objects (files) inside our docs folder
    blobs = list(bucket.list_blobs(prefix=settings.GCS_DOCS_PREFIX))
    pdf_blobs = [b for b in blobs if b.name.lower().endswith(".pdf")]

    if not pdf_blobs:
        raise FileNotFoundError(
            f"No PDF files found in gs://{settings.GCS_BUCKET_NAME}/{settings.GCS_DOCS_PREFIX}"
        )

    logger.info(f"Found {len(pdf_blobs)} PDF file(s) in GCS.")
    all_documents: list[Document] = []

    # We use a temporary directory so downloaded files are cleaned up automatically
    with tempfile.TemporaryDirectory() as tmp_dir:
        for blob in pdf_blobs:
            # e.g. "insurance-docs/life_insurance_guide.pdf" → "life_insurance_guide.pdf"
            filename = os.path.basename(blob.name)
            local_path = os.path.join(tmp_dir, filename)

            logger.info(f"  Downloading: {blob.name}")
            blob.download_to_filename(local_path)

            # PyPDFLoader reads each page as a separate Document
            loader = PyPDFLoader(local_path)
            pages = loader.load()

            # Tag each page with the original GCS path so we can cite sources later
            for page in pages:
                page.metadata["gcs_path"] = f"gs://{settings.GCS_BUCKET_NAME}/{blob.name}"

            all_documents.extend(pages)
            logger.info(f"  Loaded {len(pages)} page(s) from '{filename}'")

    logger.info(f"Total pages loaded: {len(all_documents)}")
    return all_documents


def split_into_chunks(documents: list[Document]) -> list[Document]:
    """
    Split large pages into smaller overlapping text chunks.

    Why overlap?  If a sentence is cut in half at a chunk boundary, the
    overlap ensures the next chunk also contains the beginning of that
    sentence, so the context is never lost.

    Args:
        documents: List of full-page Documents from load_documents_from_gcs().

    Returns:
        List of smaller Document chunks ready to be embedded.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,       # ~500 characters per chunk
        chunk_overlap=settings.CHUNK_OVERLAP,  # 100-character overlap
        # Try splitting at paragraphs → sentences → words before hard-splitting
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        f"Split {len(documents)} pages → {len(chunks)} chunks "
        f"(chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})"
    )
    return chunks

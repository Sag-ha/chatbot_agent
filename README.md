# Virtual Insurance Agent — GCP Edition

An AI-powered insurance Q&A chatbot built with **Gemini 2.5 Flash**, **Cloud SQL (pgvector)**, and **Cloud Storage**. Ask questions about your insurance documents and get grounded, cited answers.

---

## Project Structure

```
insurance-agent/
├── config.py               ← All settings (reads from .env)
├── main.py                 ← Start the server: python main.py
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← Container for Cloud Run deployment
├── .env.example            ← Template — copy to .env and fill in values
├── scripts/
│   └── ingest.py           ← Run ONCE to load PDFs into the database
└── src/
    ├── __init__.py
    ├── api.py              ← FastAPI endpoints (/health, /ask)
    ├── document_loader.py  ← Downloads PDFs from GCS and splits into chunks
    ├── rag_chain.py        ← RAG pipeline: retrieve → augment → generate
    └── vector_store.py     ← Cloud SQL connection and vector operations
```

---

## GCP Services Used

| Service | Purpose |
|---|---|
| **Vertex AI — Gemini 2.5 Flash** | Answers questions (the "brain") |
| **Vertex AI — text-embedding-004** | Converts text to searchable numbers |
| **Cloud SQL (PostgreSQL + pgvector)** | Stores and searches document embeddings |
| **Cloud Storage (GCS)** | Stores the source PDF documents |
| **Cloud Run** *(optional)* | Hosts the API in production |

---

## Setup Guide

### Step 1 — Prerequisites

Make sure you have these installed:
```bash
python --version      # 3.11 or higher
gcloud --version      # Google Cloud CLI
```

Authenticate with GCP:
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Step 2 — GCP Resources

**Enable required APIs:**
```bash
gcloud services enable \
  sqladmin.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com
```

**Create a Cloud SQL instance** (PostgreSQL with pgvector):
```bash
gcloud sql instances create insurance-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database and user
gcloud sql databases create insurance_db --instance=insurance-db
gcloud sql users create insurance_user \
  --instance=insurance-db \
  --password=YOUR_SECURE_PASSWORD
```

**Create a GCS bucket and upload your PDFs:**
```bash
gsutil mb -l us-central1 gs://your-insurance-docs-bucket
gsutil cp your-pdfs/*.pdf gs://your-insurance-docs-bucket/insurance-docs/
```

### Step 3 — Local Setup

```bash
# Clone / navigate to project
cd insurance-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and fill in your real values
```

### Step 4 — Ingest Documents (Run Once)

This loads your PDFs into the vector database:
```bash
python scripts/ingest.py
```

You should see output like:
```
[1/4] Loading PDF documents from GCS …
[2/4] Splitting pages into text chunks …
[3/4] Connecting to Cloud SQL …
[4/4] Embedding and inserting into Cloud SQL …
INGESTION COMPLETE! 847 chunks stored.
```

### Step 5 — Start the API

```bash
python main.py
```

The server starts at `http://localhost:8080`

---

## API Usage

### Health Check
```bash
curl http://localhost:8080/health
```
```json
{"status": "healthy", "model": "gemini-2.5-flash", "ready": true}
```

### Ask a Question
```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between term and whole life insurance?"}'
```
```json
{
  "answer": "Term insurance provides coverage for a specific period (e.g., 10–30 years) and pays a death benefit only if the insured dies during that term...",
  "source_documents": [
    {
      "page_content_preview": "Term life insurance is temporary coverage...",
      "source_file": "life_insurance_guide.pdf",
      "page_number": 12,
      "gcs_path": "gs://your-bucket/insurance-docs/life_insurance_guide.pdf"
    }
  ]
}
```

### Interactive API Docs

Open in your browser: `http://localhost:8080/docs`

This gives you a visual interface to test all endpoints.

---

## Deploy to Cloud Run

```bash
gcloud run deploy insurance-agent \
  --source . \
  --region us-central1 \
  --set-env-vars GCP_PROJECT_ID=your-project \
  --set-env-vars CLOUD_SQL_INSTANCE=your-project:us-central1:insurance-db \
  --set-env-vars DB_NAME=insurance_db \
  --set-env-vars DB_USER=insurance_user \
  --set-env-vars DB_PASS=your-password \
  --set-env-vars GCS_BUCKET_NAME=your-bucket \
  --allow-unauthenticated
```

> **Security tip:** Use `--set-secrets` instead of `--set-env-vars` for passwords in production.

---

## Troubleshooting

**"No PDF files found in GCS"**
→ Check `GCS_BUCKET_NAME` and `GCS_DOCS_PREFIX` in your `.env` file.
→ Verify the bucket exists: `gsutil ls gs://your-bucket/insurance-docs/`

**"Failed to connect to Cloud SQL"**
→ Check `CLOUD_SQL_INSTANCE` format: must be `project:region:instance`
→ Ensure the Cloud SQL Admin API is enabled.

**"Permission denied"**
→ Run `gcloud auth application-default login` again.
→ Ensure your GCP account has roles: `Cloud SQL Client`, `Storage Object Viewer`, `Vertex AI User`.

**Empty answers / "I don't have enough information"**
→ Ingestion may not have run — execute `python scripts/ingest.py`.
→ Lower `SIMILARITY_THRESHOLD` in `.env` (try `0.3`).

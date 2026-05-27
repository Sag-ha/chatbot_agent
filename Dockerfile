# =============================================================================
# Dockerfile — Build a container image to deploy on Google Cloud Run
#
# Build the image locally:
#   docker build -t insurance-agent .
#
# Run locally (needs a .env file in the current directory):
#   docker run --env-file .env -p 8080:8080 insurance-agent
#
# Deploy to Cloud Run (GCP handles the build automatically):
#   gcloud run deploy insurance-agent \
#     --source . \
#     --region us-central1 \
#     --allow-unauthenticated
# =============================================================================

# ── Stage 1: Base image ────────────────────────────────────────────────────────
# We use the official "slim" Python image.
# "slim" means it has no unnecessary tools — smaller image = faster Cloud Run cold starts.
FROM python:3.11-slim

# ── Stage 2: System dependencies ──────────────────────────────────────────────
# We need gcc to compile some Python packages (e.g., asyncpg).
# We delete the apt cache afterwards to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 3: Set working directory ────────────────────────────────────────────
# All subsequent commands run inside /app inside the container.
WORKDIR /app

# ── Stage 4: Install Python dependencies ──────────────────────────────────────
# Copy requirements FIRST (before the rest of the code).
# Docker caches each layer — if requirements.txt hasn't changed, this layer
# is reused and pip install is skipped, making rebuilds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 5: Copy project code ────────────────────────────────────────────────
# Copy everything else after installing dependencies.
COPY config.py .
COPY main.py .
COPY src/ ./src/

# ── Stage 6: Runtime configuration ────────────────────────────────────────────
# Cloud Run injects environment variables from Secret Manager or the deployment config.
# Do NOT copy your .env file into the image — that would expose secrets!

# Cloud Run routes external traffic to port 8080.
EXPOSE 8080

# Create a non-root user for security (running as root is bad practice).
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# ── Stage 7: Start the server ─────────────────────────────────────────────────
# This command runs when the container starts.
# We use main.py which calls uvicorn internally.
CMD ["python", "main.py"]

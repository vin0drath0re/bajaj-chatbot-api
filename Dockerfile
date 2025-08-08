FROM python:3.11-slim

WORKDIR /app

# Install build tools + system libraries needed for chromadb & torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libsqlite3-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip & install Python deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Default port
ENV PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

FROM python:3.11-slim

WORKDIR /app

# Install build tools + libs for chromadb
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

COPY requirements.txt .

# Install CPU-only torch before everything else
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

# main.py

import os
import sys
import time
import logging
import hashlib
import asyncio
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from cachetools import TTLCache

import model  # LangChain logic

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("hackrx_logger")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

file_handler = RotatingFileHandler("logs/timings.log.txt", maxBytes=2_000_000, backupCount=3)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- FastAPI App ---
app = FastAPI()
API_KEY = os.getenv("API_KEY", "supersecretkey")

# Cache for QA chains (TTL: 1 hour, max 100 documents)
qa_cache = TTLCache(maxsize=100, ttl=3600)

class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

def get_cache_key(url: str) -> str:
    """Generate cache key from URL"""
    return hashlib.md5(url.encode()).hexdigest()

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(payload: HackRxRequest, authorization: Optional[str] = Header(None)):
    start_time = time.time()
    logger.info("New request to /hackrx/run")

    # Auth check
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split("Bearer ")[1]
    if token != API_KEY:
        logger.warning("Invalid API token")
        raise HTTPException(status_code=403, detail="Invalid API token")

    source_url = payload.documents
    cache_key = get_cache_key(source_url)
    
    # Try to get QA chain from cache
    qa_chain = qa_cache.get(cache_key)
    if qa_chain:
        logger.info("Using cached QA chain")
    else:
        try:
            qa_chain = await model.get_qa_chain_for_url_async(source_url)
            qa_cache[cache_key] = qa_chain
            logger.info("QA chain cached for future use")
        except Exception as e:
            logger.exception("Failed to prepare retriever")
            raise HTTPException(status_code=500, detail=str(e))

    # Process questions concurrently
    tasks = []
    for i, question in enumerate(payload.questions):
        logger.info(f"Queuing question {i+1}: {question}")
        tasks.append(model.ask_query_async(question, qa_chain=qa_chain))
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.exception(f"Error answering question {i+1}")
                answers.append("Error processing question")
            else:
                answers.append(result.get("result", "No answer"))
    except Exception as e:
        logger.exception("Error processing questions concurrently")
        raise HTTPException(status_code=500, detail="Error processing questions")

    logger.info(f"Total request completed in {time.time() - start_time:.2f} sec")
    return {"answers": answers}

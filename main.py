# main.py

import os
import sys
import time
import logging
import hashlib
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional

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

class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

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
    try:
        qa_chain = model.get_qa_chain_for_url(source_url)
    except Exception as e:
        logger.exception("Failed to prepare retriever")
        raise HTTPException(status_code=500, detail=str(e))

    answers = []
    for i, question in enumerate(payload.questions):
        q_start = time.time()
        logger.info(f"Processing question {i+1}: {question}")
        try:
            result = model.ask_query(question, qa_chain=qa_chain)
            answers.append(result.get("result", "No answer"))
            logger.info(f"Answered question {i+1} in {time.time() - q_start:.2f} sec")
        except Exception as e:
            logger.exception(f"Error answering question {i+1}")
            answers.append("Error processing question")

    logger.info(f"Total request completed in {time.time() - start_time:.2f} sec")
    return {"answers": answers}

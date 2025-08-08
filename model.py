# model.py

import os
import time
import asyncio
import aiohttp
import tempfile
import logging
from dotenv import load_dotenv
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Logger ---
logger = logging.getLogger("hackrx_logger")

# --- Load env vars ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY3")

# --- LLM & Prompt Setup ---
system_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant for an intelligent document reasoning system that handles 
natural language queries about insurance policies, contracts, and other official documents.

Your role is to:
- Parse vague or plain-English queries to extract structured details like age, gender, location, 
procedure, and policy duration.
- Retrieve relevant clauses from provided documents (PDFs, Word, emails) using semantic understanding, 
not just keyword matching.
- Evaluate the query using the retrieved clauses and return a factual response.

Guidelines:
- Only tell the details mentioned in the documents, don't add additional details that are not mentioned in the documents.
- If required info is missing, return `"decision": "needs_clarification"` and explain what’s needed.
- If no relevant clause is found, say so and show the closest matching content.
- Be explainable, traceable, and cautious. Don’t hallucinate.

Only handle insurance/policy/legal document queries. For unrelated questions, redirect via the general_chat tool.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.4
)

# --- Optimized embeddings (cached) ---
@lru_cache(maxsize=1)
def get_embeddings():
    """Cache embeddings model to avoid reloading"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster, smaller model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# --- HTTP Session for connection pooling ---
_http_session = None

async def get_http_session():
    global _http_session
    if _http_session is None:
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=60)
        _http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _http_session

# --- Async Utilities ---

async def load_pdf_from_url_async(pdf_url: str):
    start = time.time()
    session = await get_http_session()
    
    try:
        async with session.get(pdf_url) as response:
            response.raise_for_status()
            content = await response.read()
    except Exception as e:
        logger.exception("Failed to download PDF")
        raise ValueError(f"Error fetching PDF: {str(e)}")

    # Run PDF processing in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, _process_pdf_content, content)
    
    logger.info(f"PDF downloaded and loaded in {time.time() - start:.2f} sec")
    return docs

def _process_pdf_content(content: bytes):
    """Helper function to process PDF content in thread pool"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_file_path)
    
    return docs

async def retriever_from_docs_async(docs):
    start = time.time()
    embeddings = get_embeddings()
    
    # Optimized text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced chunk size for faster processing
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Run text splitting in thread pool
    loop = asyncio.get_event_loop()
    splits = await loop.run_in_executor(None, text_splitter.split_documents, docs)
    
    logger.info(f"Split into {len(splits)} chunks")

    # Run vector store creation in thread pool
    vectorstore = await loop.run_in_executor(
        None, 
        lambda: Chroma.from_documents(splits, embeddings)
    )
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}  # Reduced for faster retrieval
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        verbose=False
    )

    logger.info(f"Retriever initialized in {time.time() - start:.2f} sec")
    return qa_chain

async def ask_query_async(query: str, qa_chain=None) -> dict:
    if not qa_chain:
        raise ValueError("Retriever QA chain not provided.")
    
    start = time.time()
    # Run LLM query in thread pool
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, 
        qa_chain.invoke, 
        {"query": query}
    )
    logger.info(f"LLM responded in {time.time() - start:.2f} sec")
    return response

# --- Async wrapper functions ---

async def get_qa_chain_for_url_async(source_url: str):
    """Async version of get_qa_chain_for_url"""
    docs = await load_pdf_from_url_async(source_url)
    return await retriever_from_docs_async(docs)

# --- Backward compatibility (sync functions) ---

def load_pdf_from_url(pdf_url: str):
    return asyncio.run(load_pdf_from_url_async(pdf_url))

def retriever_from_docs(docs):
    return asyncio.run(retriever_from_docs_async(docs))

def get_qa_chain_for_url(source_url: str):
    return asyncio.run(get_qa_chain_for_url_async(source_url))

def ask_query(query: str, qa_chain=None) -> dict:
    return asyncio.run(ask_query_async(query, qa_chain))

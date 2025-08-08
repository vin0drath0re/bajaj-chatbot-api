# model.py

import os
import time
import requests
import tempfile
import logging
from dotenv import load_dotenv

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

# --- Utilities ---

def load_pdf_from_url(pdf_url: str):
    start = time.time()
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
    except Exception as e:
        logger.exception("Failed to download PDF")
        raise ValueError(f"Error fetching PDF: {str(e)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    os.unlink(tmp_file_path)

    logger.info(f"PDF downloaded and loaded in {time.time() - start:.2f} sec")
    return docs

def retriever_from_docs(docs):
    start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    logger.info(f"Split into {len(splits)} chunks")

    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30}
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

def ask_query(query: str, qa_chain=None) -> dict:
    if not qa_chain:
        raise ValueError("Retriever QA chain not provided.")
    
    start = time.time()
    response = qa_chain.invoke({"query": query})
    logger.info(f"LLM responded in {time.time() - start:.2f} sec")
    return response

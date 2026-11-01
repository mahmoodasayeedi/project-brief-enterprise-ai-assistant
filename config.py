"""Central settings for the IronStore Enterprise AI Assistant."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DOCS_DIR = os.getenv("DOCS_DIR", "documents")          # folder with the company documents
STORE_PATH = os.getenv("STORE_PATH", "vector_store.json")  # saved vector database
CATALOG_PATH = os.getenv("CATALOG_PATH", "doc_catalog.json")  # info about every document
REPORT_PATH = os.getenv("REPORT_PATH", "data_report.md")     # data-quality report

# --- OpenAI models (model routing: cheap vs smart) ---
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "gpt-5.4-mini")
SMART_MODEL = os.getenv("SMART_MODEL", "gpt-5.4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "512"))
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ironstore-assistant-v2")
# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "6"))
# If the best chunk is less similar than this, the assistant says "I don't know".
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))

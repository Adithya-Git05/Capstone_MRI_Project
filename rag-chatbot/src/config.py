from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
KB_DIR = ROOT_DIR / "knowledge_base"
VECTOR_DIR = KB_DIR / "vector_db"
CHUNKS_PATH = KB_DIR / "chunks.json"
PATIENT_PATH = DATA_DIR / "demo_patient.json"
METADATA_PATH = DATA_DIR / "literature_metadata.csv"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CHUNKS_PER_PAPER = int(os.getenv("MAX_CHUNKS_PER_PAPER", "2"))
RRF_K = int(os.getenv("RRF_K", "60"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))

for directory in (RAW_PDF_DIR, PROCESSED_DIR, KB_DIR, VECTOR_DIR):
    directory.mkdir(parents=True, exist_ok=True)

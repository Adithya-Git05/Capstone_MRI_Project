from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.config import CHUNKS_PATH, PATIENT_PATH
from src.patient_context import load_patient_context
from src.rag_pipeline import RAGPipeline
from src.schemas import QueryRequest, QueryResponse

app = FastAPI(title="Brain Tumor Surgical Planning RAG Prototype")
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
_pipeline: RAGPipeline | None = None


def pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

@app.get("/")
def index():
    return FileResponse(frontend_dir / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "knowledge_base_exists": CHUNKS_PATH.exists(), "ollama_checked_on_query": True}

@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    try:
        return load_patient_context(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.post("/api/rag/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        return pipeline().query(request.question, request.case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/api/knowledge-base/rebuild")
def rebuild():
    global _pipeline
    from src.ingestion import ingest_directory
    chunks = ingest_directory()
    _pipeline = RAGPipeline(chunks)
    return {"status": "rebuilt", "chunks": len(chunks), "note": "Only local PDFs with metadata are included."}

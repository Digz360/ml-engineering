from contextlib import asynccontextmanager
from src.llm_system.orchestrator import LLMOrchestrator
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from src.minigpt.model_loader import load_minigpt
from src.minigpt.inference_engine import InferenceEngine
from src.llm_system.inference_service import InferenceService
from src.rag.ingestion import chunk_markdown, chunk_text
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import VectorStore
from src.rag.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from src.rag.reranker import Reranker
from src.rag.context_builder import ContextBuilder
from src.rag.rag_service import RAGService
import logging
import os
from fastapi.staticfiles import StaticFiles
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "models/minigpt_v1_inference.pt"
TOKENIZER_PATH = "models/tokenizer_large.json"

inference_engine = None
inference_service = None
orchestrator = None
rag_service = None
tokenizer = None
embedding_service = None
reranker = None



class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    use_rag: bool = False
    max_new_tokens: int = Field(default=100, ge=1, le=500)

def build_rag_service(chunks, tokenizer, embedding_service, reranker):

    embeddings = embedding_service.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    vector_store = VectorStore()
    vector_store.add(chunks, embeddings)

    dense = DenseRetriever(
        embedding_service,
        vector_store,
    )

    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25)

    context_builder = ContextBuilder(
        tokenizer,
        max_context_tokens=600,
    )

    return RAGService(
        hybrid,
        reranker,
        context_builder,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, inference_service, orchestrator
    global rag_service, tokenizer, embedding_service, reranker

    print("Starting AI Platform...")
    print("Loading MiniGPT...")

    model, tokenizer, checkpoint, device = load_minigpt(
        MODEL_PATH,
        TOKENIZER_PATH,
    )

    print(f"MiniGPT loaded on {device}")

    inference_engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    inference_engine.start()

    inference_service = InferenceService(
        engine=inference_engine,
    )

    print("Inference Engine started.")

    print("Loading RAG system...")

    embedding_service = EmbeddingService()
    reranker = Reranker()

    chunks = chunk_markdown(
        "data/rag/documents/authentication.md"
    )

    rag_service = build_rag_service(
        chunks,
        tokenizer,
        embedding_service,
        reranker,
    )

    print("RAG system loaded.")

    orchestrator = LLMOrchestrator(
        inference_service=inference_service,
        rag_service=rag_service,
    )

    yield

    print("Shutting down AI Platform...")


app = FastAPI(
    title="MiniGPT AI Platform",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="AI Platform is not ready.",
        )

    try:
        result = orchestrator.handle_request(
            prompt=request.prompt,
            use_rag=request.use_rag,
            max_new_tokens=request.max_new_tokens,
        )

        return result

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )



@app.post("/rag/upload")
async def upload_rag_file(file: UploadFile = File(...)):
    global rag_service, tokenizer, embedding_service, reranker
    
    allowed = {".txt", ".md"}
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Only MD and TXT files are supported.",
        )

    content = await file.read()
    text = content.decode("utf-8")

    chunks = chunk_text(
        text=text,
        source=file.filename,
    )


    rag_service = build_rag_service(
        chunks,
        tokenizer,
        embedding_service,
        reranker,
    )

    orchestrator.rag_service = rag_service

    return {
        "status": "uploaded",
        "filename": file.filename,
        "characters": len(text),
        "chunks": len(chunks),
    }


app.mount(
    "/",
    StaticFiles(directory="src/api/static", html=True),
    name="static",
)
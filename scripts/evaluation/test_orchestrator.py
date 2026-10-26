from src.rag.ingestion import chunk_markdown
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import VectorStore
from src.rag.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from src.rag.reranker import Reranker
from src.rag.context_builder import ContextBuilder
from src.rag.rag_service import RAGService
from src.llm_system.orchestrator import LLMOrchestrator
from src.minigpt.model_loader import load_minigpt
from src.minigpt.inference_engine import InferenceEngine
from src.llm_system.inference_service import InferenceService


# MiniGPT tokenizer
model, tokenizer, checkpoint, device = load_minigpt(
    "models/minigpt_large_v2_sft_best.pt",
    "models/tokenizer_large.json",
    device="cpu",
)

engine = InferenceEngine(model, tokenizer, device)
engine.start()

inference_service = InferenceService(engine)


# RAG setup
chunks = chunk_markdown(
    "data/rag/documents/authentication.md"
)

embedding_service = EmbeddingService()

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

hybrid = HybridRetriever(
    dense,
    bm25,
)

reranker = Reranker()

context_builder = ContextBuilder(
    tokenizer,
    max_context_tokens=600,
)

rag_service = RAGService(
    hybrid,
    reranker,
    context_builder,
)


# Orchestrator
orchestrator = LLMOrchestrator(
    inference_service=inference_service,
    rag_service=rag_service,
)



result = orchestrator.handle_request(
    "Explica em poucas palavras o que é inteligência artificial.",
    use_rag=False,
)

print("\nROUTE:", result["route"])
print("\nANSWER:")
print(result["answer"])
print("\nSOURCES:")
print(result["sources"])
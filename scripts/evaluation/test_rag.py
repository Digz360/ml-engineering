from src.rag.ingestion import chunk_markdown
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import VectorStore
from src.rag.retriever import (
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
)
from src.rag.reranker import Reranker
from src.rag.context_builder import ContextBuilder
from src.minigpt.model_loader import load_minigpt

MODEL_PATH = "models/minigpt_large_v2_sft_best.pt"
TOKENIZER_PATH = "models/tokenizer_large.json"

model, tokenizer, checkpoint, device = load_minigpt(
    MODEL_PATH,
    TOKENIZER_PATH,
    device="cpu",
)

# INGESTION
chunks = chunk_markdown(
    "data/rag/documents/authentication.md"
)


# EMBEDDINGS
embedding_service = EmbeddingService()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_service.embed_documents(texts)


# VECTOR STORE
vector_store = VectorStore()

vector_store.add(
    chunks,
    embeddings,
)


# RETRIEVER
retriever = DenseRetriever(
    embedding_service=embedding_service,
    vector_store=vector_store,
)

bm25_retriever = BM25Retriever(chunks)

hybrid_retriever = HybridRetriever(
    dense_retriever=retriever,
    bm25_retriever=bm25_retriever,
)

reranker = Reranker()

context_builder = ContextBuilder(
    tokenizer=tokenizer,
    max_context_tokens=600,
)

# TESTE
# ============================================================
# RAG END-TO-END TEST
# ============================================================

query = "Como posso obter um novo access token sem voltar a fazer login?"

print("\n==============================")
print("1. USER QUERY")
print("==============================")
print(query)


# 1. RETRIEVAL
hybrid_results = hybrid_retriever.retrieve(
    query,
    top_k=6,
)

print("\n==============================")
print("2. HYBRID RETRIEVAL")
print("==============================")

for i, result in enumerate(hybrid_results, start=1):
    print(
        f"{i}. {result['chunk']['metadata']['section']} "
        f"| RRF: {result['rrf_score']:.5f}"
    )


# 2. RERANKING
reranked_results = reranker.rerank(
    query,
    hybrid_results,
    top_k=3,
)

print("\n==============================")
print("3. RERANKING - TOP 3")
print("==============================")

for i, result in enumerate(reranked_results, start=1):
    print(
        f"{i}. {result['chunk']['metadata']['section']} "
        f"| Score: {result['reranker_score']:.4f}"
    )


# 3. CONTEXT CONSTRUCTION
context_result = context_builder.build(
    reranked_results
)

context = context_result["context"]

print("\n==============================")
print("4. FINAL RAG CONTEXT")
print("==============================")
print(context)

print(
    f"\nContext tokens: {context_result['token_count']}"
)


# 4. BUILD PROMPT
rag_prompt = f"""Utilizador: Responde à pergunta usando apenas o contexto fornecido.

Contexto:
{context}

Pergunta:
{query}

Assistente:"""

prompt_ids = tokenizer.encode(rag_prompt).ids


# 5. TOKEN BUDGET
BLOCK_SIZE = 1024
MAX_RESPONSE_TOKENS = 200

available_tokens = BLOCK_SIZE - len(prompt_ids)

if available_tokens <= 0:
    raise ValueError("Prompt too large")

max_new_tokens = min(
    MAX_RESPONSE_TOKENS,
    available_tokens
)

print("\n==============================")
print("5. TOKEN BUDGET")
print("==============================")
print(f"MiniGPT limit:      {BLOCK_SIZE}")
print(f"Prompt tokens:      {len(prompt_ids)}")
print(f"Available tokens:   {available_tokens}")
print(f"Max answer tokens:  {max_new_tokens}")


# 6. GENERATION
import torch

input_ids = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device,
)

eos_id = tokenizer.token_to_id("[EOS]")

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        eos_id=eos_id,
    )

generated_ids = output_ids[0].tolist()[len(prompt_ids):]

answer = tokenizer.decode(generated_ids)


print("\n==============================")
print("6. MINIGPT ANSWER")
print("==============================")
print(answer)


print("\n==============================")
print("7. SOURCES")
print("==============================")

for source in context_result["sources"]:
    print(
        f"- {source['source']} -> {source['section']}"
    )

print("\n==============================")
print("RAG TEST COMPLETE")
print("==============================")
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def embed_documents(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str):
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )
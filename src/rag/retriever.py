from rank_bm25 import BM25Okapi
import re
class DenseRetriever:

    def __init__(
        self,
        embedding_service,
        vector_store,
        min_score: float | None = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:

        query_embedding = self.embedding_service.embed_query(query)

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        if self.min_score is not None:
            results = [
                result
                for result in results
                if result["score"] >= self.min_score
            ]

        return results

    from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        self.tokenized_chunks = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[\w-]+\b", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "score": float(scores[index]),
                "chunk": self.chunks[index],
            })

        return results
        
class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 10,
    ) -> list[dict]:

        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query,
            top_k=candidate_k,
        )

        fused = {}

        # Dense ranking
        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk = result["chunk"]
            chunk_id = chunk["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        # BM25 ranking
        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk = result["chunk"]
            chunk_id = chunk["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        results = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return results[:top_k]        
from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 3,
    ) -> list[dict]:

        if not results:
            return []

        pairs = [
            [query, result["chunk"]["text"]]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            reranked_results.append({
                "chunk": result["chunk"],
                "reranker_score": float(score),
            })

        reranked_results.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        return reranked_results[:top_k]
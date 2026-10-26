import numpy as np


class VectorStore:

    def __init__(self):
        self.chunks = []
        self.embeddings = None

    def add(self, chunks: list[dict], embeddings):
        self.chunks.extend(chunks)

        embeddings = np.asarray(embeddings)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([
                self.embeddings,
                embeddings
            ])

    def search(self, query_embedding, top_k: int = 3):

        if self.embeddings is None:
            return []

        # embeddings are normalized,
        # so dot product = cosine similarity
        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "score": float(scores[index]),
                "chunk": self.chunks[index],
            })

        return results
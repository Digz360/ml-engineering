class RAGService:

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        context_builder,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder

    def retrieve_context(self, query: str):

        # 1. Retrieval
        results = self.hybrid_retriever.retrieve(
            query,
            top_k=6,
        )

        # 2. Reranking
        results = self.reranker.rerank(
            query,
            results,
            top_k=3,
        )

        # 3. Context
        return self.context_builder.build(results)
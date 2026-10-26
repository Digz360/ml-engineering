class ContextBuilder:

    def __init__(self, tokenizer, max_context_tokens: int = 600):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens

    def _count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(text).ids
        )

    def build(self, results: list[dict]) -> dict:

        context_parts = []
        sources = []
        seen_chunks = set()

        for result in results:

            chunk = result["chunk"]
            chunk_id = chunk["id"]

            if chunk_id in seen_chunks:
                continue

            source = chunk["metadata"]["source"]
            section = chunk["metadata"]["section"]

            formatted_chunk = (
                f"[Source: {source} | Section: {section}]\n"
                f"{chunk['text']}"
            )

            # Simula o contexto caso adicionemos este chunk
            candidate_context = "\n\n".join(
                context_parts + [formatted_chunk]
            )

            if self._count_tokens(candidate_context) > self.max_context_tokens:
                break

            context_parts.append(formatted_chunk)
            seen_chunks.add(chunk_id)

            sources.append({
                "source": source,
                "section": section,
            })

        context = "\n\n".join(context_parts)

        return {
            "context": context,
            "sources": sources,
            "token_count": self._count_tokens(context),
        }
from src.llm_system.router import RequestRouter, Route
import time
import uuid
import logging

logger = logging.getLogger(__name__)

class LLMOrchestrator:

    def __init__(self, inference_service, rag_service=None):
        self.router = RequestRouter()
        self.inference_service = inference_service
        self.rag_service = rag_service

    def handle_request(self, prompt: str, use_rag: bool, max_new_tokens: int = 100,):
        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        route = self.router.route(use_rag)

        # RAG
        if route == Route.RAG:
            if self.rag_service is None:
                raise RuntimeError("RAG service is not configured")

            rag_result = self.rag_service.retrieve_context(prompt)

            final_prompt = (
                "Responde à pergunta usando apenas o contexto fornecido.\n\n"
                f"Contexto:\n{rag_result['context']}\n\n"
                f"Pergunta:\n{prompt}"
            )

            answer = self.inference_service.generate(final_prompt, max_new_tokens=max_new_tokens)

            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "trace_id=%s route=%s latency_ms=%.2f",
                trace_id,
                route.value,
                latency_ms,
            )

            
            return {
                "route": "rag",
                "answer": answer,
                "sources": rag_result["sources"],
                "trace_id": trace_id,
                "latency_ms": round(latency_ms, 2),
            }

        # DIRECT
        answer = self.inference_service.generate(prompt, max_new_tokens=max_new_tokens)

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "trace_id=%s route=%s latency_ms=%.2f",
            trace_id,
            route.value,
            latency_ms,
        )

        return {
            "route": "direct",
            "answer": answer,
            "sources": [],
            "trace_id": trace_id,
            "latency_ms": round(latency_ms, 2),
        }
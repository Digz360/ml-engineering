from enum import Enum


class Route(Enum):
    DIRECT = "direct"
    RAG = "rag"


class RequestRouter:

    def route(self, use_rag: bool) -> Route:

        if use_rag:
            return Route.RAG

        return Route.DIRECT
from __future__ import annotations

from ..embeddings import DistilBertEmbedder
from ..vectordb import NumpyVectorStore


class Retriever:
    def __init__(self, embedder: DistilBertEmbedder, vector_store: NumpyVectorStore, top_k: int = 5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        query_embedding = self.embedder.encode([question])[0]
        return self.vector_store.search(query_embedding, top_k or self.top_k)

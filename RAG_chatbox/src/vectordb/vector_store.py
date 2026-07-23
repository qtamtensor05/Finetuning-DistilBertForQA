from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils.schemas import Chunk


class NumpyVectorStore:
    def __init__(self):
        self.embeddings: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if len(chunks) == 0:
            return
        embeddings = embeddings.astype(np.float32)
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        if top_k <= 0:
            return []
        query = query_embedding.reshape(-1).astype(np.float32)
        scores = self.embeddings @ query
        top_k = min(top_k, len(scores))
        indexes = np.argpartition(-scores, top_k - 1)[:top_k]
        indexes = indexes[np.argsort(-scores[indexes])]
        return [
            {
                "chunk": self.chunks[int(index)],
                "similarity": float(scores[int(index)]),
            }
            for index in indexes
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        embeddings = self.embeddings if self.embeddings is not None else np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(directory / "embeddings.npz", embeddings=embeddings)
        (directory / "chunks.json").write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "NumpyVectorStore":
        directory = Path(directory)
        store = cls()
        embeddings_path = directory / "embeddings.npz"
        chunks_path = directory / "chunks.json"
        if not embeddings_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Vector index not found in: {directory}")
        store.embeddings = np.load(embeddings_path)["embeddings"].astype(np.float32)
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        store.chunks = [Chunk.from_dict(item) for item in chunks_data]
        return store

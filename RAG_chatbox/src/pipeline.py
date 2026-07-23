from __future__ import annotations

from pathlib import Path

import yaml

from .chunking import TextChunker
from .embeddings import DistilBertEmbedder
from .ingestion import load_documents
from .qa import ExtractiveQAReader
from .retrieval import Retriever
from .utils.paths import RAG_ROOT, resolve_path
from .vectordb import NumpyVectorStore


class RAGPipeline:
    def __init__(
        self,
        reader: ExtractiveQAReader,
        chunker: TextChunker,
        embedder: DistilBertEmbedder,
        vector_store: NumpyVectorStore,
        top_k: int = 5,
        vector_dir: str | Path = RAG_ROOT / "vector_store",
    ):
        self.reader = reader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = Retriever(embedder, vector_store, top_k=top_k)
        self.top_k = top_k
        self.vector_dir = Path(vector_dir)

    @classmethod
    def from_config(cls, config_path: str | Path = RAG_ROOT / "config.yaml") -> "RAGPipeline":
        config_path = Path(config_path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        model_dir = resolve_path(data.get("model", {}).get("model_dir", "../outputs/checkpoints_vi/best_model"))
        qa_config = data.get("qa", {})
        reader = ExtractiveQAReader.from_model_dir(
            model_dir,
            max_answer_tokens=int(qa_config.get("max_answer_tokens", 30)),
        )

        chunking_config = data.get("chunking", {})
        chunker = TextChunker(
            chunk_size=int(chunking_config.get("chunk_size", 900)),
            chunk_overlap=int(chunking_config.get("chunk_overlap", 180)),
        )

        embedding_config = data.get("embedding", {})
        embedder = DistilBertEmbedder(
            tokenizer=reader.tokenizer,
            encoder=reader.encoder,
            device=reader.device,
            max_length=int(embedding_config.get("max_length", 256)),
            batch_size=int(embedding_config.get("batch_size", 16)),
        )

        storage_config = data.get("storage", {})
        vector_dir = resolve_path(storage_config.get("vector_dir", "vector_store"))
        return cls(
            reader=reader,
            chunker=chunker,
            embedder=embedder,
            vector_store=NumpyVectorStore(),
            top_k=int(data.get("retrieval", {}).get("top_k", 5)),
            vector_dir=vector_dir,
        )

    @property
    def device(self):
        return self.reader.device

    def build_index(self, paths: list[str | Path], save: bool = True) -> None:
        documents = load_documents(paths)
        chunks = self.chunker.split_documents(documents)
        if not chunks:
            raise ValueError("No text chunks were created from the provided data.")
        embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        self.vector_store = NumpyVectorStore()
        self.vector_store.add(chunks, embeddings)
        self.retriever = Retriever(self.embedder, self.vector_store, top_k=self.top_k)
        if save:
            self.save_index()

    def save_index(self) -> None:
        self.vector_store.save(self.vector_dir)

    def load_index(self) -> None:
        self.vector_store = NumpyVectorStore.load(self.vector_dir)
        self.retriever = Retriever(self.embedder, self.vector_store, top_k=self.top_k)

    def answer(self, question: str, top_k: int | None = None) -> dict:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved:
            raise ValueError("Vector index is empty. Build an index with --data before asking questions.")

        best_result: dict | None = None
        contexts = []
        for item in retrieved:
            chunk = item["chunk"]
            qa_result = self.reader.answer(question, chunk.text)
            enriched = {
                **qa_result,
                "context": chunk,
                "similarity": item["similarity"],
            }
            contexts.append(
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "similarity": item["similarity"],
                    "qa_score": qa_result["score"],
                }
            )
            if best_result is None or qa_result["score"] > best_result["score"]:
                best_result = enriched

        assert best_result is not None
        return {
            "answer": best_result["answer"],
            "score": best_result["score"],
            "source": best_result["context"].source,
            "similarity": best_result["similarity"],
            "contexts": contexts,
        }

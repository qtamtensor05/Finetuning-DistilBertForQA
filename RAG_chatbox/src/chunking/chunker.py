from __future__ import annotations

import hashlib
import re

from ..utils.schemas import Chunk, Document


class TextChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 180):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_document(self, document: Document) -> list[Chunk]:
        text = _normalize_whitespace(document.text)
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            end = _prefer_boundary(text, start, end, self.chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = _chunk_id(document.source, start, end, chunk_text)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        source=document.source,
                        start_char=start,
                        end_char=end,
                        metadata=document.metadata.copy(),
                    )
                )
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _prefer_boundary(text: str, start: int, end: int, chunk_size: int) -> int:
    if end >= len(text):
        return len(text)

    window_start = max(start + chunk_size // 2, start)
    candidates = [
        text.rfind("\n\n", window_start, end),
        text.rfind(". ", window_start, end),
        text.rfind("? ", window_start, end),
        text.rfind("! ", window_start, end),
        text.rfind("; ", window_start, end),
        text.rfind(", ", window_start, end),
        text.rfind(" ", window_start, end),
    ]
    boundary = max(candidates)
    if boundary <= start:
        return end
    if text[boundary : boundary + 2] in {". ", "? ", "! ", "; ", ", "}:
        return boundary + 1
    return boundary


def _chunk_id(source: str, start: int, end: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{start}:{end}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"{digest}:{start}-{end}"

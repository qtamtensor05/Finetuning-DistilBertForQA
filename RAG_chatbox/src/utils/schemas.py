from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    source: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            start_char=int(data["start_char"]),
            end_char=int(data["end_char"]),
            metadata=dict(data.get("metadata") or {}),
        )

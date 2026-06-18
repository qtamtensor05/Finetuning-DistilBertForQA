from __future__ import annotations

from pathlib import Path


RAG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, base: Path = RAG_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()

"""A small on-disk vector store for matching text against embedded entities.

Embeddings are kept in a NumPy array next to a JSON file of ids and texts, and
searched by brute force. That is plenty fast for entity lists and spreadsheets
(up to hundreds of thousands of rows) and needs nothing beyond NumPy.
"""

from __future__ import annotations

__all__ = ["VectorStore"]

import json
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document

VECTORS_FILE = "vectors.npy"
DOCUMENTS_FILE = "documents.json"


class VectorStore:
    """Embedded documents that can be searched by similarity to a query.

    Distances are squared Euclidean (L2) distances, so smaller is closer. For
    unit-length embeddings such as OpenAI's this equals ``2 - 2 * cosine``.
    """

    def __init__(
        self,
        embeddings: Any,
        ids: list[str],
        texts: list[str],
        vectors: np.ndarray,
        embedding_model: str = "",
    ) -> None:
        self.embeddings = embeddings
        self.ids = ids
        self.texts = texts
        self.vectors = vectors
        self.embedding_model = embedding_model

    @staticmethod
    def exists(path: str | Path, embedding_model: str = "") -> bool:
        """Whether ``path`` holds a store (built with ``embedding_model``, if given)."""
        path = Path(path)
        documents = path / DOCUMENTS_FILE
        if not (path / VECTORS_FILE).is_file() or not documents.is_file():
            return False
        if not embedding_model:
            return True
        try:
            stored = json.loads(documents.read_text()).get("embedding_model", "")
        except (OSError, ValueError):
            return False
        return stored == embedding_model

    @classmethod
    def build(
        cls,
        path: str | Path,
        embeddings: Any,
        texts: list[str],
        ids: list[str],
        embedding_model: str = "",
    ) -> "VectorStore":
        """Embed ``texts``, save them under ``path`` and return the store."""
        if len(texts) != len(ids):
            raise ValueError("texts and ids must be the same length")
        vectors = (
            np.asarray(embeddings.embed_documents(texts), dtype=np.float32)
            if texts
            else np.zeros((0, 0), dtype=np.float32)
        )
        store = cls(embeddings, list(ids), list(texts), vectors, embedding_model)
        store.save(path)
        return store

    @classmethod
    def load(cls, path: str | Path, embeddings: Any) -> "VectorStore":
        path = Path(path)
        data = json.loads((path / DOCUMENTS_FILE).read_text())
        vectors = np.load(path / VECTORS_FILE)
        return cls(
            embeddings,
            data["ids"],
            data["texts"],
            vectors,
            data.get("embedding_model", ""),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / VECTORS_FILE, self.vectors)
        (path / DOCUMENTS_FILE).write_text(
            json.dumps(
                {
                    "embedding_model": self.embedding_model,
                    "ids": self.ids,
                    "texts": self.texts,
                }
            )
        )

    def __len__(self) -> int:
        return len(self.ids)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Return the ``k`` closest documents to ``query`` with their distances."""
        if not len(self) or k < 1:
            return []
        q = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        distances = (
            (self.vectors**2).sum(axis=1) + float(q @ q) - 2.0 * (self.vectors @ q)
        )
        distances = np.maximum(distances, 0.0)
        k = min(k, len(self))
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest = nearest[np.lexsort((nearest, distances[nearest]))]
        return [
            (
                Document(id=self.ids[i], page_content=self.texts[i]),
                float(distances[i]),
            )
            for i in nearest
        ]

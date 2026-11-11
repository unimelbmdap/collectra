"""BM25 search over short texts, such as entity names and spreadsheet rows.

Each text is split into words, and each word also into its character trigrams,
so a misspelt or misread name (``Monograptas``) still shares most of its terms
with the right entry (``Monograptus``). Texts are scored with Okapi BM25: terms
that few texts contain count for more, and a long text needs more matches to
score as well as a short one.

It needs nothing beyond NumPy, sends nothing to any service and costs nothing
to run. Entity lists and spreadsheets of hundreds of thousands of rows index in
seconds.
"""

from __future__ import annotations

__all__ = ["BM25Index", "Document", "tokenize"]

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOCUMENTS_FILE = "documents.json"

_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Document:
    """One indexed entry: its id and its text."""

    id: str
    page_content: str


def _fold(text: str) -> str:
    """Lower case, without accents, so ``Müller`` and ``muller`` are the same."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def tokenize(text: str) -> list[str]:
    """Each word of ``text``, followed by the trigrams of ``#word#``."""
    tokens = []
    for word in _WORD.findall(_fold(text)):
        tokens.append(word)
        padded = f"#{word}#"
        tokens.extend(padded[i : i + 3] for i in range(len(padded) - 2))
    return tokens


class BM25Index:
    """Texts that can be searched by how well they match a query.

    Scores are BM25 scores: higher is better, and only texts that share at
    least one term with the query score above zero.
    """

    def __init__(
        self,
        ids: list[str],
        texts: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must be the same length")
        self.ids = [str(i) for i in ids]
        self.texts = list(texts)

        counts = [Counter(tokenize(text)) for text in self.texts]
        lengths = np.array([sum(c.values()) for c in counts], dtype=np.float64)
        average = lengths.mean() if lengths.size and lengths.sum() else 1.0
        # The part of each text's term weights that depends on its length.
        length_norm = k1 * (1 - b + b * lengths / average)

        postings: dict[str, tuple[list[int], list[int]]] = {}
        for index, text_counts in enumerate(counts):
            for term, count in text_counts.items():
                docs, freqs = postings.setdefault(term, ([], []))
                docs.append(index)
                freqs.append(count)

        # Every term's contribution to each text's score, worked out once.
        size = len(self.texts)
        self._weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, (docs, freqs) in postings.items():
            docs_array = np.array(docs)
            tf = np.array(freqs, dtype=np.float64)
            idf = math.log(1 + (size - len(docs) + 0.5) / (len(docs) + 0.5))
            weights = idf * tf * (k1 + 1) / (tf + length_norm[docs_array])
            self._weights[term] = (docs_array, weights)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """The ``k`` best-matching texts, best first, with their scores.

        Texts that share no term with the query are left out, so fewer than
        ``k`` may come back. Equal scores keep the texts' order.
        """
        if not len(self) or k < 1:
            return []
        scores = np.zeros(len(self))
        for term in sorted(set(tokenize(query))):
            if term in self._weights:
                docs, weights = self._weights[term]
                scores[docs] += weights
        matched = np.flatnonzero(scores > 0)
        best = matched[np.lexsort((matched, -scores[matched]))][:k]
        return [
            (Document(id=self.ids[i], page_content=self.texts[i]), float(scores[i]))
            for i in best
        ]

    @staticmethod
    def exists(path: str | Path) -> bool:
        return (Path(path) / DOCUMENTS_FILE).is_file()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / DOCUMENTS_FILE).write_text(
            json.dumps({"ids": self.ids, "texts": self.texts})
        )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        data = json.loads((Path(path) / DOCUMENTS_FILE).read_text())
        return cls(data["ids"], data["texts"])

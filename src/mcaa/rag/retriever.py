"""Hybrid retrieval: BM25 + TF-IDF cosine, fused with reciprocal rank."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .store import Chunk
from .text import content_tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused, 0..1
    bm25: float
    dense: float
    coverage: float  # share of query content terms found in the chunk


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [content_tokens(f"{c.title} {c.section} {c.text}") for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = float(np.mean([len(d) for d in self.docs])) if self.docs else 1.0
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        texts = [" ".join(d) for d in self.docs] or [""]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform(texts)

    def _bm25(self, q: list[str]) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        q = content_tokens(query)
        if not q:
            return []
        bm = self._bm25(q)
        dense = (self.mat @ self.vec.transform([" ".join(q)]).T).toarray().ravel()
        rrf = np.zeros(len(self.chunks))
        for arr in (bm, dense):
            for rank, i in enumerate(np.argsort(-arr)):
                rrf[i] += 1.0 / (60 + rank)
        order = np.argsort(-rrf)[:k]
        qs = set(q)
        top = rrf.max() or 1.0
        return [
            Hit(self.chunks[i], float(rrf[i] / top), float(bm[i]), float(dense[i]), len(qs & set(self.docs[i])) / len(qs))
            for i in order
            if bm[i] > 0 or dense[i] > 0
        ]

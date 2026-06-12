import re
from functools import lru_cache

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from convagent.embedder import get_embeddings

# All retrieval methods supported by retrieve()
METHODS = ["l2", "cosine", "dot", "manhattan", "chebyshev", "bm25", "hybrid"]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _load_corpus():
    """Load every chunk, its embedding and metadata from the persisted Chroma DB."""
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_embeddings()
    )
    data = vectordb.get(include=["documents", "embeddings", "metadatas"])
    texts = data["documents"]
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    metadatas = data["metadatas"] or [{} for _ in texts]
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    return texts, embeddings, metadatas, bm25


def _vector_scores(query_vec: np.ndarray, doc_embs: np.ndarray, method: str) -> np.ndarray:
    """Return one score per document; higher is always better."""
    if method == "cosine":
        q = query_vec / np.linalg.norm(query_vec)
        d = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
        return d @ q
    if method == "dot":
        return doc_embs @ query_vec
    # Distance metrics: negate so that higher = closer
    if method == "l2":
        return -np.linalg.norm(doc_embs - query_vec, axis=1)
    if method == "manhattan":
        return -np.abs(doc_embs - query_vec).sum(axis=1)
    if method == "chebyshev":
        return -np.abs(doc_embs - query_vec).max(axis=1)
    raise ValueError(f"Unknown vector method: {method}")


def _rrf(rankings: list[np.ndarray], n_docs: int, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: combine multiple rankings into one score per doc."""
    scores = np.zeros(n_docs)
    for ranked_indices in rankings:
        for rank, idx in enumerate(ranked_indices):
            scores[idx] += 1.0 / (k + rank + 1)
    return scores


def retrieve(query: str, method: str = "l2", k: int = 10) -> list[tuple[Document, float]]:
    """Retrieve the top-k chunks for a query using the given method.

    Returns (Document, score) pairs sorted best-first. Scores are only
    comparable within a single method, not across methods.
    """
    texts, embeddings, metadatas, bm25 = _load_corpus()

    if method == "bm25":
        scores = bm25.get_scores(_tokenize(query))
    elif method == "hybrid":
        # Fuse BM25 (keyword) and cosine (semantic) rankings with RRF
        bm25_scores = bm25.get_scores(_tokenize(query))
        query_vec = np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
        cos_scores = _vector_scores(query_vec, embeddings, "cosine")
        scores = _rrf(
            [np.argsort(-bm25_scores), np.argsort(-cos_scores)],
            n_docs=len(texts),
        )
    elif method in METHODS:
        query_vec = np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
        scores = _vector_scores(query_vec, embeddings, method)
    else:
        raise ValueError(f"Unknown method {method!r}, expected one of {METHODS}")

    top = np.argsort(-scores)[:k]
    return [
        (Document(page_content=texts[i], metadata=metadatas[i]), float(scores[i]))
        for i in top
    ]

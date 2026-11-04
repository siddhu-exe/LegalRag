"""
BM25 retriever on ``bm25s`` with a memory-mapped index (plan task C4, behind a flag).

Why: in the v1 RAM baseline (``results/ram_baseline.json``) the ``rank_bm25`` index added
4.04 GiB of a 9.30 GiB peak and took about 2.7 s of the 3.4 s mean retrieval-and-rerank time per
query, because ``BM25Okapi`` keeps a Python dict of term counts per chunk and scores every chunk in
Python. ``bm25s`` keeps the scores as a sparse matrix in three numpy arrays, which can be opened
with ``mmap`` so the operating system pages in only the postings of the query terms. Whether this
meets the recall, RAM and speed targets is for plan task P5 to measure on the real corpus; nothing
here claims a measured gain.

Same tokenizer, same parameters (k1 = 1.5, b = 0.75), same interface as ``BM25Retriever``:
``search(query, top_k) -> [(chunk_id, score)]``, ``chunk_ids``, ``validate()``.

How the scores relate to v1 (tested in ``tests/test_bm25s_backend.py``): with the "robertson"
method every chunk's score equals ``BM25Okapi``'s divided by (k1 + 1), to float32 precision, as long
as no query term is in more than half of all chunks. For such very common terms ``rank_bm25`` gives
a small positive weight (a quarter of the average idf) and ``bm25s`` gives zero. Results with a
score of zero are not returned (v1 returned arbitrary zero-score chunks when nothing matched).

Directory layout written by ``save`` / read by ``load``::

    <dir>/chunk_ids.npy            ids in index order (memory-mappable)
    <dir>/legalrag_bm25s.json      format, tokenizer name, k1, b, method, num_docs, bm25s version
    <dir>/data.csc.index.npy ...   the bm25s files (scores, indices, indptr, vocab, params)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from legalrag.retrieval.bm25 import tokenize_legal_text
from legalrag.retrieval.chunk_ids import ChunkIdArray

try:
    import bm25s
except ImportError:  # the default (rank_bm25) backend works without it
    bm25s = None  # type: ignore

FORMAT_VERSION = 1
TOKENIZER_NAME = "legalrag.retrieval.bm25.tokenize_legal_text/lower+\\w+"
METADATA_FILE = "legalrag_bm25s.json"
CHUNK_IDS_FILE = "chunk_ids.npy"
DEFAULT_METHOD = "robertson"


def _require_bm25s() -> Any:
    if bm25s is None:
        raise ImportError("bm25s is required for the bm25s backend. Install it with: pip install bm25s")
    return bm25s


class Bm25sRetriever:
    """BM25 over a ``bm25s`` sparse index. Drop-in for ``BM25Retriever`` in the cascade."""

    def __init__(
        self,
        model: Optional[Any] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = DEFAULT_METHOD,
    ):
        self.model = model
        self.chunk_ids = chunk_ids if chunk_ids is not None else []
        self.k1 = k1
        self.b = b
        self.method = method

    # ------------------------------------------------------------------ build / save / load

    @classmethod
    def build(
        cls,
        chunk_ids: Sequence[str],
        texts: Iterable[str],
        k1: float = 1.5,
        b: float = 0.75,
        method: str = DEFAULT_METHOD,
        show_progress: bool = False,
    ) -> "Bm25sRetriever":
        """Indexes ``texts`` (one per chunk id, in order) with the v1 tokenizer.

        Build memory grows with the number of tokens in the corpus; try ``--max-chunks`` in
        ``scripts/build_retrieval_indexes_v2.py`` first and measure before the full corpus.
        """
        lib = _require_bm25s()
        vocab: Dict[str, int] = {}
        corpus_ids: List[List[int]] = []
        for text in texts:
            corpus_ids.append([vocab.setdefault(token, len(vocab)) for token in tokenize_legal_text(text)])
        if len(corpus_ids) != len(chunk_ids):
            raise ValueError(f"{len(corpus_ids)} texts but {len(chunk_ids)} chunk ids; they must match one to one.")
        if not corpus_ids:
            raise ValueError("Cannot build a BM25 index from an empty corpus.")
        if not vocab:
            raise ValueError("No chunk contains any token; nothing to index.")
        model = lib.BM25(k1=k1, b=b, method=method)
        model.index((corpus_ids, vocab), show_progress=show_progress)
        ids = chunk_ids if isinstance(chunk_ids, ChunkIdArray) else ChunkIdArray.from_ids(chunk_ids)
        return cls(model=model, chunk_ids=ids, k1=k1, b=b, method=method)

    def save(self, directory: Union[str, Path]) -> None:
        lib = _require_bm25s()
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(str(directory))
        ids = self.chunk_ids if isinstance(self.chunk_ids, ChunkIdArray) else ChunkIdArray.from_ids(self.chunk_ids)
        ids.save(directory / CHUNK_IDS_FILE)
        metadata = {
            "format": FORMAT_VERSION,
            "tokenizer": TOKENIZER_NAME,
            "k1": self.k1,
            "b": self.b,
            "method": self.method,
            "num_docs": len(ids),
            "bm25s_version": getattr(lib, "__version__", "unknown"),
        }
        (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Union[str, Path], mmap: bool = True) -> "Bm25sRetriever":
        """Opens a saved index. With ``mmap=True`` the score arrays and ids are paged in on demand.

        Fails closed: a missing metadata file, a different tokenizer or format, or a document count
        that disagrees with the ids raises ``ValueError``.
        """
        lib = _require_bm25s()
        directory = Path(directory)
        meta_path = directory / METADATA_FILE
        if not meta_path.is_file():
            raise ValueError(f"{directory} is not a bm25s index written by legalrag (missing {METADATA_FILE}).")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("format") != FORMAT_VERSION:
            raise ValueError(f"Unsupported bm25s index format {metadata.get('format')!r}; expected {FORMAT_VERSION}.")
        if metadata.get("tokenizer") != TOKENIZER_NAME:
            raise ValueError(
                f"Index was built with tokenizer {metadata.get('tokenizer')!r}, but queries use {TOKENIZER_NAME!r}; "
                "searching it would silently miss terms."
            )
        model = lib.BM25.load(str(directory), mmap=mmap, load_corpus=False)
        retriever = cls(
            model=model,
            chunk_ids=ChunkIdArray.load(directory / CHUNK_IDS_FILE, mmap=mmap),
            k1=metadata["k1"],
            b=metadata["b"],
            method=metadata["method"],
        )
        retriever.validate()
        if int(metadata["num_docs"]) != len(retriever.chunk_ids):
            raise ValueError("bm25s metadata and chunk_ids.npy disagree on the number of chunks.")
        return retriever

    # ------------------------------------------------------------------ use

    def validate(self) -> None:
        """Same contract as ``BM25Retriever.validate``: loaded model, ids, and matching sizes."""
        if self.model is None:
            raise ValueError("BM25 model is not loaded (model is None).")
        if not len(self.chunk_ids):
            raise ValueError("BM25 retriever has no chunk_ids loaded.")
        scores = getattr(self.model, "scores", None)
        if not scores or "num_docs" not in scores:
            raise ValueError("bm25s model has no index (call build or load first).")
        if int(scores["num_docs"]) != len(self.chunk_ids):
            raise ValueError(
                f"BM25 corpus size ({int(scores['num_docs'])}) does not match the number of "
                f"chunk_ids ({len(self.chunk_ids)}); the artifact is inconsistent."
            )

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Top ``top_k`` chunks by BM25 score, best first, ties broken by index order.

        Returns ``[]`` when no query term is in the index. Chunks with a score of zero are not returned.
        """
        if self.model is None or not len(self.chunk_ids):
            raise ValueError("BM25 model is not initialized or corpus is empty.")
        if top_k <= 0:
            return []
        vocab = self.model.vocab_dict
        token_ids = [vocab[t] for t in tokenize_legal_text(query) if t in vocab and t != ""]
        if not token_ids:
            return []

        scores = np.asarray(self.model.get_scores_from_ids(token_ids))
        if scores.shape[0] != len(self.chunk_ids):
            raise ValueError(
                f"BM25 score count ({scores.shape[0]}) does not match chunk_ids "
                f"({len(self.chunk_ids)}); the artifact is inconsistent."
            )
        total = scores.shape[0]
        k = min(top_k, total)
        candidates = np.argpartition(-scores, k - 1)[:k] if k < total else np.arange(total)
        candidates = candidates[scores[candidates] > 0]
        order = np.lexsort((candidates, -scores[candidates]))
        winners = candidates[order]
        ids = self.chunk_ids.take(winners) if isinstance(self.chunk_ids, ChunkIdArray) else [self.chunk_ids[i] for i in winners]
        return [(chunk_id, float(scores[i])) for chunk_id, i in zip(ids, winners)]

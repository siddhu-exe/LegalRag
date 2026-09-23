"""
BM25 Lexical Retriever for Legal Text.

Implements BM25Okapi (k1=1.5, b=0.75) with lowercased alphanumeric regex tokenization.

Serialized artifact contract
----------------------------
Production ``bm25.pkl`` artifacts are a pickled ``dict`` of the form::

    {
        "bm25": <rank_bm25.BM25Okapi object>,    # canonical key
        "chunk_ids": [<chunk id in BM25 corpus order>, ...],
        "k1": 1.5,                                # optional
        "b": 0.75,                                # optional
    }

``load()`` also accepts the legacy in-repo dict that used the key ``"model"`` instead of
``"bm25"``, and a bare pickled ``BM25Okapi`` object (the original Kaggle notebook output),
in which case ``chunk_ids`` must be supplied by the caller from the chunk metadata.
"""

import pickle
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional, Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

TOKEN_REGEX = re.compile(r"\w+")

MODEL_STATE_KEYS = ("bm25", "model")


def tokenize_legal_text(text: str) -> List[str]:
    """Tokenizes text using lowercased alphanumeric regex tokens."""
    if not text:
        return []
    return TOKEN_REGEX.findall(text.lower())


class BM25Retriever:
    """
    Lexical retrieval engine based on rank-bm25 BM25Okapi.
    """

    def __init__(
        self,
        chunk_ids: Optional[List[str]] = None,
        corpus_texts: Optional[List[str]] = None,
        bm25_model: Optional[Any] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if BM25Okapi is None and bm25_model is None and corpus_texts is not None:
            raise ImportError(
                "rank-bm25 is required to build a BM25Retriever from corpus texts. "
                "Install it with: pip install rank-bm25"
            )

        self.chunk_ids = chunk_ids or []
        self.k1 = k1
        self.b = b

        if bm25_model is not None:
            self.model = bm25_model
        elif corpus_texts is not None and BM25Okapi is not None:
            tokenized_corpus = [tokenize_legal_text(text) for text in corpus_texts]
            self.model = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        else:
            self.model = None

    def validate(self) -> None:
        """
        Verifies this retriever is structurally usable for search.

        Raises:
            ValueError: if the BM25 model is missing, no chunk IDs are mapped, or the
                number of chunk IDs does not match the BM25 corpus size.
        """
        if self.model is None:
            raise ValueError("BM25 model is not loaded (model is None).")
        if not self.chunk_ids:
            raise ValueError("BM25 retriever has no chunk_ids loaded.")
        corpus_size = getattr(self.model, "corpus_size", None)
        if corpus_size is not None and int(corpus_size) != len(self.chunk_ids):
            raise ValueError(
                f"BM25 corpus size ({int(corpus_size)}) does not match the number of "
                f"chunk_ids ({len(self.chunk_ids)}); the artifact is inconsistent."
            )

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Retrieves top_k chunk IDs with their BM25 scores for a query.

        Args:
            query: The user search query or question.
            top_k: Number of candidates to return.

        Returns:
            List of tuples: (chunk_id, bm25_score) sorted descending by score.
        """
        if self.model is None or not self.chunk_ids:
            raise ValueError("BM25 model is not initialized or corpus is empty.")

        tokenized_query = tokenize_legal_text(query)
        if not tokenized_query:
            return []

        scores = self.model.get_scores(tokenized_query)
        if len(scores) != len(self.chunk_ids):
            raise ValueError(
                f"BM25 score count ({len(scores)}) does not match chunk_ids "
                f"({len(self.chunk_ids)}); the artifact is inconsistent."
            )
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Serializes the BM25 model and chunk IDs to a pickle file.

        The canonical ``"bm25"`` key is written alongside the legacy ``"model"`` alias so
        the artifact stays compatible with both current and older loaders. Pickle
        memoization stores the shared model object only once, so the file size is
        unaffected.
        """
        if self.model is None:
            raise ValueError("Cannot save a BM25Retriever with no model loaded.")
        state = {
            "bm25": self.model,
            "model": self.model,
            "chunk_ids": self.chunk_ids,
            "k1": self.k1,
            "b": self.b,
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BM25Retriever":
        """
        Loads a serialized BM25Retriever from disk.

        Accepts the canonical production dict (``"bm25"``), the legacy in-repo dict
        (``"model"``), and a bare pickled ``BM25Okapi`` object. A dict carrying neither
        model key raises immediately instead of silently producing a retriever with
        ``model=None`` (the failure mode that broke production).
        """
        with open(file_path, "rb") as f:
            state = pickle.load(f)

        if BM25Okapi is not None and isinstance(state, BM25Okapi):
            return cls(bm25_model=state)

        if not isinstance(state, dict):
            raise TypeError(f"Unrecognized BM25 artifact format: {type(state)}")

        bm25_model = None
        for key in MODEL_STATE_KEYS:
            if state.get(key) is not None:
                bm25_model = state[key]
                break

        if bm25_model is None:
            raise ValueError(
                "BM25 artifact does not contain a serialized model under any of the "
                f"expected keys {MODEL_STATE_KEYS}; refusing to load an unusable retriever."
            )

        return cls(
            chunk_ids=state.get("chunk_ids", []),
            bm25_model=bm25_model,
            k1=state.get("k1", 1.5),
            b=state.get("b", 0.75),
        )

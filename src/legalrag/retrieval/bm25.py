"""
BM25 Lexical Retriever for Legal Text.

Implements BM25Okapi (k1=1.5, b=0.75) with lowercased alphanumeric regex tokenization.
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
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes the BM25 model and chunk IDs to a pickle file."""
        state = {
            "model": self.model,
            "chunk_ids": self.chunk_ids,
            "k1": self.k1,
            "b": self.b,
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BM25Retriever":
        """Loads a serialized BM25Retriever from disk."""
        with open(file_path, "rb") as f:
            state = pickle.load(f)

        # Handle raw BM25Okapi object or packaged dict
        if BM25Okapi is not None and isinstance(state, BM25Okapi):
            return cls(bm25_model=state)
        elif isinstance(state, dict):
            return cls(
                chunk_ids=state.get("chunk_ids", []),
                bm25_model=state.get("model"),
                k1=state.get("k1", 1.5),
                b=state.get("b", 0.75),
            )
        else:
            raise TypeError(f"Unrecognized BM25 artifact format: {type(state)}")

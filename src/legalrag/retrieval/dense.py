"""
Dense Vector Retriever using BAAI/bge-base-en-v1.5 and FAISS IndexFlatIP.
"""

from pathlib import Path
from typing import List, Tuple, Union, Optional, Any
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


class DenseRetriever:
    """
    Dense semantic retrieval engine powered by BGE-base-en-v1.5 embeddings
    and FAISS IndexFlatIP (cosine similarity over L2-normalized embeddings).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        index: Optional[Any] = None,
        chunk_ids: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model: Optional[Any] = None
        self.index = index
        self.chunk_ids = chunk_ids or []

    @property
    def model(self) -> Any:
        """Lazy load SentenceTransformer model."""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for DenseRetriever. "
                "Install it with: pip install sentence-transformers"
            )
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Encodes texts into normalized dense embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def build_index(self, embeddings: np.ndarray, chunk_ids: List[str]) -> None:
        """
        Builds a FAISS IndexFlatIP from normalized embeddings.

        Args:
            embeddings: Float32 numpy array of shape (N, dimension).
            chunk_ids: List of N chunk identifiers.
        """
        if faiss is None:
            raise ImportError("faiss is required to build a FAISS index. Install it with: pip install faiss-cpu")

        if len(embeddings) != len(chunk_ids):
            raise ValueError(f"Embedding count ({len(embeddings)}) != chunk ID count ({len(chunk_ids)})")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunk_ids = list(chunk_ids)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Encodes query and retrieves top_k candidate chunks.

        Args:
            query: User query string.
            top_k: Number of candidates to return.

        Returns:
            List of (chunk_id, cosine_score) sorted descending by score.
        """
        if self.index is None or not self.chunk_ids:
            raise ValueError("FAISS index is not initialized.")

        query_vec = self.encode([query], batch_size=1, normalize=True)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results

    def save_index(self, index_path: Union[str, Path]) -> None:
        """Saves FAISS index to disk."""
        if faiss is None:
            raise ImportError("faiss is required. Install it with: pip install faiss-cpu")
        if self.index is None:
            raise ValueError("No index to save.")
        faiss.write_index(self.index, str(index_path))

    @classmethod
    def load_index(
        cls,
        index_path: Union[str, Path],
        chunk_ids: List[str],
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
    ) -> "DenseRetriever":
        """Loads a FAISS index from disk and attaches chunk ID mappings."""
        if faiss is None:
            raise ImportError("faiss is required. Install it with: pip install faiss-cpu")
        index = faiss.read_index(str(index_path))
        return cls(model_name=model_name, index=index, chunk_ids=chunk_ids, device=device)

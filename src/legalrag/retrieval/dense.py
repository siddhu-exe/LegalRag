"""
Dense Vector Retriever using BAAI/bge-base-en-v1.5 and FAISS IndexFlatIP.

Artifact contract
-----------------
``dense.index`` is a FAISS ``IndexFlatIP`` built over L2-normalized embeddings in the
exact row order of ``legal_chunks.parquet``. Row ``i`` of the index therefore corresponds
to ``chunk_ids[i]``. ``load_index()`` enforces ``index.ntotal == len(chunk_ids)`` so a
mismatched index/metadata pair fails closed at load time instead of silently mapping
FAISS row indices onto the wrong chunks.

Query-side contract
-------------------
BGE models are trained for asymmetric retrieval: relevant *queries* must be prefixed with
the retrieval instruction, while indexed *documents* are embedded without it. The frozen
production ``dense.index`` and the final experiment encoded queries with this instruction;
``search()`` therefore applies it to the query only (``encode()`` remains instruction-free
so document/index embeddings are unchanged).
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


#: BGE retrieval query instruction (must match the prefix used when the index experiment
#: encoded queries). Documents/embeddings of the corpus are never prefixed.
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


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
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ):
        self.model_name = model_name
        self.device = device
        self._model: Optional[Any] = None
        self.index = index
        self.chunk_ids = chunk_ids or []
        self.query_instruction = query_instruction

    def _format_query(self, query: str) -> str:
        """
        Applies the BGE retrieval instruction to a query only.

        Idempotent: a query that already carries the instruction is left untouched, and an
        empty ``query_instruction`` disables prefixing entirely.
        """
        instruction = self.query_instruction or ""
        if not instruction:
            return query
        prefix = instruction if instruction.endswith(" ") else instruction + " "
        if query.startswith(prefix):
            return query
        return prefix + query

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

    def validate(self) -> None:
        """
        Verifies the loaded FAISS index is usable and consistent with the chunk mapping.

        Raises:
            ValueError: if the index is missing, has no chunk IDs, reports an invalid
                dimensionality, or contains a different number of vectors than there are
                mapped chunk IDs.
        """
        if self.index is None:
            raise ValueError("FAISS index is not loaded.")
        if not self.chunk_ids:
            raise ValueError("DenseRetriever has no chunk_ids loaded.")
        ntotal = int(getattr(self.index, "ntotal", 0))
        if ntotal != len(self.chunk_ids):
            raise ValueError(
                f"FAISS index contains {ntotal} vectors but {len(self.chunk_ids)} chunk_ids "
                "are mapped; dense.index and legal_chunks.parquet are inconsistent."
            )
        if int(getattr(self.index, "d", 0)) <= 0:
            raise ValueError("FAISS index reports an invalid dimensionality.")

    def warmup(self) -> None:
        """
        Loads the embedding model and proves it is compatible with the loaded index.

        Raises:
            ValueError: if the embedding model's dimensionality does not match the FAISS
                index dimension (e.g. an index built with a different embedding model).
        """
        self.validate()
        index_dim = int(getattr(self.index, "d", 0))
        model_dim = getattr(self.model, "get_sentence_embedding_dimension", lambda: None)()
        if model_dim is not None and int(model_dim) != index_dim:
            raise ValueError(
                f"Embedding model '{self.model_name}' produces {int(model_dim)}-dim vectors "
                f"but dense.index was built with dimension {index_dim}; the dense index was "
                "likely generated with a different embedding model."
            )
        embeddings = self.encode(["warmup"], batch_size=1, normalize=True)
        if embeddings.ndim != 2 or int(embeddings.shape[1]) != index_dim:
            raise ValueError(
                f"Embedding model '{self.model_name}' emitted vectors of shape "
                f"{tuple(embeddings.shape)} which do not match the FAISS index dimension "
                f"{index_dim}."
            )

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

        query_vec = self.encode([self._format_query(query)], batch_size=1, normalize=True)
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
        """
        Loads a FAISS index from disk and attaches chunk ID mappings.

        Raises:
            ValueError: if the index vector count does not match ``len(chunk_ids)``.
        """
        if faiss is None:
            raise ImportError("faiss is required. Install it with: pip install faiss-cpu")
        index = faiss.read_index(str(index_path))
        ntotal = int(getattr(index, "ntotal", 0))
        if ntotal != len(chunk_ids):
            raise ValueError(
                f"FAISS index '{index_path}' contains {ntotal} vectors but {len(chunk_ids)} "
                "chunk_ids were supplied; dense.index and legal_chunks.parquet are inconsistent."
            )
        return cls(model_name=model_name, index=index, chunk_ids=chunk_ids, device=device)

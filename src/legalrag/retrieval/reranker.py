"""
Neural Cross-Encoder Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2.
"""

from typing import List, Tuple, Optional, Any

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore


class CrossEncoderReranker:
    """
    Reranks candidate (query, passage) pairs using a transformer Cross-Encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self._model: Optional[Any] = None

    @property
    def model(self) -> Any:
        """Lazy load CrossEncoder model."""
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            )
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
        return self._model

    def validate(self) -> None:
        """
        Verifies the reranker is configurable, raising if it could never be initialized.

        Raises:
            ImportError: if sentence-transformers is not installed.
            ValueError: if no model name is configured.
        """
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            )
        if not self.model_name:
            raise ValueError("CrossEncoderReranker requires a non-empty model_name.")

    def warmup(self) -> None:
        """
        Loads the CrossEncoder weights and runs a trivial prediction to prove usability.

        This surfaces a missing/undownloadable reranker model at initialization time
        (readiness -> 503) instead of failing the first user query.
        """
        self.validate()
        self.model.predict([["warmup query", "warmup passage"]], show_progress_bar=False)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """
        Reranks a list of candidate passages for a query.

        Args:
            query: Search query or legal question.
            candidates: List of (chunk_id, chunk_text) tuples.
            top_k: Number of highest scoring passages to return.
            batch_size: Batch size for model inference.

        Returns:
            List of (chunk_id, cross_encoder_score) sorted descending by score.
        """
        if not candidates:
            return []

        chunk_ids = [c[0] for c in candidates]
        pairs = [[query, c[1]] for c in candidates]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(chunk_ids[i], float(scores[i])) for i in ranked_indices]

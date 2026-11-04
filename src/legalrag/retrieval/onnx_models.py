"""
ONNX Runtime embedder and reranker, with no PyTorch (plan task C4, behind a flag).

Serving the v1 models needs ``torch`` and ``sentence-transformers`` (the PyTorch and FAISS imports
alone were 0.37 GiB in ``results/ram_baseline.json``, plus 0.46 GiB for the embedding model and
0.09 GiB for the reranker). These classes run exported ONNX files with ``onnxruntime`` and the
``tokenizers`` library only. Nothing in this module imports ``torch`` or ``transformers``.

Model directory (written by ``legalrag.retrieval.onnx_export`` / ``scripts/export_onnx.py``)::

    model.onnx         inputs input_ids, attention_mask [, token_type_ids]
    tokenizer.json     a Hugging Face fast tokenizer
    onnx_config.json   {"kind": "embedder" | "reranker", "max_length": 512, "dim": 768,
                        "pooling": "cls" | "mean", "normalize": true,         (embedder)
                        "output": "<output name>", "activation": "sigmoid" | "none"}   (reranker)

Both classes mirror the objects they replace: ``OnnxEmbedder.encode`` follows
``SentenceTransformer.encode`` (used by ``DenseRetriever``) and ``OnnxReranker.rerank`` follows
``CrossEncoderReranker.rerank``. Whether the ONNX models match the PyTorch ones closely enough is
checked at export time (``verify_*`` in the export module) and must be checked again on the real
models by plan task P5; nothing here claims a measured speed or memory gain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import onnxruntime
except ImportError:  # the default backends work without it
    onnxruntime = None  # type: ignore

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # type: ignore

MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "onnx_config.json"


class _OnnxModel:
    """Shared loading, tokenizing and running for the embedder and the reranker."""

    kind = ""

    def __init__(
        self,
        directory: Union[str, Path],
        intra_op_threads: Optional[int] = None,
        session: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.directory = Path(directory)
        self._intra_op_threads = intra_op_threads
        self._session = session
        self._tokenizer = tokenizer
        self.config = self._read_config()

    # ------------------------------------------------------------------ files and config

    def _read_config(self) -> Dict[str, Any]:
        path = self.directory / CONFIG_FILE
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def validate(self) -> None:
        """Checks the directory is complete and the right kind, without loading the model."""
        if self._session is not None and self._tokenizer is not None:
            return
        missing = [n for n in (MODEL_FILE, TOKENIZER_FILE, CONFIG_FILE) if not (self.directory / n).is_file()]
        if missing:
            raise ValueError(f"ONNX model directory {self.directory} is missing: {', '.join(missing)}.")
        if self.config.get("kind") != self.kind:
            raise ValueError(
                f"{self.directory} holds a {self.config.get('kind')!r} model but a {self.kind!r} was expected."
            )
        if onnxruntime is None:
            raise ImportError("onnxruntime is required for the ONNX backend. Install it with: pip install onnxruntime")
        if Tokenizer is None:
            raise ImportError("tokenizers is required for the ONNX backend. Install it with: pip install tokenizers")

    @property
    def max_length(self) -> int:
        return int(self.config.get("max_length", 512))

    @property
    def session(self) -> Any:
        if self._session is None:
            self.validate()
            options = onnxruntime.SessionOptions()
            if self._intra_op_threads:
                options.intra_op_num_threads = int(self._intra_op_threads)
            self._session = onnxruntime.InferenceSession(
                str(self.directory / MODEL_FILE), options, providers=["CPUExecutionProvider"]
            )
        return self._session

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self.validate()
            tokenizer = Tokenizer.from_file(str(self.directory / TOKENIZER_FILE))
            tokenizer.enable_truncation(max_length=self.max_length)
            pad_id = tokenizer.token_to_id(self.config.get("pad_token", "[PAD]"))
            tokenizer.enable_padding(pad_id=pad_id or 0, pad_token=self.config.get("pad_token", "[PAD]"))
            self._tokenizer = tokenizer
        return self._tokenizer

    # ------------------------------------------------------------------ running

    def _run(self, inputs: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenizes a batch (texts, or (text, text) pairs); returns the model output and attention mask."""
        encodings = self.tokenizer.encode_batch(list(inputs))
        feeds: Dict[str, np.ndarray] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        wanted = [node.name for node in self.session.get_inputs()]
        unknown = [name for name in wanted if name not in feeds]
        if unknown:
            raise ValueError(f"ONNX model expects inputs this runtime cannot supply: {unknown}")
        output_name = self.config.get("output")
        outputs = self.session.run([output_name] if output_name else None, {n: feeds[n] for n in wanted})
        return np.asarray(outputs[0]), feeds["attention_mask"]


class OnnxEmbedder(_OnnxModel):
    """Dense query/passage encoder. Same call shape as ``SentenceTransformer.encode``."""

    kind = "embedder"

    @classmethod
    def load(cls, directory: Union[str, Path], intra_op_threads: Optional[int] = None) -> "OnnxEmbedder":
        embedder = cls(directory, intra_op_threads)
        embedder.validate()
        return embedder

    def get_sentence_embedding_dimension(self) -> Optional[int]:
        dim = self.config.get("dim")
        return int(dim) if dim else None

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,  # accepted for SentenceTransformer compatibility; ignored
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,  # accepted for compatibility; always numpy
    ) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.get_sentence_embedding_dimension() or 0), dtype=np.float32)
        order = sorted(range(len(items)), key=lambda i: len(items[i]))  # similar lengths share a batch
        pooling = self.config.get("pooling", "cls")
        if pooling not in ("cls", "mean"):
            raise ValueError(f"Unknown pooling {pooling!r}; expected 'cls' or 'mean'.")

        result: List[Optional[np.ndarray]] = [None] * len(items)
        for start in range(0, len(order), max(1, batch_size)):
            positions = order[start : start + max(1, batch_size)]
            hidden, mask = self._run([items[i] for i in positions])
            if hidden.ndim == 2:  # the exported model already pooled
                vectors = hidden
            elif pooling == "cls":
                vectors = hidden[:, 0, :]
            else:
                weights = mask[..., None].astype(np.float32)
                vectors = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
            vectors = vectors.astype(np.float32)
            if normalize_embeddings:
                vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            for row, position in enumerate(positions):
                result[position] = vectors[row]
        stacked = np.stack(result).astype(np.float32)  # type: ignore[arg-type]
        return stacked[0] if single else stacked

    def warmup(self) -> None:
        """Loads the model and proves its output width matches the declared dimension."""
        vector = self.encode(["warmup"], batch_size=1)
        declared = self.get_sentence_embedding_dimension()
        if vector.ndim != 2 or (declared and vector.shape[1] != declared):
            raise ValueError(f"ONNX embedder produced shape {vector.shape}, declared dimension is {declared}.")


class OnnxReranker(_OnnxModel):
    """Cross-encoder reranker. Same ``rerank`` signature as ``CrossEncoderReranker``."""

    kind = "reranker"

    def __init__(self, directory: Union[str, Path], model_name: str = "", **kwargs: Any):
        super().__init__(directory, **kwargs)
        self.model_name = model_name or str(directory)

    @classmethod
    def load(cls, directory: Union[str, Path], intra_op_threads: Optional[int] = None) -> "OnnxReranker":
        reranker = cls(directory, intra_op_threads=intra_op_threads)
        reranker.validate()
        return reranker

    def _activate(self, logits: np.ndarray) -> np.ndarray:
        if self.config.get("activation", "sigmoid") == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        return logits

    def predict(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        scores: List[float] = []
        for start in range(0, len(pairs), max(1, batch_size)):
            logits, _ = self._run([tuple(p) for p in pairs[start : start + max(1, batch_size)]])
            logits = logits.reshape(logits.shape[0], -1)
            if logits.shape[1] != 1:
                raise ValueError(f"Reranker output has {logits.shape[1]} columns; expected one score per pair.")
            scores.extend(self._activate(logits[:, 0]).astype(np.float32).tolist())
        return np.array(scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """Scores (query, passage) pairs; best first, ties broken by candidate order."""
        if not candidates:
            return []
        scores = self.predict([(query, text) for _, text in candidates], batch_size)
        if not np.isfinite(scores).all():
            raise ValueError("Reranker produced non-finite scores.")
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[: max(0, top_k)]
        return [(candidates[i][0], float(scores[i])) for i in ranked]

    def warmup(self) -> None:
        self.predict([("warmup query", "warmup passage")], batch_size=1)

"""
Approximate FAISS index builders for the v2 corpus (plan task C4, behind a flag).

A flat ``IndexFlatIP`` over the planned 5.7 million chunks would need about 17.5 GB (estimate in
docs/DATA_PROFILE_BOMBAY.md), so v2 needs a smaller index. Two builders are provided; which one to
use, and how much recall each loses on this corpus, is plan task P5 and has not been measured:

``ivf_sq8``  inverted file with 8-bit scalar quantization (about 1 byte per dimension per vector).
``opq_pq``   an OPQ rotation, an inverted file and product quantization (about ``pq_m`` bytes per
             vector), the smallest option and the one with the largest expected recall loss.

Both use inner product over L2-normalized embeddings, like the v1 flat index, and write a small
``<index>.json`` beside the index recording how it was built. ``recall_against_exact`` measures an
index against exact search so P5 can compare settings.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

try:
    import faiss
except ImportError:  # the flat index and the rest of the package work without it
    faiss = None  # type: ignore

logger = logging.getLogger(__name__)

INDEX_KINDS = ("ivf_sq8", "opq_pq")
FORMAT_VERSION = 1
FAISS_DEFAULT_MIN_TRAIN_PER_CENTROID = 39  # FAISS warns below this many training points per centroid


def _require_faiss() -> Any:
    if faiss is None:
        raise ImportError("faiss is required. Install it with: pip install faiss-cpu")
    return faiss


def suggest_nlist(n_vectors: int) -> int:
    """A starting point for the number of inverted lists: about 4 * sqrt(N), the low end of FAISS's guidance.

    Not tuned for this corpus. Plan task P5 should compare values.
    """
    if n_vectors <= 0:
        raise ValueError("n_vectors must be positive")
    return max(1, int(round(4 * math.sqrt(n_vectors))))


class EmbeddingShards:
    """Float32 embeddings stored as one or more ``.npy`` files of shape (rows, dim), in chunk order.

    Files are opened with ``mmap`` so the whole corpus never has to fit in memory.
    """

    def __init__(self, paths: Sequence[Union[str, Path]]):
        if not paths:
            raise ValueError("No embedding files given.")
        self.paths = [Path(p) for p in paths]
        self._arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in self.paths]
        dims = {a.shape[1] if a.ndim == 2 else None for a in self._arrays}
        if None in dims or len(dims) != 1:
            raise ValueError(f"Embedding files must all be 2-D with the same width; found widths {sorted(map(str, dims))}.")
        for path, array in zip(self.paths, self._arrays):
            if array.dtype not in (np.float32, np.float16):
                raise ValueError(f"{path.name}: expected float32 or float16 embeddings, got {array.dtype}.")
        self.dim = int(next(iter(dims)))
        self.sizes = [int(a.shape[0]) for a in self._arrays]
        self.total = sum(self.sizes)
        if self.total == 0:
            raise ValueError("The embedding files contain no rows.")

    def batches(self, batch_rows: int = 100_000) -> Iterator[np.ndarray]:
        for array in self._arrays:
            for start in range(0, array.shape[0], batch_rows):
                yield self._clean(array[start : start + batch_rows])

    def sample(self, count: int, seed: int = 42) -> np.ndarray:
        """``count`` rows chosen uniformly at random (reproducible), in corpus order."""
        count = min(count, self.total)
        rows = np.sort(np.random.default_rng(seed).choice(self.total, size=count, replace=False))
        offsets = np.cumsum([0] + self.sizes)
        parts = []
        for shard, array in enumerate(self._arrays):
            inside = rows[(rows >= offsets[shard]) & (rows < offsets[shard + 1])] - offsets[shard]
            if len(inside):
                parts.append(self._clean(array[inside]))
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _clean(block: np.ndarray) -> np.ndarray:
        block = np.ascontiguousarray(block, dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Embeddings contain NaN or infinite values.")
        return block


@dataclass
class IndexSpec:
    """How to build an approximate index. ``factory`` is the FAISS index_factory string."""

    kind: str
    dim: int
    nlist: int
    pq_m: Optional[int] = None
    pq_bits: int = 8

    def __post_init__(self) -> None:
        if self.kind not in INDEX_KINDS:
            raise ValueError(f"kind must be one of {INDEX_KINDS}, got {self.kind!r}")
        if self.nlist < 1:
            raise ValueError("nlist must be at least 1")
        if self.kind == "opq_pq":
            if not self.pq_m:
                raise ValueError("opq_pq needs pq_m (the number of PQ sub-vectors; it must divide the dimension)")
            if self.dim % self.pq_m:
                raise ValueError(f"pq_m ({self.pq_m}) must divide the embedding dimension ({self.dim})")
            if not 1 <= self.pq_bits <= 16:
                raise ValueError("pq_bits must be between 1 and 16")

    @property
    def factory(self) -> str:
        if self.kind == "ivf_sq8":
            return f"IVF{self.nlist},SQ8"
        return f"OPQ{self.pq_m},IVF{self.nlist},PQ{self.pq_m}x{self.pq_bits}"

    @property
    def min_training_points(self) -> int:
        """The least FAISS can train on: one point per list, and one per PQ centroid."""
        needed = self.nlist
        if self.kind == "opq_pq":
            needed = max(needed, 2**self.pq_bits)
        return needed


def build_index(
    shards: EmbeddingShards,
    spec: IndexSpec,
    train_size: Optional[int] = None,
    seed: int = 42,
    add_batch_rows: int = 100_000,
    nprobe: int = 16,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Trains on a random sample, adds every vector in order, and returns ``(index, metadata)``.

    ``train_size`` defaults to 39 points per list (FAISS's own recommendation), capped at the corpus.
    Row ``i`` of the index is row ``i`` of the shards, so it lines up with the chunk id order.
    """
    lib = _require_faiss()
    if spec.dim != shards.dim:
        raise ValueError(f"Spec dimension {spec.dim} does not match the embeddings ({shards.dim}).")
    if spec.nlist > shards.total:
        raise ValueError(f"nlist ({spec.nlist}) is larger than the number of vectors ({shards.total}).")
    if train_size is None:
        train_size = FAISS_DEFAULT_MIN_TRAIN_PER_CENTROID * spec.nlist
    train_size = min(train_size, shards.total)
    if train_size < spec.min_training_points:
        raise ValueError(
            f"Training sample of {train_size} is too small for {spec.factory} "
            f"(needs at least {spec.min_training_points})."
        )
    if train_size < FAISS_DEFAULT_MIN_TRAIN_PER_CENTROID * spec.nlist:
        logger.warning(
            "Training on %d points for %d lists is below FAISS's recommended %d; recall may suffer.",
            train_size, spec.nlist, FAISS_DEFAULT_MIN_TRAIN_PER_CENTROID * spec.nlist,
        )

    index = lib.index_factory(spec.dim, spec.factory, lib.METRIC_INNER_PRODUCT)
    index.train(shards.sample(train_size, seed))
    for block in shards.batches(add_batch_rows):
        index.add(block)
    if index.ntotal != shards.total:
        raise RuntimeError(f"Index holds {index.ntotal} vectors but {shards.total} were added.")
    set_nprobe(index, nprobe)

    metadata = {
        "format": FORMAT_VERSION,
        "kind": spec.kind,
        "factory": spec.factory,
        "metric": "inner_product",
        "dim": spec.dim,
        "ntotal": int(index.ntotal),
        "nlist": spec.nlist,
        "pq_m": spec.pq_m,
        "pq_bits": spec.pq_bits if spec.kind == "opq_pq" else None,
        "nprobe_default": nprobe,
        "train_size": int(train_size),
        "seed": seed,
        "source_files": [{"name": p.name, "rows": n} for p, n in zip(shards.paths, shards.sizes)],
        "faiss_version": getattr(lib, "__version__", "unknown"),
        **(extra_metadata or {}),
    }
    return index, metadata


def metadata_path(index_path: Union[str, Path]) -> Path:
    index_path = Path(index_path)
    return index_path.with_name(index_path.name + ".json")


def write_index(index: Any, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Writes the index (to a temporary name, then renamed) and its metadata file beside it."""
    lib = _require_faiss()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    lib.write_index(index, str(temporary))
    temporary.replace(path)
    if metadata is not None:
        metadata_path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def set_nprobe(index: Any, nprobe: int) -> bool:
    """Sets how many inverted lists a search visits. Returns False for an index with no lists (flat)."""
    lib = _require_faiss()
    if nprobe < 1:
        raise ValueError("nprobe must be at least 1")
    try:
        ivf = lib.extract_index_ivf(index)
    except Exception:  # noqa: BLE001 - FAISS raises a generic RuntimeError for non-IVF indexes
        return False
    ivf.nprobe = int(min(nprobe, ivf.nlist))
    return True


def load_index(path: Union[str, Path], nprobe: Optional[int] = None, mmap: bool = False) -> Any:
    """Reads any FAISS index file. ``mmap`` maps the file instead of reading it into memory."""
    lib = _require_faiss()
    flag = getattr(lib, "IO_FLAG_MMAP", None) if mmap else None
    index = lib.read_index(str(path), flag) if flag else lib.read_index(str(path))
    if nprobe is not None:
        set_nprobe(index, nprobe)
    return index


def recall_against_exact(
    index: Any,
    exact_vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    nprobe_values: Optional[Sequence[int]] = None,
) -> List[Dict[str, float]]:
    """Share of the exact top-k that the approximate index also returns, for each nprobe.

    ``exact_vectors`` must be the vectors the index holds, in the same order (use a sample for a
    large corpus and build a small index from it). This is the comparison plan task P5 needs; it
    measures the index, not retrieval quality on the benchmark.
    """
    lib = _require_faiss()
    flat = lib.IndexFlatIP(exact_vectors.shape[1])
    flat.add(np.ascontiguousarray(exact_vectors, dtype=np.float32))
    _, truth = flat.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    results = []
    for nprobe in nprobe_values or [None]:
        if nprobe is not None:
            set_nprobe(index, nprobe)
        _, found = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        hits = [len(set(t.tolist()) & set(f.tolist()) - {-1}) / k for t, f in zip(truth, found)]
        results.append({"nprobe": float(nprobe) if nprobe is not None else float("nan"), "recall_at_k": float(np.mean(hits)), "k": k})
    return results

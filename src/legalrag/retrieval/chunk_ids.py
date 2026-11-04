"""
Chunk ids stored as a memory-mapped array instead of a Python list.

A list of millions of id strings costs hundreds of megabytes of resident memory (v1: 0.044 GiB
for 538,079 ids, so roughly 0.5 GiB at the planned 5.7 million). ``ChunkIdArray`` keeps the ids in
one fixed-width ``.npy`` file that can be opened with ``mmap``, and behaves like a read-only list
of ``str`` (``len``, indexing, slicing, iteration, ``in`` via iteration, truthiness), so the
retrievers and the pipeline checks that expect ``chunk_ids`` keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Union, overload

import numpy as np


class ChunkIdArray(Sequence):
    """A read-only sequence of chunk ids backed by a numpy array (optionally memory-mapped)."""

    def __init__(self, array: np.ndarray):
        if array.ndim != 1:
            raise ValueError("chunk ids must be a one-dimensional array")
        self._array = array

    @classmethod
    def from_ids(cls, ids: Iterable[str]) -> "ChunkIdArray":
        values = [str(i) for i in ids]
        if any(not v for v in values):
            raise ValueError("chunk ids must not be empty strings")
        return cls(np.array(values, dtype=np.str_) if values else np.array([], dtype="U1"))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._array, allow_pickle=False)

    @classmethod
    def load(cls, path: Union[str, Path], mmap: bool = True) -> "ChunkIdArray":
        return cls(np.load(Path(path), mmap_mode="r" if mmap else None, allow_pickle=False))

    def __len__(self) -> int:
        return int(self._array.shape[0])

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> List[str]: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [str(v) for v in self._array[index]]
        return str(self._array[index])

    def __iter__(self) -> Iterator[str]:
        for start in range(0, len(self), 65536):  # bounded chunks keep an mmap'd file from being fully read at once
            for value in self._array[start : start + 65536]:
                yield str(value)

    def take(self, positions: Sequence[int]) -> List[str]:
        """Ids at the given row positions (the usual way to turn a search result into chunk ids)."""
        return [str(v) for v in self._array[np.asarray(positions, dtype=np.int64)]]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ChunkIdArray):
            return bool(np.array_equal(self._array, other._array))
        if isinstance(other, (list, tuple)):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

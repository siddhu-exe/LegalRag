"""
SQLite chunk store (plan task C4, behind a flag).

v1 keeps every chunk's text in a pandas frame: 1.46 GiB in ``results/ram_baseline.json`` for 538,079
chunks, which would be roughly 15 GiB for the planned 5.7 million. A query only needs the text of
the few dozen chunks it retrieves, so the store keeps text on disk and fetches those rows by id.

Layout: one table, ``chunks(position, chunk_id, cnr, text, court_code, decision_date, title)``.
``position`` is the row number in index order, so row ``i`` of the BM25 and FAISS indexes is
``position = i``. Chunk ids themselves are also written to ``chunk_ids.npy`` beside the database
(``ChunkIdArray``, memory-mappable) so the retrievers do not hold ids in memory.

Read connections are opened per thread and read-only, because the API serves requests from a thread
pool. The store never writes after ``build``.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from legalrag.retrieval.chunk_ids import ChunkIdArray

STORE_FILE = "chunks.sqlite"
CHUNK_IDS_FILE = "chunk_ids.npy"
FIELDS = ("cnr", "text", "court_code", "decision_date", "title")
_SQL_VARIABLE_BATCH = 500  # well under SQLite's per-statement limit


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if value != value:  # NaN from pandas
            return None
    except Exception:  # noqa: BLE001 - arrays and other odd values: fall through to str()
        pass
    return str(value)


class SqliteChunkStore:
    """Read-only lookup of chunk rows by chunk id."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Chunk store not found: {self.path}")
        self._local = threading.local()
        with self._connection() as connection:
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "chunks" not in tables:
            raise ValueError(f"{self.path} is not a chunk store (no 'chunks' table).")

    # ------------------------------------------------------------------ connections

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
            connection.execute("PRAGMA query_only = ON")
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def __enter__(self) -> "SqliteChunkStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ reads

    def __len__(self) -> int:
        return int(self._connection().execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def get_many(self, chunk_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Rows for the ids that exist, keyed by chunk id. Unknown ids are simply absent."""
        wanted = list(dict.fromkeys(str(c) for c in chunk_ids))
        found: Dict[str, Dict[str, Any]] = {}
        connection = self._connection()
        for start in range(0, len(wanted), _SQL_VARIABLE_BATCH):
            batch = wanted[start : start + _SQL_VARIABLE_BATCH]
            marks = ",".join("?" * len(batch))
            query = f"SELECT chunk_id, {', '.join(FIELDS)} FROM chunks WHERE chunk_id IN ({marks})"
            for row in connection.execute(query, batch):
                found[row[0]] = dict(zip(FIELDS, row[1:]))
        return found

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.get_many([chunk_id]).get(str(chunk_id))

    def id_at(self, position: int) -> Optional[str]:
        row = self._connection().execute("SELECT chunk_id FROM chunks WHERE position = ?", (int(position),)).fetchone()
        return row[0] if row else None

    def positions(self, chunk_ids: Sequence[str]) -> Dict[str, int]:
        found: Dict[str, int] = {}
        connection = self._connection()
        wanted = list(dict.fromkeys(str(c) for c in chunk_ids))
        for start in range(0, len(wanted), _SQL_VARIABLE_BATCH):
            batch = wanted[start : start + _SQL_VARIABLE_BATCH]
            marks = ",".join("?" * len(batch))
            for chunk_id, position in connection.execute(
                f"SELECT chunk_id, position FROM chunks WHERE chunk_id IN ({marks})", batch
            ):
                found[chunk_id] = position
        return found

    def load_chunk_ids(self, mmap: bool = True) -> ChunkIdArray:
        """The ids in index order, from ``chunk_ids.npy`` beside the database."""
        path = self.path.with_name(CHUNK_IDS_FILE)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; rebuild the store with SqliteChunkStore.build.")
        return ChunkIdArray.load(path, mmap=mmap)

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        path: Union[str, Path],
        rows: Iterable[Mapping[str, Any]],
        batch_rows: int = 20_000,
    ) -> "SqliteChunkStore":
        """Writes a new store from an iterable of row mappings (needs ``chunk_id``; other fields optional).

        Positions follow the order of ``rows``. Duplicate or empty ids raise ``ValueError``, and a
        failed build leaves no partial database behind. Also writes ``chunk_ids.npy`` beside it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".building")
        ids_path = path.with_name(CHUNK_IDS_FILE)
        for stale in (temporary, Path(str(temporary) + "-journal")):
            stale.unlink(missing_ok=True)

        ids: List[str] = []
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute(
                "CREATE TABLE chunks (position INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, "
                "cnr TEXT, text TEXT, court_code TEXT, decision_date TEXT, title TEXT)"
            )
            insert = "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)"
            batch: List[tuple] = []
            for row in rows:
                chunk_id = _clean(row.get("chunk_id"))
                if not chunk_id:
                    raise ValueError(f"Row {len(ids)} has no chunk_id.")
                batch.append((len(ids), chunk_id, *(_clean(row.get(f)) for f in FIELDS)))
                ids.append(chunk_id)
                if len(batch) >= batch_rows:
                    connection.executemany(insert, batch)
                    batch = []
            if batch:
                connection.executemany(insert, batch)
            if not ids:
                raise ValueError("Cannot build a chunk store with no rows.")
            connection.execute("CREATE INDEX chunks_cnr ON chunks (cnr)")
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise ValueError("Duplicate chunk_id in the input; every chunk id must be unique.") from exc
        except BaseException:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        connection.close()
        ChunkIdArray.from_ids(ids).save(ids_path)
        temporary.replace(path)
        return cls(path)

    @classmethod
    def build_from_parquet(
        cls, parquet_path: Union[str, Path], store_path: Union[str, Path], batch_rows: int = 20_000
    ) -> "SqliteChunkStore":
        """Streams a chunk parquet (such as v1's ``legal_chunks.parquet``) into a new store."""
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(str(parquet_path))
        names = set(parquet.schema_arrow.names)
        if "chunk_id" not in names:
            raise ValueError(f"{parquet_path} has no 'chunk_id' column; columns found: {', '.join(sorted(names))}.")
        columns = ["chunk_id"] + [f for f in FIELDS if f in names]

        def rows() -> Iterator[Dict[str, Any]]:
            for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
                yield from batch.to_pylist()

        return cls.build(store_path, rows(), batch_rows)


class DataFrameChunkSource:
    """The v1 in-memory chunk table behind the same ``get_many`` call as the SQLite store."""

    def __init__(self, frame: Any):
        self.frame = frame

    def __len__(self) -> int:
        return int(len(self.frame))

    def get_many(self, chunk_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        frame = self.frame
        if frame is None or frame.empty or "chunk_id" not in frame.columns:
            return {}
        subset = frame[frame["chunk_id"].isin(list(chunk_ids))]
        return subset.set_index("chunk_id").to_dict(orient="index")

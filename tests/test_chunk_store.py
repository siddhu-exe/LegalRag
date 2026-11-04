"""Tests for the SQLite chunk store (plan task C4)."""

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

import pandas as pd

from legalrag.retrieval.chunk_ids import ChunkIdArray
from legalrag.retrieval.chunk_store import CHUNK_IDS_FILE, DataFrameChunkSource, SqliteChunkStore

ROWS = [
    {"chunk_id": f"CNR{i // 3}_chunk_{i % 3}", "cnr": f"CNR{i // 3}", "text": f"text of chunk {i} धारा",
     "court_code": "27_1", "decision_date": f"2020-01-{i % 28 + 1:02d}", "title": f"Case {i // 3}"}
    for i in range(30)
]


def build(tmp, rows=ROWS):
    return SqliteChunkStore.build(Path(tmp) / "chunks" / "chunks.sqlite", iter(rows), batch_rows=7)


class TestSqliteChunkStore(unittest.TestCase):
    def test_build_and_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = build(tmp)
            self.assertEqual(len(store), 30)
            row = store.get("CNR2_chunk_1")
            self.assertEqual(row["cnr"], "CNR2")
            self.assertEqual(row["text"], "text of chunk 7 धारा")
            self.assertEqual(set(row), {"cnr", "text", "court_code", "decision_date", "title"})

    def test_get_many_returns_only_existing_ids_and_ignores_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = build(tmp).get_many(["CNR0_chunk_0", "missing", "CNR0_chunk_0", "CNR9_chunk_2"])
            self.assertEqual(set(found), {"CNR0_chunk_0", "CNR9_chunk_2"})

    def test_large_lookup_is_batched_under_sqlite_limits(self):
        rows = [{"chunk_id": f"id{i}", "text": str(i)} for i in range(2500)]
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteChunkStore.build(Path(tmp) / "s.sqlite", iter(rows))
            found = store.get_many([f"id{i}" for i in range(2500)])
            self.assertEqual(len(found), 2500)
            self.assertEqual(found["id2499"]["text"], "2499")

    def test_positions_follow_input_order_and_ids_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = build(tmp)
            self.assertEqual(store.id_at(0), "CNR0_chunk_0")
            self.assertEqual(store.id_at(29), "CNR9_chunk_2")
            self.assertIsNone(store.id_at(30))
            self.assertEqual(store.positions(["CNR1_chunk_0", "nope"]), {"CNR1_chunk_0": 3})
            ids = store.load_chunk_ids()
            self.assertIsInstance(ids, ChunkIdArray)
            self.assertEqual(list(ids), [r["chunk_id"] for r in ROWS])
            self.assertTrue((store.path.with_name(CHUNK_IDS_FILE)).is_file())

    def test_missing_optional_fields_become_none_and_nan_is_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteChunkStore.build(Path(tmp) / "s.sqlite", iter([{"chunk_id": "a", "text": "t", "title": float("nan")}]))
            row = store.get("a")
            self.assertEqual((row["text"], row["title"], row["cnr"]), ("t", None, None))

    def test_duplicate_or_empty_ids_and_empty_input_are_refused_without_leaving_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.sqlite"
            for rows, pattern in (
                ([{"chunk_id": "a"}, {"chunk_id": "a"}], "Duplicate"),
                ([{"chunk_id": "a"}, {"text": "no id"}], "no chunk_id"),
                ([], "no rows"),
            ):
                with self.subTest(pattern=pattern):
                    with self.assertRaisesRegex(ValueError, pattern):
                        SqliteChunkStore.build(path, iter(rows))
                    self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_opening_something_else_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                SqliteChunkStore(Path(tmp) / "nowhere.sqlite")
            other = Path(tmp) / "other.sqlite"
            sqlite3.connect(other).execute("CREATE TABLE x (a)").connection.commit()
            with self.assertRaisesRegex(ValueError, "not a chunk store"):
                SqliteChunkStore(other)

    def test_the_store_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = build(tmp)
            with self.assertRaises(sqlite3.OperationalError):
                store._connection().execute("DELETE FROM chunks")

    def test_usable_from_several_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = build(tmp)
            errors, results = [], []

            def worker(n):
                try:
                    results.append(store.get(f"CNR{n}_chunk_0")["cnr"])
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(n,)) for n in range(10)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            self.assertEqual((errors, sorted(results)), ([], sorted(f"CNR{n}" for n in range(10))))

    def test_build_from_parquet_streams_the_v1_chunk_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet = Path(tmp) / "legal_chunks.parquet"
            pd.DataFrame(ROWS).assign(chunk_index=range(30)).to_parquet(parquet, index=False)
            store = SqliteChunkStore.build_from_parquet(parquet, Path(tmp) / "out" / "chunks.sqlite", batch_rows=8)
            self.assertEqual(len(store), 30)
            self.assertEqual(store.get("CNR3_chunk_2")["title"], "Case 3")
            pd.DataFrame({"text": ["x"]}).to_parquet(Path(tmp) / "bad.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "no 'chunk_id'"):
                SqliteChunkStore.build_from_parquet(Path(tmp) / "bad.parquet", Path(tmp) / "bad.sqlite")

    def test_agrees_with_the_dataframe_source_the_v1_path_uses(self):
        frame = pd.DataFrame(ROWS)
        wanted = ["CNR1_chunk_2", "CNR5_chunk_0", "absent"]
        from_frame = DataFrameChunkSource(frame).get_many(wanted)
        with tempfile.TemporaryDirectory() as tmp:
            from_store = build(tmp).get_many(wanted)
        self.assertEqual(set(from_frame), set(from_store))
        for chunk_id in from_store:
            for field in ("cnr", "text", "court_code", "decision_date", "title"):
                self.assertEqual(from_frame[chunk_id][field], from_store[chunk_id][field])

    def test_dataframe_source_handles_empty_and_incomplete_frames(self):
        self.assertEqual(DataFrameChunkSource(pd.DataFrame()).get_many(["a"]), {})
        self.assertEqual(DataFrameChunkSource(pd.DataFrame({"text": ["x"]})).get_many(["a"]), {})
        self.assertEqual(len(DataFrameChunkSource(pd.DataFrame(ROWS))), 30)


if __name__ == "__main__":
    unittest.main()

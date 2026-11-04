"""Tests for scripts/build_retrieval_indexes_v2.py and scripts/export_onnx.py (plan task C4)."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from test_onnx_models import HAVE_EXPORT_STACK, make_tiny_model

from legalrag.retrieval.bm25s_backend import Bm25sRetriever, bm25s
from legalrag.retrieval.chunk_store import SqliteChunkStore
from legalrag.retrieval.faiss_builders import load_index, metadata_path


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_retrieval_indexes_v2")
exporter = load_script("export_onnx")

WORDS = ["bail", "section", "writ", "cheque", "arbitration", "notice", "appeal", "tax", "lease", "decree"]


def chunk_rows(n=60):
    return [
        {"chunk_id": f"C{i // 3}_chunk_{i % 3}", "cnr": f"C{i // 3}", "text": f"{WORDS[i % 10]} {WORDS[(i * 3) % 10]} judgment {i}",
         "court_code": "27_1", "decision_date": "2020-01-01", "title": f"Case {i // 3}"}
        for i in range(n)
    ]


def write_chunks(directory: Path, n=60) -> Path:
    path = directory / "chunks.parquet"
    pd.DataFrame(chunk_rows(n)).to_parquet(path, index=False)
    return path


def run(argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        builder.main(argv)
    return buffer.getvalue()


@unittest.skipIf(bm25s is None, "bm25s is not installed")
class TestBm25sCommand(unittest.TestCase):
    def test_builds_a_loadable_index_with_the_chunk_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "v2"
            output = run(["bm25s", "--chunks", str(write_chunks(tmp)), "--out-dir", str(out)])
            self.assertEqual(json.loads(output.strip().splitlines()[-1])["chunks"], 60)
            retriever = Bm25sRetriever.load(out / "bm25s")
            self.assertEqual(len(retriever.chunk_ids), 60)
            self.assertTrue(retriever.search("arbitration", 3))
            self.assertFalse((out / ".bm25s_staging").exists())

    def test_max_chunks_is_marked_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output = run(["bm25s", "--chunks", str(write_chunks(tmp)), "--out-dir", str(tmp / "v2"), "--max-chunks", "20"])
            report = json.loads(output.strip().splitlines()[-1])
            self.assertEqual((report["chunks"], report["partial"]), (20, True))

    def test_refuses_to_overwrite_and_refuses_a_directory_without_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            chunks = write_chunks(tmp)
            run(["bm25s", "--chunks", str(chunks), "--out-dir", str(tmp / "v2")])
            with self.assertRaisesRegex(SystemExit, "already exists"):
                run(["bm25s", "--chunks", str(chunks), "--out-dir", str(tmp / "v2")])
            run(["bm25s", "--chunks", str(chunks), "--out-dir", str(tmp / "v2"), "--overwrite", "--max-chunks", "10"])
            self.assertEqual(len(Bm25sRetriever.load(tmp / "v2" / "bm25s").chunk_ids), 10)
            with self.assertRaisesRegex(SystemExit, "'v2' part"):
                run(["bm25s", "--chunks", str(chunks), "--out-dir", str(tmp / "artifacts")])
            self.assertFalse((tmp / "artifacts").exists())

    def test_missing_column_is_reported_with_the_columns_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pd.DataFrame({"chunk_id": ["a"], "body": ["x"]}).to_parquet(tmp / "bad.parquet", index=False)
            with self.assertRaisesRegex(SystemExit, "lacks column.*text.*Columns found: chunk_id, body"):
                run(["bm25s", "--chunks", str(tmp / "bad.parquet"), "--out-dir", str(tmp / "v2")])


class TestChunksCommand(unittest.TestCase):
    def test_builds_the_store_and_the_id_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run(["chunks", "--chunks", str(write_chunks(tmp)), "--out-dir", str(tmp / "v2")])
            store = SqliteChunkStore(tmp / "v2" / "chunks" / "chunks.sqlite")
            self.assertEqual(len(store), 60)
            self.assertEqual(store.get("C4_chunk_1")["title"], "Case 4")
            self.assertEqual(len(store.load_chunk_ids()), 60)

    def test_works_when_optional_columns_are_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pd.DataFrame({"chunk_id": ["a", "b"], "text": ["x", "y"]}).to_parquet(tmp / "min.parquet", index=False)
            run(["chunks", "--chunks", str(tmp / "min.parquet"), "--out-dir", str(tmp / "v2")])
            self.assertEqual(SqliteChunkStore(tmp / "v2" / "chunks" / "chunks.sqlite").get("a")["text"], "x")

    def test_overwrite_and_v2_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            chunks = write_chunks(tmp)
            run(["chunks", "--chunks", str(chunks), "--out-dir", str(tmp / "v2")])
            with self.assertRaisesRegex(SystemExit, "already exists"):
                run(["chunks", "--chunks", str(chunks), "--out-dir", str(tmp / "v2")])
            with self.assertRaisesRegex(SystemExit, "'v2' part"):
                run(["chunks", "--chunks", str(chunks), "--out-dir", str(tmp / "data")])


def clustered(n, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(8, dim))
    x = (centers[rng.integers(0, 8, n)] + 0.3 * rng.normal(size=(n, dim))).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestFaissCommands(unittest.TestCase):
    def shards(self, tmp, n=600):
        x = clustered(n)
        directory = tmp / "emb"
        directory.mkdir()
        for i, block in enumerate(np.array_split(x, 2)):
            np.save(directory / f"part{i}.npy", block)
        return directory, x

    def test_builds_ivf_sq8_from_a_folder_of_shards_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            directory, x = self.shards(tmp)
            run(["faiss", "--embeddings", str(directory), "--kind", "ivf_sq8", "--nlist", "8", "--train-size", "600",
                 "--nprobe", "8", "--out-dir", str(tmp / "v2"), "--embedding-model", "test-model"])
            index = load_index(tmp / "v2" / "dense.index")
            self.assertEqual((index.ntotal, index.d), (600, 16))
            meta = json.loads(metadata_path(tmp / "v2" / "dense.index").read_text())
            self.assertEqual((meta["factory"], meta["embedding_model"], meta["nprobe_default"]), ("IVF8,SQ8", "test-model", 8))
            self.assertEqual(int(index.search(x[:1], 1)[1][0][0]), 0)

    def test_builds_opq_pq_from_a_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            directory, _ = self.shards(tmp, n=1200)
            run(["faiss", "--embeddings", str(directory / "*.npy"), "--kind", "opq_pq", "--nlist", "4", "--pq-m", "4",
                 "--pq-bits", "8", "--train-size", "1200", "--out-dir", str(tmp / "v2")])
            meta = json.loads(metadata_path(tmp / "v2" / "dense.index").read_text())
            self.assertEqual(meta["factory"], "OPQ4,IVF4,PQ4x8")

    def test_default_nlist_comes_from_the_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            directory, _ = self.shards(tmp, n=400)
            run(["faiss", "--embeddings", str(directory), "--kind", "ivf_sq8", "--out-dir", str(tmp / "v2"), "--train-size", "400"])
            self.assertEqual(json.loads(metadata_path(tmp / "v2" / "dense.index").read_text())["nlist"], 80)  # 4 * sqrt(400)

    def test_row_count_must_match_the_chunk_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            directory, _ = self.shards(tmp, n=600)
            store = SqliteChunkStore.build(tmp / "store" / "chunks.sqlite", chunk_rows(50))
            with self.assertRaisesRegex(SystemExit, "600 rows.*50"):
                run(["faiss", "--embeddings", str(directory), "--kind", "ivf_sq8", "--nlist", "4", "--train-size", "600",
                     "--out-dir", str(tmp / "v2"), "--chunk-store", str(store.path)])
            self.assertFalse((tmp / "v2" / "dense.index").exists())

    def test_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            directory, _ = self.shards(tmp)
            base = ["faiss", "--embeddings", str(directory), "--kind", "ivf_sq8", "--nlist", "4", "--train-size", "600"]
            with self.assertRaisesRegex(SystemExit, "'v2' part"):
                run(base + ["--out-dir", str(tmp / "out")])
            run(base + ["--out-dir", str(tmp / "v2")])
            with self.assertRaisesRegex(SystemExit, "already exists"):
                run(base + ["--out-dir", str(tmp / "v2")])
            with self.assertRaisesRegex(SystemExit, "No embedding files"):
                run(["faiss", "--embeddings", str(tmp / "nothing" / "*.npy"), "--kind", "ivf_sq8", "--out-dir", str(tmp / "v2" / "b")])

    def test_recall_command_prints_a_table_and_says_what_it_measures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            directory, _ = self.shards(tmp, n=1500)
            output = run(["faiss-recall", "--embeddings", str(directory), "--kind", "ivf_sq8", "--nlist", "8",
                          "--subset-rows", "1000", "--queries", "100", "--nprobe", "1", "8", "--train-size", "1000"])
            self.assertIn("IVF8,SQ8 on 1000 vectors, 100 queries", output)
            self.assertIn("nprobe    1:", output)
            self.assertIn("nprobe    8:", output)
            self.assertIn("not retrieval quality on the benchmark", output)
            low, high = [float(line.split(":")[1]) for line in output.splitlines() if "nprobe" in line and ":" in line and "recall" not in line]
            self.assertLessEqual(low, high)


@unittest.skipUnless(HAVE_EXPORT_STACK, "needs torch, transformers, tokenizers, onnx and onnxruntime")
class TestExportCommand(unittest.TestCase):
    def test_exports_and_verifies_a_tiny_embedder_and_reranker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_tiny_model(tmp / "emb")
            make_tiny_model(tmp / "rr", labels=1)
            for argv, name in (
                (["embedder", "--model", str(tmp / "emb"), "--out-dir", str(tmp / "o" / "embedder"), "--pooling", "mean"], "embedder"),
                (["reranker", "--model", str(tmp / "rr"), "--out-dir", str(tmp / "o" / "reranker")], "reranker"),
            ):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    exporter.main(argv)
                report = json.loads(buffer.getvalue())
                self.assertLessEqual(report["verification"]["max_abs_difference"], 1e-4, name)
                self.assertTrue((tmp / "o" / name / "model.onnx").is_file())

    def test_arguments(self):
        args = exporter.parse_args(["embedder", "--model", "m", "--out-dir", "o"])
        self.assertEqual((args.pooling, args.no_normalize, args.max_length), ("cls", False, 512))
        with self.assertRaises(SystemExit):
            exporter.parse_args(["embedder", "--model", "m"])


if __name__ == "__main__":
    unittest.main()

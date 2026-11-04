"""Tests for the approximate FAISS builders (plan task C4): real small indexes on random vectors, no models."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from legalrag.retrieval.dense import DenseRetriever
from legalrag.retrieval.faiss_builders import (
    EmbeddingShards,
    IndexSpec,
    build_index,
    faiss,
    load_index,
    metadata_path,
    recall_against_exact,
    set_nprobe,
    suggest_nlist,
    write_index,
)

DIM = 32


def clustered_vectors(n=3000, dim=DIM, clusters=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    x = (centers[rng.integers(0, clusters, n)] + 0.3 * rng.normal(size=(n, dim))).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def write_shards(directory: Path, vectors: np.ndarray, parts: int = 3, dtype="float32"):
    paths = []
    for i, block in enumerate(np.array_split(vectors, parts)):
        path = directory / f"emb_{i:03d}.npy"
        np.save(path, block.astype(dtype))
        paths.append(path)
    return paths


needs_faiss = unittest.skipIf(faiss is None, "faiss is not installed")


class TestEmbeddingShards(unittest.TestCase):
    def test_sizes_batches_and_order(self):
        x = clustered_vectors(1000)
        with tempfile.TemporaryDirectory() as tmp:
            shards = EmbeddingShards(write_shards(Path(tmp), x, parts=3))
            self.assertEqual((shards.total, shards.dim), (1000, DIM))
            rebuilt = np.concatenate(list(shards.batches(batch_rows=100)))
            np.testing.assert_allclose(rebuilt, x, atol=1e-7)

    def test_sample_is_reproducible_ordered_and_bounded(self):
        x = clustered_vectors(1000)
        with tempfile.TemporaryDirectory() as tmp:
            shards = EmbeddingShards(write_shards(Path(tmp), x))
            first = shards.sample(100, seed=5)
            np.testing.assert_array_equal(first, shards.sample(100, seed=5))
            self.assertFalse(np.array_equal(first, shards.sample(100, seed=6)))
            self.assertEqual(shards.sample(5000).shape[0], 1000)  # capped at the corpus size
            rows = np.sort(np.random.default_rng(5).choice(1000, size=100, replace=False))
            np.testing.assert_allclose(first, x[rows], atol=1e-7)  # rows come back in corpus order

    def test_float16_shards_are_accepted_and_converted(self):
        x = clustered_vectors(200)
        with tempfile.TemporaryDirectory() as tmp:
            shards = EmbeddingShards(write_shards(Path(tmp), x, parts=1, dtype="float16"))
            self.assertEqual(next(iter(shards.batches())).dtype, np.float32)

    def test_bad_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaisesRegex(ValueError, "No embedding files"):
                EmbeddingShards([])
            np.save(tmp / "a.npy", np.zeros((4, 8), dtype=np.float32))
            np.save(tmp / "b.npy", np.zeros((4, 9), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "same width"):
                EmbeddingShards([tmp / "a.npy", tmp / "b.npy"])
            np.save(tmp / "c.npy", np.zeros(8, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "2-D"):
                EmbeddingShards([tmp / "c.npy"])
            np.save(tmp / "d.npy", np.zeros((4, 8), dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "float32 or float16"):
                EmbeddingShards([tmp / "d.npy"])
            np.save(tmp / "e.npy", np.zeros((0, 8), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "no rows"):
                EmbeddingShards([tmp / "e.npy"])
            bad = np.ones((4, 8), dtype=np.float32)
            bad[1, 1] = np.nan
            np.save(tmp / "f.npy", bad)
            with self.assertRaisesRegex(ValueError, "NaN"):
                list(EmbeddingShards([tmp / "f.npy"]).batches())


class TestIndexSpec(unittest.TestCase):
    def test_factory_strings(self):
        self.assertEqual(IndexSpec("ivf_sq8", 768, 64).factory, "IVF64,SQ8")
        self.assertEqual(IndexSpec("opq_pq", 768, 64, pq_m=96).factory, "OPQ96,IVF64,PQ96x8")
        self.assertEqual(IndexSpec("opq_pq", 32, 8, pq_m=8, pq_bits=4).factory, "OPQ8,IVF8,PQ8x4")

    def test_invalid_specs(self):
        for kwargs, pattern in (
            (dict(kind="hnsw", dim=8, nlist=4), "kind"),
            (dict(kind="ivf_sq8", dim=8, nlist=0), "nlist"),
            (dict(kind="opq_pq", dim=8, nlist=4), "pq_m"),
            (dict(kind="opq_pq", dim=10, nlist=4, pq_m=4), "divide"),
            (dict(kind="opq_pq", dim=8, nlist=4, pq_m=4, pq_bits=17), "pq_bits"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    IndexSpec(**kwargs)

    def test_minimum_training_points(self):
        self.assertEqual(IndexSpec("ivf_sq8", 8, 100).min_training_points, 100)
        self.assertEqual(IndexSpec("opq_pq", 8, 4, pq_m=4, pq_bits=8).min_training_points, 256)

    def test_suggested_nlist_scales_with_the_square_root(self):
        self.assertEqual(suggest_nlist(1_000_000), 4000)
        self.assertEqual(suggest_nlist(1), 4)
        with self.assertRaises(ValueError):
            suggest_nlist(0)


@needs_faiss
class TestBuildAndSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = clustered_vectors(3000)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shards = EmbeddingShards(write_shards(Path(cls.tmp.name), cls.x))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def exact_top(self, rows=100, k=10):
        flat = faiss.IndexFlatIP(DIM)
        flat.add(self.x)
        return flat.search(self.x[:rows], k)[1]

    def recall(self, index, rows=100, k=10, nprobe=None):
        if nprobe:
            set_nprobe(index, nprobe)
        found = index.search(self.x[:rows], k)[1]
        return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(self.exact_top(rows, k), found)]))

    def test_ivf_sq8_holds_every_vector_in_order_and_is_accurate_when_all_lists_are_visited(self):
        spec = IndexSpec("ivf_sq8", DIM, 16)
        index, meta = build_index(self.shards, spec, train_size=2000, seed=1, nprobe=16)
        self.assertEqual((index.ntotal, index.d), (3000, DIM))
        self.assertEqual((meta["ntotal"], meta["factory"], meta["kind"], meta["metric"]), (3000, "IVF16,SQ8", "ivf_sq8", "inner_product"))
        self.assertGreaterEqual(self.recall(index, nprobe=16), 0.9)
        top1 = index.search(self.x[:200], 1)[1][:, 0]
        self.assertGreaterEqual(float(np.mean(top1 == np.arange(200))), 0.95)  # row i is still row i

    def test_sq8_recall_rises_as_more_lists_are_visited(self):
        index, _ = build_index(self.shards, IndexSpec("ivf_sq8", DIM, 16), train_size=3000, seed=1)
        recalls = [self.recall(index, nprobe=n) for n in (1, 4, 16)]
        self.assertEqual(recalls, sorted(recalls))
        self.assertGreater(recalls[-1], recalls[0])

    def test_pq_recall_is_limited_by_quantization_not_by_the_number_of_lists(self):
        # With product quantization the distances are approximate, so visiting more lists need not
        # help: this only checks that recall stays well above chance and roughly flat.
        index, _ = build_index(self.shards, IndexSpec("opq_pq", DIM, 16, pq_m=8, pq_bits=8), train_size=3000, seed=1)
        recalls = [self.recall(index, nprobe=n) for n in (1, 4, 16)]
        self.assertTrue(all(r > 0.2 for r in recalls), recalls)
        self.assertLess(max(recalls) - min(recalls), 0.1)

    def test_opq_pq_index_works_and_beats_chance(self):
        index, meta = build_index(self.shards, IndexSpec("opq_pq", DIM, 16, pq_m=8, pq_bits=8), train_size=3000, seed=1, nprobe=16)
        self.assertEqual((meta["pq_m"], meta["pq_bits"], index.ntotal), (8, 8, 3000))
        self.assertGreater(self.recall(index, nprobe=16), 0.2)  # chance would be 10 / 3000

    def test_build_is_reproducible(self):
        spec = IndexSpec("ivf_sq8", DIM, 8)
        first = build_index(self.shards, spec, train_size=1000, seed=3)[0].search(self.x[:20], 5)
        second = build_index(self.shards, spec, train_size=1000, seed=3)[0].search(self.x[:20], 5)
        np.testing.assert_array_equal(first[1], second[1])

    def test_metadata_records_how_it_was_built(self):
        _, meta = build_index(self.shards, IndexSpec("ivf_sq8", DIM, 8), train_size=800, seed=9, nprobe=4,
                              extra_metadata={"embedding_model": "test-model"})
        self.assertEqual((meta["train_size"], meta["seed"], meta["nprobe_default"], meta["nlist"]), (800, 9, 4, 8))
        self.assertEqual(meta["embedding_model"], "test-model")
        self.assertEqual(sum(f["rows"] for f in meta["source_files"]), 3000)
        json.dumps(meta)  # must be serialisable

    def test_default_training_size_is_39_points_per_list(self):
        _, meta = build_index(self.shards, IndexSpec("ivf_sq8", DIM, 16), seed=1)
        self.assertEqual(meta["train_size"], 39 * 16)

    def test_bad_builds_are_refused(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            build_index(self.shards, IndexSpec("ivf_sq8", DIM + 1, 8))
        with self.assertRaisesRegex(ValueError, "larger than the number of vectors"):
            build_index(self.shards, IndexSpec("ivf_sq8", DIM, 5000))
        with self.assertRaisesRegex(ValueError, "too small"):
            build_index(self.shards, IndexSpec("ivf_sq8", DIM, 64), train_size=10)

    def test_write_and_load_roundtrip_with_nprobe_and_mmap(self):
        index, meta = build_index(self.shards, IndexSpec("ivf_sq8", DIM, 16), train_size=2000, seed=1, nprobe=16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "dense.index"
            write_index(index, path, meta)
            self.assertFalse((path.parent / "dense.index.tmp").exists())
            self.assertEqual(json.loads(metadata_path(path).read_text())["ntotal"], 3000)
            reference = index.search(self.x[:30], 5)[1]
            for mmap in (False, True):
                loaded = load_index(path, nprobe=16, mmap=mmap)
                np.testing.assert_array_equal(loaded.search(self.x[:30], 5)[1], reference)
            self.assertEqual(faiss.extract_index_ivf(load_index(path, nprobe=3)).nprobe, 3)
            self.assertEqual(faiss.extract_index_ivf(load_index(path, nprobe=10**6)).nprobe, 16)  # capped at nlist

    def test_set_nprobe_on_a_flat_index_is_a_harmless_no(self):
        flat = faiss.IndexFlatIP(DIM)
        flat.add(self.x[:10])
        self.assertFalse(set_nprobe(flat, 4))
        with self.assertRaises(ValueError):
            set_nprobe(flat, 0)

    def test_dense_retriever_loads_and_searches_an_approximate_index(self):
        index, meta = build_index(self.shards, IndexSpec("ivf_sq8", DIM, 16), train_size=2000, seed=1, nprobe=16)
        ids = [f"chunk{i}" for i in range(3000)]

        class Embedder:
            def get_sentence_embedding_dimension(self):
                return DIM

            def encode(self, texts, **kwargs):
                # texts are row numbers as strings; anything else (the warmup text) maps to row 0
                return np.stack([self_x[int(t) if str(t).isdigit() else 0] for t in texts]).astype(np.float32)

        self_x = self.x
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense.index"
            write_index(index, path, meta)
            retriever = DenseRetriever.load_index(path, chunk_ids=ids, nprobe=16, mmap=True, embedder=Embedder())
            retriever.query_instruction = ""
            retriever.warmup()
            results = retriever.search("42", top_k=3)
            self.assertEqual(results[0][0], "chunk42")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                DenseRetriever.load_index(path, chunk_ids=ids[:-1])

    def test_recall_against_exact_reports_each_nprobe(self):
        index, _ = build_index(self.shards, IndexSpec("ivf_sq8", DIM, 16), train_size=2000, seed=1)
        table = recall_against_exact(index, self.x, self.x[:100], k=10, nprobe_values=[1, 16])
        self.assertEqual([row["nprobe"] for row in table], [1.0, 16.0])
        self.assertLessEqual(table[0]["recall_at_k"], table[1]["recall_at_k"])
        self.assertGreaterEqual(table[1]["recall_at_k"], 0.9)
        flat = faiss.IndexFlatIP(DIM)
        flat.add(self.x)
        self.assertEqual(recall_against_exact(flat, self.x, self.x[:50], k=10)[0]["recall_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()

"""Tests for the memory-mapped bm25s backend and the chunk id array (plan task C4)."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from legalrag.retrieval.bm25 import BM25Retriever
from legalrag.retrieval.bm25s_backend import (
    CHUNK_IDS_FILE,
    METADATA_FILE,
    Bm25sRetriever,
    bm25s,
)
from legalrag.retrieval.chunk_ids import ChunkIdArray
from legalrag.retrieval.fusion import reciprocal_rank_fusion

TEXTS = [
    "anticipatory bail under section 438 crpc granted to the applicant",
    "writ petition under article 226 of the constitution dismissed",
    "section 138 negotiable instruments act cheque dishonour statutory notice",
    "arbitration award set aside section 34 patent illegality",
    "bail rejected bail bail bail",
]
IDS = [f"c{i}" for i in range(len(TEXTS))]

needs_bm25s = unittest.skipIf(bm25s is None, "bm25s is not installed")


class TestChunkIdArray(unittest.TestCase):
    def test_behaves_like_a_read_only_list_of_strings(self):
        ids = ChunkIdArray.from_ids(["a1", "b22", "c333"])
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[1], "b22")
        self.assertEqual(ids[-1], "c333")
        self.assertIsInstance(ids[0], str)  # plain str, not numpy.str_
        self.assertEqual(ids[1:], ["b22", "c333"])
        self.assertEqual(list(ids), ["a1", "b22", "c333"])
        self.assertEqual(ids.take([2, 0]), ["c333", "a1"])
        self.assertTrue(ids)
        self.assertIn("b22", ids)
        self.assertEqual(ids, ["a1", "b22", "c333"])
        self.assertEqual(ids, ChunkIdArray.from_ids(["a1", "b22", "c333"]))
        self.assertNotEqual(ids, ["a1"])

    def test_empty_array_is_falsy(self):
        self.assertFalse(ChunkIdArray.from_ids([]))

    def test_empty_id_is_rejected(self):
        with self.assertRaises(ValueError):
            ChunkIdArray.from_ids(["ok", ""])

    def test_save_and_load_with_and_without_mmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "ids.npy"
            ChunkIdArray.from_ids([f"id-{i}" for i in range(1000)]).save(path)
            mapped = ChunkIdArray.load(path, mmap=True)
            self.assertIsInstance(mapped._array, np.memmap)
            self.assertEqual(mapped[999], "id-999")
            self.assertEqual(len(list(mapped)), 1000)
            self.assertNotIsInstance(ChunkIdArray.load(path, mmap=False)._array, np.memmap)

    def test_unicode_ids_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.npy"
            ChunkIdArray.from_ids(["धारा-1", "plain"]).save(path)
            self.assertEqual(list(ChunkIdArray.load(path)), ["धारा-1", "plain"])

    def test_must_be_one_dimensional(self):
        with self.assertRaises(ValueError):
            ChunkIdArray(np.zeros((2, 2)))


@needs_bm25s
class TestBm25sRetriever(unittest.TestCase):
    def build(self):
        return Bm25sRetriever.build(IDS, TEXTS)

    def test_finds_the_relevant_chunk_first(self):
        results = self.build().search("anticipatory bail section 438", top_k=3)
        self.assertEqual(results[0][0], "c0")
        self.assertTrue(all(score > 0 for _, score in results))
        self.assertEqual([s for _, s in results], sorted((s for _, s in results), reverse=True))

    def test_returns_python_types_in_the_v1_shape(self):
        (chunk_id, score), *_ = self.build().search("cheque", top_k=1)
        self.assertEqual((type(chunk_id), type(score)), (str, float))

    def test_scores_equal_rank_bm25_divided_by_k1_plus_one_when_no_term_is_in_half_the_chunks(self):
        rng = random.Random(0)
        rare = [f"w{i}" for i in range(200)]
        common = ["court", "section", "order"]
        texts = [" ".join(rng.choice(rare + common) for _ in range(rng.randint(20, 120))) for _ in range(400)]
        ids = [f"x{i}" for i in range(400)]
        # Precondition, computed directly: rank_bm25 floors a negative idf, so check document frequency.
        doc_freq = {}
        for text in texts:
            for token in set(text.split()):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self.assertLess(max(doc_freq.values()) * 2, len(texts), "fixture must keep every term below half the chunks")

        old = BM25Retriever(chunk_ids=ids, corpus_texts=texts, k1=1.5, b=0.75)
        new = Bm25sRetriever.build(ids, texts, k1=1.5, b=0.75)
        for _ in range(40):
            query = " ".join(rng.choice(rare + common) for _ in range(rng.randint(1, 7)))
            expected = dict(old.search(query, top_k=len(ids)))
            got = new.search(query, top_k=10)
            self.assertTrue(got)
            for cid, score in got:
                self.assertAlmostEqual(score * 2.5, expected[cid], delta=1e-4 * max(1.0, expected[cid]))
            # Same top results, apart from ties at the cut-off.
            best_old = sorted(expected.values(), reverse=True)[9]
            self.assertTrue(all(expected[cid] >= best_old - 1e-4 for cid, _ in got))

    def test_a_term_in_most_chunks_scores_zero_where_rank_bm25_gives_a_small_weight(self):
        texts = [f"court filler{i}" for i in range(10)] + ["different words entirely"]
        ids = [f"d{i}" for i in range(11)]
        old = BM25Retriever(chunk_ids=ids, corpus_texts=texts).search("court", top_k=1)
        self.assertGreater(old[0][1], 0)  # epsilon * average idf
        self.assertEqual(Bm25sRetriever.build(ids, texts).search("court", top_k=5), [])  # documented difference

    def test_repeated_query_terms_add_up_like_rank_bm25(self):
        old = BM25Retriever(chunk_ids=IDS, corpus_texts=TEXTS)
        new = self.build()
        for cid, score in new.search("bail bail", top_k=2):
            self.assertAlmostEqual(score * 2.5, dict(old.search("bail bail", top_k=5))[cid], places=4)

    def test_unknown_or_empty_query_returns_nothing(self):
        retriever = self.build()
        self.assertEqual(retriever.search("zzzz qqqq", 5), [])
        self.assertEqual(retriever.search("", 5), [])
        self.assertEqual(retriever.search("   ...  ", 5), [])

    def test_top_k_edges(self):
        retriever = self.build()
        self.assertEqual(retriever.search("bail", 0), [])
        self.assertEqual(retriever.search("bail", -3), [])
        self.assertEqual(len(retriever.search("bail", 100)), 2)  # only two chunks match; zero scores are not padded
        self.assertEqual(len(retriever.search("bail", 1)), 1)

    def test_ties_are_broken_by_index_order_and_repeatably(self):
        texts = ["same words here", "same words here", "other text", "another thing", "more", "last"]
        retriever = Bm25sRetriever.build(["a", "b", "c", "d", "e", "f"], texts)
        first = retriever.search("same", 3)
        self.assertEqual([cid for cid, _ in first], ["a", "b"])
        self.assertEqual(first[0][1], first[1][1])
        self.assertEqual(first, retriever.search("same", 3))

    def test_document_with_no_tokens_does_not_break_indexing_or_search(self):
        retriever = Bm25sRetriever.build(["empty", "bail", "x", "y"], ["", "bail granted", "other text", "more words"])
        self.assertEqual([cid for cid, _ in retriever.search("bail", 2)], ["bail"])

    def test_works_inside_the_rrf_fusion_step(self):
        retriever = self.build()
        fused = reciprocal_rank_fusion([[cid for cid, _ in retriever.search("bail", 5)], ["c1", "c0"]], k=60, top_n=5)
        self.assertEqual(fused[0][0], "c0")

    def test_save_load_roundtrip_gives_identical_results(self):
        original = self.build()
        with tempfile.TemporaryDirectory() as tmp:
            original.save(tmp)
            self.assertTrue((Path(tmp) / CHUNK_IDS_FILE).is_file())
            for mmap in (True, False):
                loaded = Bm25sRetriever.load(tmp, mmap=mmap)
                for query in ("bail", "section 138 cheque", "writ article 226"):
                    self.assertEqual(loaded.search(query, 5), original.search(query, 5))
                self.assertEqual((loaded.k1, loaded.b, loaded.method), (1.5, 0.75, "robertson"))

    def test_mmap_load_keeps_the_arrays_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build().save(tmp)
            mapped = Bm25sRetriever.load(tmp, mmap=True)
            for key in ("data", "indices", "indptr"):
                self.assertIsInstance(mapped.model.scores[key], np.memmap, key)
            self.assertIsInstance(mapped.chunk_ids._array, np.memmap)
            resident = Bm25sRetriever.load(tmp, mmap=False)
            self.assertNotIsInstance(resident.model.scores["data"], np.memmap)

    def test_load_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaisesRegex(ValueError, "not a bm25s index"):
                Bm25sRetriever.load(tmp)
            self.build().save(tmp)
            meta = json.loads((tmp / METADATA_FILE).read_text())
            for change, pattern in (
                ({"tokenizer": "another tokenizer"}, "tokenizer"),
                ({"format": 99}, "format"),
                ({"num_docs": 3}, "disagree"),
            ):
                (tmp / METADATA_FILE).write_text(json.dumps({**meta, **change}))
                with self.subTest(change=change):
                    with self.assertRaisesRegex(ValueError, pattern):
                        Bm25sRetriever.load(tmp)
            (tmp / METADATA_FILE).write_text(json.dumps(meta))
            ChunkIdArray.from_ids(["only", "two"]).save(tmp / CHUNK_IDS_FILE)
            with self.assertRaisesRegex(ValueError, "does not match"):
                Bm25sRetriever.load(tmp)

    def test_validate_matches_the_v1_contract(self):
        with self.assertRaisesRegex(ValueError, "not loaded"):
            Bm25sRetriever(model=None, chunk_ids=IDS).validate()
        built = self.build()
        with self.assertRaisesRegex(ValueError, "no chunk_ids"):
            Bm25sRetriever(model=built.model, chunk_ids=[]).validate()
        with self.assertRaisesRegex(ValueError, "does not match"):
            Bm25sRetriever(model=built.model, chunk_ids=IDS[:2]).validate()
        built.validate()

    def test_search_on_an_uninitialised_retriever_raises(self):
        with self.assertRaises(ValueError):
            Bm25sRetriever().search("bail")

    def test_build_rejects_bad_input(self):
        with self.assertRaisesRegex(ValueError, "one to one"):
            Bm25sRetriever.build(["a"], ["x", "y"])
        with self.assertRaisesRegex(ValueError, "empty corpus"):
            Bm25sRetriever.build([], [])
        with self.assertRaisesRegex(ValueError, "any token"):
            Bm25sRetriever.build(["a"], ["!!! ..."])

    def test_chunk_ids_are_memory_mappable_after_build_and_save(self):
        built = self.build()
        self.assertIsInstance(built.chunk_ids, ChunkIdArray)
        self.assertEqual(built.chunk_ids, IDS)


if __name__ == "__main__":
    unittest.main()

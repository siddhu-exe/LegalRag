"""Tests for LegalChunker with locked baseline parameters."""

import unittest
from legalrag.preprocessing.chunker import LegalChunker, Chunk


class TestChunker(unittest.TestCase):
    def test_chunker_basic_splitting(self):
        chunker = LegalChunker(chunk_size=1200, chunk_overlap=200, min_chunk_length=100)
        text = ("Paragraph one of the judgment. " * 30) + "\n\n" + ("Paragraph two holding that the appeal is allowed. " * 30)
        chunks = chunker.split_text(text, cnr="DLHC010000012020")

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))
        self.assertEqual(chunks[0].chunk_id, "DLHC010000012020_chunk_0")
        self.assertEqual(chunks[1].chunk_id, "DLHC010000012020_chunk_1")
        self.assertEqual(chunks[0].cnr, "DLHC010000012020")
        self.assertGreaterEqual(len(chunks[0].text), 100)

    def test_chunker_short_text_filtered(self):
        chunker = LegalChunker(chunk_size=1200, chunk_overlap=200, min_chunk_length=100)
        short_text = "Short text under 100 chars."
        chunks = chunker.split_text(short_text, cnr="DLHC010000012020")
        self.assertEqual(len(chunks), 0)

    def test_chunker_with_metadata(self):
        chunker = LegalChunker(chunk_size=1200, chunk_overlap=200, min_chunk_length=100)
        text = "Valid long judgment passage. " * 20
        meta = {"court_name": "Delhi High Court", "decision_date": "2020-05-15"}
        chunks = chunker.split_text(text, cnr="DLHC010000012020", metadata=meta)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].court_name, "Delhi High Court")
        self.assertEqual(chunks[0].decision_date, "2020-05-15")


if __name__ == "__main__":
    unittest.main()

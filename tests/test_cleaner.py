"""Tests for LegalRAG corpus cleaning and corruption detection."""

import unittest
from legalrag.preprocessing.cleaner import clean_judgment_text, is_corrupted_document


class TestCleaner(unittest.TestCase):
    def test_clean_judgment_text_normal(self):
        raw = "The accused was convicted under Section 302 IPC.\n\nSentence: Life imprisonment."
        cleaned = clean_judgment_text(raw)
        self.assertEqual(
            cleaned,
            "The accused was convicted under Section 302 IPC. Sentence: Life imprisonment.",
        )

    def test_clean_judgment_text_control_characters(self):
        raw = "Judgment\x00\x08Text\x0bwith\x1fcontrols\x7f."
        cleaned = clean_judgment_text(raw)
        self.assertEqual(cleaned, "Judgment Text with controls .")

    def test_clean_judgment_text_empty_and_none(self):
        self.assertEqual(clean_judgment_text(""), "")
        self.assertEqual(clean_judgment_text(None), "")

    def test_is_corrupted_document_clean(self):
        clean_text = "This is a clean judgment transcript with zero corrupt characters."
        self.assertFalse(is_corrupted_document(clean_text))

    def test_is_corrupted_document_corrupted(self):
        corrupted_text = "A" * 100 + "\x00" * 10  # 10 control characters out of 110 (~9%)
        self.assertTrue(is_corrupted_document(corrupted_text, threshold=0.01))

    def test_is_corrupted_document_empty(self):
        self.assertTrue(is_corrupted_document(""))
        self.assertTrue(is_corrupted_document(None))


if __name__ == "__main__":
    unittest.main()

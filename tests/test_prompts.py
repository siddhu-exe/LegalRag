"""Tests for prompt formatting, context metadata, and citation parsing."""

import unittest

from legalrag.generation.prompts import (
    SYSTEM_PROMPT,
    build_rag_prompt,
    extract_citations,
    format_context_block,
)


class TestPrompts(unittest.TestCase):
    def test_format_context_block_uses_court_code(self):
        block = format_context_block(
            chunk_id="DLHC_chunk_0",
            text="The writ petition stands dismissed.",
            court_code="DLHC",
            decision_date="2021-03-01",
        )
        self.assertIn("[Chunk ID: DLHC_chunk_0 | Court: DLHC | Date: 2021-03-01]", block)
        self.assertIn("The writ petition stands dismissed.", block)
        # The real schema has no court_name column; no court name should be invented.
        self.assertNotIn("High Court", block)

    def test_format_context_block_handles_missing_court_code(self):
        block = format_context_block(chunk_id="c1", text="text")
        self.assertIn("[Chunk ID: c1 | Court: Unspecified | Date: Unspecified]", block)

    def test_build_rag_prompt(self):
        context_blocks = [
            {
                "chunk_id": "chunk_1",
                "text": "The respondent failed to pay Rs. 50,000.",
                "court_code": "BHC",
                "decision_date": "2019-11-20",
            }
        ]
        prompt = build_rag_prompt("What was the default amount?", context_blocks)

        self.assertIn("Retrieved Judgment Context Passages:", prompt)
        self.assertIn("[Chunk ID: chunk_1 | Court: BHC | Date: 2019-11-20]", prompt)
        self.assertIn("The respondent failed to pay Rs. 50,000.", prompt)
        self.assertIn("What was the default amount?", prompt)

    def test_system_prompt_requires_citation(self):
        self.assertIn("[Chunk ID:", SYSTEM_PROMPT)


class TestCitationParser(unittest.TestCase):
    """Citation parser must extract ONLY the chunk ID from any citation form."""

    def test_normal_bare_citation(self):
        self.assertEqual(extract_citations("Holding per [Chunk ID: abc123]."), ["abc123"])

    def test_full_context_header(self):
        text = "See [Chunk ID: abc123 | Court: Indian High Court | Date: 2020-01-01] for details."
        self.assertEqual(extract_citations(text), ["abc123"])

    def test_multiple_citations_are_deduplicated_in_order(self):
        text = (
            "[Chunk ID: c1 | Court: X | Date: 2020-01-01] and "
            "[Chunk ID: c2] and again [Chunk ID: c1 | Court: X | Date: 2020-01-01]."
        )
        self.assertEqual(extract_citations(text), ["c1", "c2"])

    def test_case_insensitive(self):
        self.assertEqual(extract_citations("[chunk id: AbC-9]"), ["AbC-9"])

    def test_malformed_citation_ignored(self):
        self.assertEqual(extract_citations("[Chunk ID: ]"), [])
        self.assertEqual(extract_citations("[Chunk ID:]"), [])
        self.assertEqual(extract_citations("no citations here"), [])
        self.assertEqual(extract_citations(""), [])

    def test_unknown_chunk_id_is_extracted_for_later_validation(self):
        # The parser extracts; the API layer separately filters against retrieved context.
        self.assertEqual(extract_citations("[Chunk ID: unknown_chunk_999]"), ["unknown_chunk_999"])


if __name__ == "__main__":
    unittest.main()

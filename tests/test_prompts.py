"""Tests for prompt formatting and context block builder."""

import unittest
from legalrag.generation.prompts import format_context_block, build_rag_prompt, SYSTEM_PROMPT


class TestPrompts(unittest.TestCase):
    def test_format_context_block(self):
        block = format_context_block(
            chunk_id="DLHC_chunk_0",
            text="The writ petition stands dismissed.",
            court_name="Delhi High Court",
            decision_date="2021-03-01",
        )
        self.assertIn("[Chunk ID: DLHC_chunk_0 | Court: Delhi High Court | Date: 2021-03-01]", block)
        self.assertIn("The writ petition stands dismissed.", block)

    def test_build_rag_prompt(self):
        context_blocks = [
            {
                "chunk_id": "chunk_1",
                "text": "The respondent failed to pay Rs. 50,000.",
                "court_name": "Bombay High Court",
                "decision_date": "2019-11-20",
            }
        ]
        prompt = build_rag_prompt("What was the default amount?", context_blocks)

        self.assertIn("Retrieved Judgment Context Passages:", prompt)
        self.assertIn("[Chunk ID: chunk_1 | Court: Bombay High Court | Date: 2019-11-20]", prompt)
        self.assertIn("The respondent failed to pay Rs. 50,000.", prompt)
        self.assertIn("What was the default amount?", prompt)


if __name__ == "__main__":
    unittest.main()

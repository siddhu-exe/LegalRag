"""Tests for cost estimates, resumable checkpoints and run-config checks (plan task C3)."""

import json
import tempfile
import unittest
from pathlib import Path

from legalrag.evaluation.runtime import (
    CostLine,
    JsonlCheckpoint,
    build_cost_line,
    confirm_or_abort,
    ensure_run_config,
    estimate_tokens,
    format_cost_estimate,
)


class TestCost(unittest.TestCase):
    def test_token_estimate(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcde"), 2)

    def test_cost_line_math(self):
        line = CostLine("judge", calls=10, input_tokens=2_000_000, output_tokens=500_000,
                        price_in_per_mtok=0.5, price_out_per_mtok=2.0)
        self.assertAlmostEqual(line.usd, 2.0)

    def test_no_price_means_no_dollar_figure(self):
        line = build_cost_line("judge", ["a" * 400] * 3, expected_output_tokens=100)
        self.assertIsNone(line.usd)
        self.assertEqual((line.calls, line.input_tokens, line.output_tokens), (3, 300, 300))
        text = format_cost_estimate([line])
        self.assertIn("price not given", text)
        self.assertNotIn("$", text.split("Pass")[0])

    def test_total_is_shown_for_several_priced_lines(self):
        lines = [
            build_cost_line("generation", ["x" * 4000], 100, 1.0, 2.0),
            build_cost_line("judge", ["x" * 4000], 100, 1.0, 2.0),
        ]
        self.assertIn("total", format_cost_estimate(lines))


class TestConfirm(unittest.TestCase):
    def test_yes_flag_skips_the_question(self):
        confirm_or_abort("go?", assume_yes=True, input_fn=lambda _: self.fail("asked"))

    def test_non_interactive_without_yes_aborts(self):
        with self.assertRaises(SystemExit) as ctx:
            confirm_or_abort("go?", assume_yes=False, interactive=False)
        self.assertIn("--yes", str(ctx.exception))

    def test_interactive_answers(self):
        confirm_or_abort("go?", input_fn=lambda _: "y", interactive=True)
        with self.assertRaises(SystemExit):
            confirm_or_abort("go?", input_fn=lambda _: "", interactive=True)
        with self.assertRaises(SystemExit):
            confirm_or_abort("go?", input_fn=lambda _: "no", interactive=True)


class TestCheckpoint(unittest.TestCase):
    def test_saves_each_item_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "items.jsonl"
            first = JsonlCheckpoint(path)
            first.append({"item_id": "q1", "status": "ok", "x": 1})
            # The line is on disk before the next item starts.
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            first.append({"item_id": "q2", "status": "ok"})

            again = JsonlCheckpoint(path)
            self.assertTrue(again.is_done("q1"))
            self.assertFalse(again.is_done("q3"))
            self.assertEqual({r["item_id"] for r in again.values()}, {"q1", "q2"})

    def test_truncated_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            path.write_text(json.dumps({"item_id": "q1", "status": "ok"}) + "\n" + '{"item_id": "q2", "sta', encoding="utf-8")
            checkpoint = JsonlCheckpoint(path)
            self.assertTrue(checkpoint.is_done("q1"))
            self.assertFalse(checkpoint.is_done("q2"))
            self.assertEqual(checkpoint.skipped_lines, 1)

    def test_error_items_can_be_retried_and_the_last_record_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = JsonlCheckpoint(Path(tmp) / "items.jsonl")
            checkpoint.append({"item_id": "q1", "status": "error"})
            self.assertTrue(checkpoint.is_done("q1"))
            self.assertFalse(checkpoint.is_done("q1", retry_statuses=["error"]))
            checkpoint.append({"item_id": "q1", "status": "ok"})
            self.assertTrue(checkpoint.is_done("q1", retry_statuses=["error"]))
            self.assertEqual(JsonlCheckpoint(Path(tmp) / "items.jsonl").records["q1"]["status"], "ok")

    def test_ids_are_compared_as_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = JsonlCheckpoint(Path(tmp) / "items.jsonl")
            checkpoint.append({"item_id": 7, "status": "ok"})
            self.assertTrue(checkpoint.is_done("7"))

    def test_unicode_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            JsonlCheckpoint(path).append({"item_id": "q1", "status": "ok", "answer": "धारा 438"})
            self.assertEqual(JsonlCheckpoint(path).records["q1"]["answer"], "धारा 438")


class TestRunConfig(unittest.TestCase):
    def test_first_run_writes_and_same_settings_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.json"
            config = {"judge_model": "judge-a", "generator_model": "gen-b", "note": "x"}
            ensure_run_config(path, config)
            self.assertTrue(path.exists())
            ensure_run_config(path, dict(config, note="changed"), keys=["judge_model", "generator_model"])

    def test_changed_judge_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.json"
            ensure_run_config(path, {"judge_model": "judge-a"})
            with self.assertRaises(SystemExit) as ctx:
                ensure_run_config(path, {"judge_model": "judge-z"})
            self.assertIn("judge_model", str(ctx.exception))
            ensure_run_config(path, {"judge_model": "judge-z"}, force=True)


if __name__ == "__main__":
    unittest.main()

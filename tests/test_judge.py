"""Tests for the independent judge and generation summary (plan task C3). Stub clients only."""

import json
import unittest
from types import SimpleNamespace

from legalrag.evaluation.judge import (
    FAILURE_LABELS,
    JudgeIndependenceError,
    JudgeParseError,
    LLMJudge,
    JudgeVerdict,
    build_judge_prompt,
    check_independence,
    is_refusal,
    label_failure,
    model_family,
    normalize_model_id,
    parse_judge_response,
    summarize_generation,
)
from legalrag.generation.client import GenerationError

GOOD = {
    "answer_relevance": 4,
    "faithfulness": 3,
    "citation_correctness": 4,
    "overall_score": 4,
    "unsupported_claim_rate": 0.1,
    "hallucination": False,
    "rationale": "Mostly supported.",
}
BLOCKS = [{"chunk_id": "c1", "text": "Section 438 CrPC allows anticipatory bail.", "court_code": "27_1"}]


class StubClient:
    """Replays queued replies. A reply may be text, or an exception to raise."""

    def __init__(self, replies, model_name="judge-model-1"):
        self.model_name = model_name
        self.replies = list(replies)
        self.prompts = []

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1024):
        self.prompts.append((system_prompt, user_prompt, temperature))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply, prompt_tokens=100, completion_tokens=20)


class TestIndependence(unittest.TestCase):
    def test_same_model_is_refused(self):
        with self.assertRaises(JudgeIndependenceError):
            check_independence("qwen/qwen3.8-27b", "qwen/qwen3.8-27b")

    def test_same_model_with_different_provider_prefix_or_case_is_refused(self):
        with self.assertRaises(JudgeIndependenceError):
            check_independence("Qwen3.8-27B", "groq/qwen3.8-27b")

    def test_blank_judge_is_refused(self):
        with self.assertRaises(JudgeIndependenceError):
            check_independence("  ", "qwen/qwen3.8-27b")

    def test_different_model_passes_without_warnings(self):
        self.assertEqual(check_independence("claude-judge-5", "qwen/qwen3.8-27b", "gpt-writer-1"), [])

    def test_same_family_is_allowed_with_a_warning(self):
        warnings = check_independence("qwen/qwen3.8-72b", "qwen/qwen3.8-27b")
        self.assertEqual(len(warnings), 1)
        self.assertIn("family", warnings[0])

    def test_judge_equal_to_question_model_warns(self):
        warnings = check_independence("judge-a", "generator-b", "judge-a")
        self.assertTrue(any("question" in w for w in warnings))

    def test_helpers(self):
        self.assertEqual(normalize_model_id(" Meta/Llama-3.3-70B "), "llama-3.3-70b")
        self.assertEqual(model_family("qwen/qwen3.8-27b"), "qwen")
        self.assertEqual(model_family("gpt-4o"), "gpt")

    def test_llm_judge_refuses_to_start_with_the_generator_model(self):
        with self.assertRaises(JudgeIndependenceError):
            LLMJudge(StubClient([], "same-model"), generator_model="same-model")


class TestParsing(unittest.TestCase):
    def test_parses_plain_json(self):
        scores = parse_judge_response(json.dumps(GOOD))
        self.assertEqual((scores.answer_relevance, scores.faithfulness), (4, 3))
        self.assertFalse(scores.hallucination)
        self.assertEqual(scores.to_eval_record()["unsupported_claim_rate"], 0.1)

    def test_tolerates_fences_think_blocks_and_chatter(self):
        reply = "<think>reasoning here {not json}</think>\nHere you go:\n```json\n" + json.dumps(GOOD) + "\n```"
        self.assertEqual(parse_judge_response(reply).overall_score, 4)

    def test_accepts_string_booleans_and_integral_floats(self):
        data = dict(GOOD, hallucination="True", faithfulness=3.0)
        scores = parse_judge_response(json.dumps(data))
        self.assertTrue(scores.hallucination)
        self.assertEqual(scores.faithfulness, 3)

    def test_rejects_bad_replies(self):
        bad_cases = [
            "",
            "no json at all",
            "{broken json",
            json.dumps([1, 2]),
            json.dumps({k: v for k, v in GOOD.items() if k != "faithfulness"}),
            json.dumps(dict(GOOD, faithfulness=5)),
            json.dumps(dict(GOOD, faithfulness=0)),
            json.dumps(dict(GOOD, faithfulness=2.5)),
            json.dumps(dict(GOOD, faithfulness="high")),
            json.dumps(dict(GOOD, unsupported_claim_rate=1.5)),
            json.dumps(dict(GOOD, unsupported_claim_rate="0.1")),
            json.dumps(dict(GOOD, hallucination="maybe")),
            json.dumps(dict(GOOD, hallucination=1)),
        ]
        for reply in bad_cases:
            with self.subTest(reply=reply):
                with self.assertRaises(JudgeParseError):
                    parse_judge_response(reply)


class TestNonFiniteNumbers(unittest.TestCase):
    """NaN and Infinity are valid to json.loads but are never valid scores (review fix 3)."""

    FIELDS = ("answer_relevance", "faithfulness", "citation_correctness", "overall_score", "unsupported_claim_rate")
    CONSTANTS = ("NaN", "Infinity", "-Infinity", "1e999", "-1e999")

    def reply(self, field, literal):
        return json.dumps(dict(GOOD, **{field: 0})).replace(f'"{field}": 0', f'"{field}": {literal}')

    def test_every_numeric_field_rejects_every_non_finite_literal_as_a_parse_error(self):
        for field in self.FIELDS:
            for literal in self.CONSTANTS:
                with self.subTest(field=field, literal=literal):
                    with self.assertRaises(JudgeParseError):
                        parse_judge_response(self.reply(field, literal))

    def test_non_finite_literal_inside_the_rationale_text_is_just_text(self):
        scores = parse_judge_response(json.dumps(dict(GOOD, rationale="the model said NaN and Infinity")))
        self.assertIn("NaN", scores.rationale)

    def test_judge_records_a_parse_error_instead_of_raising(self):
        for literal in ("NaN", "Infinity"):
            with self.subTest(literal=literal):
                client = StubClient([self.reply("faithfulness", literal)] * 2)
                judge = LLMJudge(client, generator_model="generator-model-2")
                verdict = judge.judge("q", BLOCKS, "answer")
                self.assertEqual(verdict.status, "parse_error")
                self.assertEqual(verdict.error_type, "JudgeParseError")
                self.assertIsNone(verdict.scores)
                self.assertEqual(verdict.attempts, 2)  # retried once, like any unreadable reply

    def test_a_good_reply_after_a_non_finite_one_is_accepted(self):
        client = StubClient([self.reply("faithfulness", "NaN"), json.dumps(GOOD)])
        verdict = LLMJudge(client, generator_model="generator-model-2").judge("q", BLOCKS, "answer")
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.attempts, 2)

    def test_label_is_a_judge_system_error_not_a_bad_answer(self):
        client = StubClient([self.reply("overall_score", "NaN")] * 2)
        verdict = LLMJudge(client, generator_model="generator-model-2").judge("q", BLOCKS, "answer")
        label = label_failure(True, False, verdict, ["c1"], ["c1"])
        self.assertEqual(label, "System Error (judge)")


class TestPrompt(unittest.TestCase):
    def test_prompt_has_question_context_and_answer_but_no_system_identity(self):
        prompt = build_judge_prompt("What is s.438?", BLOCKS, "It allows bail [Chunk ID: c1].")
        self.assertIn("What is s.438?", prompt)
        self.assertIn("[Chunk ID: c1 | Court: 27_1", prompt)
        self.assertIn("It allows bail", prompt)
        self.assertNotIn("REFERENCE", prompt)

    def test_reference_is_included_only_when_given(self):
        self.assertIn("REFERENCE EVIDENCE", build_judge_prompt("q", BLOCKS, "a", reference="gold text"))

    def test_empty_context_is_stated(self):
        self.assertIn("no passages", build_judge_prompt("q", [], "a"))


class TestLLMJudge(unittest.TestCase):
    def make(self, replies, **kwargs):
        client = StubClient(replies)
        return LLMJudge(client, generator_model="generator-model-2", **kwargs), client

    def test_ok_verdict_and_token_accounting(self):
        judge, client = self.make([json.dumps(GOOD)])
        verdict = judge.judge("q", BLOCKS, "answer")
        self.assertTrue(verdict.ok)
        self.assertEqual((verdict.attempts, verdict.prompt_tokens, verdict.completion_tokens), (1, 100, 20))
        self.assertEqual(client.prompts[0][2], 0.0)  # temperature

    def test_one_retry_after_unreadable_reply(self):
        judge, client = self.make(["not json", json.dumps(GOOD)])
        verdict = judge.judge("q", BLOCKS, "answer")
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.attempts, 2)
        self.assertIn("could not be read", client.prompts[1][1])

    def test_gives_up_as_parse_error_not_as_a_bad_answer(self):
        judge, _ = self.make(["not json", "still not json"])
        verdict = judge.judge("q", BLOCKS, "answer")
        self.assertEqual(verdict.status, "parse_error")
        self.assertIsNone(verdict.scores)
        self.assertFalse(verdict.ok)

    def test_provider_failure_is_an_api_error_without_raw_text(self):
        judge, _ = self.make([GenerationError("secret key sk-123 leaked", 502, "AuthenticationError")])
        verdict = judge.judge("q", BLOCKS, "answer")
        self.assertEqual((verdict.status, verdict.error_type), ("api_error", "AuthenticationError"))
        self.assertNotIn("sk-123", json.dumps(verdict.__dict__, default=str))

    def test_unexpected_exception_is_an_api_error_with_class_name_only(self):
        judge, _ = self.make([RuntimeError("raw provider text")])
        verdict = judge.judge("q", BLOCKS, "answer")
        self.assertEqual((verdict.status, verdict.error_type), ("api_error", "RuntimeError"))

    def test_no_parse_retries_when_disabled(self):
        judge, client = self.make(["bad"], max_parse_retries=0)
        self.assertEqual(judge.judge("q", BLOCKS, "answer").status, "parse_error")
        self.assertEqual(len(client.prompts), 1)

    def test_independence_warnings_are_kept(self):
        judge = LLMJudge(StubClient([], "qwen/qwen3.8-72b"), generator_model="qwen/qwen3.8-27b")
        self.assertEqual(len(judge.warnings), 1)


class TestRefusalAndLabels(unittest.TestCase):
    REFUSAL = "The provided judgment context does not contain sufficient information to answer this inquiry."

    def test_refusal_detection(self):
        self.assertTrue(is_refusal(self.REFUSAL))
        self.assertTrue(is_refusal(self.REFUSAL.upper()))
        self.assertFalse(is_refusal("Bail is allowed [Chunk ID: c1]."))
        self.assertFalse(is_refusal(self.REFUSAL + " But see [Chunk ID: c1]."))
        self.assertFalse(is_refusal(None))
        self.assertFalse(is_refusal(""))

    def ok_verdict(self, **overrides):
        data = dict(GOOD, **overrides)
        return JudgeVerdict("ok", scores=parse_judge_response(json.dumps(data)), judge_model="j")

    def test_labels(self):
        top = ["c1", "c2", "c3", "c4", "c5"]
        cases = [
            (dict(generation_ok=False, refused=False, judge=None, gold_ids=["c1"]), "System Error (generation)"),
            (dict(generation_ok=True, refused=True, judge=None, gold_ids=["c9"]), "Refusal (evidence not retrieved)"),
            (dict(generation_ok=True, refused=True, judge=None, gold_ids=["c2"]), "Wrongful Refusal (evidence retrieved)"),
            (dict(generation_ok=True, refused=False, judge=None, gold_ids=["c2"]), "System Error (judge)"),
            (dict(generation_ok=True, refused=False, judge=JudgeVerdict("parse_error"), gold_ids=["c2"]), "System Error (judge)"),
            (dict(generation_ok=True, refused=False, judge=self.ok_verdict(), gold_ids=["c9"]), "Retrieval Failure"),
            (dict(generation_ok=True, refused=False, judge=self.ok_verdict(), gold_ids=["c2"]), "No Major Failure (Success)"),
            (dict(generation_ok=True, refused=False, judge=self.ok_verdict(hallucination=True), gold_ids=["c2"]), "Generation Grounding Failure"),
            (dict(generation_ok=True, refused=False, judge=self.ok_verdict(answer_relevance=1), gold_ids=["c2"]), "Low Answer Relevance"),
            (dict(generation_ok=True, refused=False, judge=self.ok_verdict(citation_correctness=2), gold_ids=["c2"]), "Context Selection / Citation"),
        ]
        for kwargs, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(label_failure(top_ids=top, **kwargs), expected)
                self.assertIn(expected, FAILURE_LABELS)

    def test_only_the_top_five_count_as_retrieved(self):
        verdict = self.ok_verdict()
        top = ["c1", "c2", "c3", "c4", "c5", "gold"]
        self.assertEqual(label_failure(True, False, verdict, top, ["gold"]), "Retrieval Failure")


def item(i, group=None, **overrides):
    record = {
        "item_id": f"q{i}",
        "group": group or f"g{i}",
        "generation_ok": True,
        "refused": False,
        "judge_status": "ok",
        "scores": dict(GOOD),
        "citations_in_context": True,
        "failure_label": "No Major Failure (Success)",
    }
    record.update(overrides)
    return record


class TestSummarizeGeneration(unittest.TestCase):
    def test_every_metric_has_an_interval(self):
        records = [item(i, scores=dict(GOOD, faithfulness=1 + i % 4, hallucination=i % 5 == 0)) for i in range(40)]
        summary = summarize_generation(records, n_resamples=300)
        for name, metric in summary["metrics"].items():
            with self.subTest(metric=name):
                self.assertIsNotNone(metric["low"])
                self.assertIsNotNone(metric["high"])
                self.assertLessEqual(metric["low"], metric["estimate"])
                self.assertGreaterEqual(metric["high"], metric["estimate"])
        self.assertAlmostEqual(summary["metrics"]["hallucination_rate"]["estimate"], 0.2)

    def test_system_errors_are_counted_but_not_scored(self):
        records = [item(i) for i in range(10)]
        records.append(item(10, generation_ok=False, scores=None, judge_status="skipped", failure_label="System Error (generation)"))
        records.append(item(11, judge_status="parse_error", scores=None, failure_label="System Error (judge)"))
        summary = summarize_generation(records, n_resamples=200)
        counts = summary["counts"]
        self.assertEqual((counts["total"], counts["generation_errors"], counts["judge_errors"], counts["judged"]), (12, 1, 1, 10))
        self.assertEqual(summary["metrics"]["faithfulness"]["n"], 10)
        self.assertTrue(any("5%" in w for w in summary["warnings"]))

    def test_refusals_are_reported_separately_from_quality(self):
        records = [item(i) for i in range(8)]
        records += [item(8 + i, refused=True, judge_status="skipped_refusal", scores=None) for i in range(2)]
        summary = summarize_generation(records, n_resamples=200)
        self.assertEqual(summary["counts"]["refused"], 2)
        self.assertEqual(summary["metrics"]["faithfulness"]["n"], 8)
        self.assertAlmostEqual(summary["metrics"]["refusal_rate"]["estimate"], 0.2)

    def test_failure_labels_have_counts_and_intervals(self):
        records = [item(i) for i in range(6)] + [item(6 + i, failure_label="Retrieval Failure") for i in range(4)]
        table = summarize_generation(records, n_resamples=100)["failure_labels"]
        self.assertEqual(table["denominator"], 10)
        self.assertEqual(table["labels"]["Retrieval Failure"]["count"], 4)
        self.assertAlmostEqual(table["labels"]["Retrieval Failure"]["share"], 0.4)
        self.assertIsNotNone(table["labels"]["Retrieval Failure"]["ci_low"])

    def test_small_samples_are_flagged(self):
        summary = summarize_generation([item(i) for i in range(5)], n_resamples=100)
        self.assertTrue(any("30" in w for w in summary["warnings"]))

    def test_empty_input_does_not_crash(self):
        summary = summarize_generation([], n_resamples=100)
        self.assertEqual(summary["counts"]["total"], 0)
        self.assertIsNone(summary["metrics"]["faithfulness"]["estimate"])

    def test_questions_from_one_judgment_use_a_cluster_bootstrap(self):
        records = [item(i, group="same-judgment") for i in range(10)]
        summary = summarize_generation(records, n_resamples=100)
        self.assertEqual(summary["metrics"]["faithfulness"]["method"], "cluster_bootstrap")


if __name__ == "__main__":
    unittest.main()

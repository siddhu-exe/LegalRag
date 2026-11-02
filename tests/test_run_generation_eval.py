"""Tests for scripts/run_generation_eval.py (plan task C3). Stub generator and judge; no network."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from legalrag.evaluation.judge import JudgeIndependenceError, LLMJudge
from legalrag.evaluation.runtime import JsonlCheckpoint
from legalrag.generation.client import GenerationError, GenerationResult

REFUSAL = "The provided judgment context does not contain sufficient information to answer this inquiry."
GOOD_SCORES = {
    "answer_relevance": 4, "faithfulness": 4, "citation_correctness": 4, "overall_score": 4,
    "unsupported_claim_rate": 0.0, "hallucination": False, "rationale": "ok",
}


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_generation_eval.py"
    spec = importlib.util.spec_from_file_location("run_generation_eval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


class StubGenerator:
    def __init__(self, behaviour=None):
        self.calls = 0
        self.behaviour = behaviour or (lambda n, prompt: "Anticipatory bail is available [Chunk ID: c0].")

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1024):
        self.calls += 1
        outcome = self.behaviour(self.calls, user_prompt)
        if isinstance(outcome, Exception):
            raise outcome
        return GenerationResult(text=outcome, status="success", prompt_tokens=300, completion_tokens=40)


class StubJudgeClient:
    def __init__(self, reply=None, model_name="judge-m"):
        self.model_name = model_name
        self.calls = 0
        self.reply = reply if reply is not None else json.dumps(GOOD_SCORES)

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1024):
        self.calls += 1
        return SimpleNamespace(text=self.reply, prompt_tokens=500, completion_tokens=60)


def retrieval_record(qid, cnrs=("G", "x", "y", "z", "w")):
    ids = [f"c{i}" for i in range(5)]
    context = [{"chunk_id": cid, "text": f"text of {cid}", "court_code": "27_1", "decision_date": "2020-01-01"} for cid in ids]
    ranked = {"chunk_ids": ids, "cnrs": list(cnrs)}
    return {
        "item_id": qid, "status": "ok",
        "systems": {name: ranked for name in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")},
        "context": {name: context for name in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")},
    }


def item(qid, gold="G"):
    return {"question_id": qid, "question": f"Question {qid}?", "gold_cnrs": [gold], "gold_chunk_ids": [], "group": f"grp{qid}"}


def make_judge(reply=None, generator_model="gen-m"):
    return LLMJudge(StubJudgeClient(reply), generator_model=generator_model)


class TestProcessItem(unittest.TestCase):
    def run_item(self, generator=None, judge=None, **kwargs):
        return script.process_item(
            item("1", **kwargs), retrieval_record("1"), "hybrid_rerank",
            generator or StubGenerator(), judge or make_judge(), "gen-m",
        )

    def test_successful_item(self):
        record = self.run_item()
        self.assertEqual((record["status"], record["judge_status"]), ("ok", "ok"))
        self.assertEqual(record["failure_label"], "No Major Failure (Success)")
        self.assertEqual(record["scores"]["faithfulness"], 4)
        self.assertTrue(record["citations_in_context"])
        self.assertTrue(record["retrieval_hit"])
        self.assertEqual(record["tokens"], {"gen_prompt": 300, "gen_completion": 40, "judge_prompt": 500, "judge_completion": 60})
        self.assertEqual((record["generator_model"], record["judge_model"]), ("gen-m", "judge-m"))

    def test_generation_error_is_a_system_error_and_skips_the_judge(self):
        judge = make_judge()
        generator = StubGenerator(lambda n, p: GenerationError("raw secret text", 502, "RateLimitError"))
        record = self.run_item(generator, judge)
        self.assertEqual((record["status"], record["generation_ok"], record["gen_error_type"]), ("error", False, "RateLimitError"))
        self.assertEqual(record["failure_label"], "System Error (generation)")
        self.assertEqual(judge.client.calls, 0)
        self.assertNotIn("raw secret text", json.dumps(record))

    def test_unexpected_generator_exception_keeps_only_the_class_name(self):
        record = self.run_item(StubGenerator(lambda n, p: RuntimeError("raw provider body")))
        self.assertEqual(record["gen_error_type"], "RuntimeError")
        self.assertNotIn("raw provider body", json.dumps(record))

    def test_refusal_skips_the_judge_and_is_labelled_by_retrieval(self):
        judge = make_judge()
        record = self.run_item(StubGenerator(lambda n, p: REFUSAL), judge)
        self.assertTrue(record["refused"])
        self.assertEqual((record["status"], record["judge_status"]), ("ok", "skipped_refusal"))
        self.assertEqual(record["failure_label"], "Wrongful Refusal (evidence retrieved)")
        self.assertEqual(judge.client.calls, 0)
        missed = self.run_item(StubGenerator(lambda n, p: REFUSAL), judge, gold="NOT-RETRIEVED")
        self.assertEqual(missed["failure_label"], "Refusal (evidence not retrieved)")

    def test_unreadable_judge_reply_is_a_judge_system_error(self):
        record = self.run_item(judge=make_judge("this is not json"))
        self.assertEqual((record["status"], record["judge_status"]), ("error", "parse_error"))
        self.assertEqual(record["failure_label"], "System Error (judge)")
        self.assertIsNone(record["scores"])

    def test_citation_outside_the_context_is_flagged(self):
        record = self.run_item(StubGenerator(lambda n, p: "Held so [Chunk ID: c0] and [Chunk ID: invented]."))
        self.assertFalse(record["citations_in_context"])
        self.assertEqual(record["cited_chunk_ids"], ["c0", "invented"])

    def test_answer_without_citations_has_no_citation_flag(self):
        record = self.run_item(StubGenerator(lambda n, p: "Bail may be granted."))
        self.assertIsNone(record["citations_in_context"])

    def test_retrieval_failure_label_when_gold_judgment_not_in_top5(self):
        record = self.run_item(gold="ELSEWHERE")
        self.assertEqual(record["failure_label"], "Retrieval Failure")
        self.assertFalse(record["retrieval_hit"])

    def test_chunk_level_gold_is_used_when_no_gold_cnr(self):
        chunk_item = dict(item("1"), gold_cnrs=[], gold_chunk_ids=["c2"])
        record = script.process_item(chunk_item, retrieval_record("1"), "bm25", StubGenerator(), make_judge(), "gen-m")
        self.assertTrue(record["retrieval_hit"])


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        questions = [{"question_id": str(i), "question": f"Question {i}?", "source_cnr": "G" if i % 2 == 0 else "H"} for i in range(6)]
        (self.dir / "bench.json").write_text(json.dumps(questions), encoding="utf-8")
        retrievals = JsonlCheckpoint(self.dir / "retrieval.jsonl")
        for q in questions:
            retrievals.append(retrieval_record(q["question_id"], cnrs=("G", "H", "x", "y", "z")))
        self.out = self.dir / "out"

    def argv(self, *extra):
        return ["--benchmark", str(self.dir / "bench.json"), "--retrieval-results", str(self.dir / "retrieval.jsonl"),
                "--out-dir", str(self.out), "--judge-model", "judge-m", "--generator-model", "gen-m",
                "--n-resamples", "200", *extra]

    def clients(self, generator=None, judge_reply=None):
        generator = generator or StubGenerator()
        return generator, make_judge(judge_reply)

    def test_full_run_writes_results_with_intervals(self):
        generator, judge = self.clients()
        script.main(self.argv("--yes"), clients=(generator, judge))
        self.assertEqual((generator.calls, judge.client.calls), (6, 6))
        metrics = json.loads((self.out / "generation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["summary"]["counts"]["judged"], 6)
        for name, interval in metrics["summary"]["metrics"].items():
            self.assertIsNotNone(interval["low"], name)
        self.assertEqual(metrics["run"]["judge_model"], "judge-m")
        text = (self.out / "generation_metrics.md").read_text(encoding="utf-8")
        self.assertIn("Judge: `judge-m`", text)
        self.assertIn("Warning", text)  # fewer than 30 judged answers

    def test_rerun_resumes_without_new_calls(self):
        script.main(self.argv("--yes"), clients=self.clients())
        generator, judge = self.clients()
        script.main(self.argv("--yes"), clients=(generator, judge))
        self.assertEqual((generator.calls, judge.client.calls), (0, 0))

    def test_resume_after_a_crash_only_runs_the_missing_items(self):
        calls = {"n": 0}

        class Crash(BaseException):
            pass

        def behaviour(n, prompt):
            calls["n"] += 1
            if calls["n"] == 4:
                raise Crash()
            return "Held [Chunk ID: c0]."

        with self.assertRaises(Crash):
            script.main(self.argv("--yes"), clients=self.clients(StubGenerator(behaviour)))
        self.assertEqual(len(JsonlCheckpoint(self.out / "generation_per_item.jsonl").records), 3)
        generator, judge = self.clients()
        script.main(self.argv("--yes"), clients=(generator, judge))
        self.assertEqual(generator.calls, 3)

    def test_retry_errors_repeats_only_failed_items(self):
        flaky = StubGenerator(lambda n, p: GenerationError("x", 502, "APIConnectionError") if n == 2 else "Held [Chunk ID: c0].")
        script.main(self.argv("--yes"), clients=self.clients(flaky))
        saved = JsonlCheckpoint(self.out / "generation_per_item.jsonl")
        self.assertEqual(sum(1 for r in saved.values() if r["status"] == "error"), 1)
        generator, judge = self.clients()
        script.main(self.argv("--yes", "--retry-errors"), clients=(generator, judge))
        self.assertEqual(generator.calls, 1)
        self.assertTrue(all(r["status"] == "ok" for r in JsonlCheckpoint(self.out / "generation_per_item.jsonl").values()))

    def test_dry_run_sends_nothing_and_needs_no_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            script.main(self.argv("--dry-run"))
        self.assertFalse((self.out / "generation_per_item.jsonl").exists())

    def test_cost_estimate_is_printed_before_any_call(self):
        events = []
        generator = StubGenerator(lambda n, p: events.append("call") or "Held [Chunk ID: c0].")
        with mock.patch("builtins.print", side_effect=lambda *a, **k: events.append("print:" + " ".join(map(str, a))[:40])):
            script.main(self.argv("--yes"), clients=self.clients(generator))
        first_call = events.index("call")
        self.assertTrue(any(e.startswith("print:Estimated usage") for e in events[:first_call]))

    def test_prices_turn_the_estimate_into_dollars(self):
        buffer = []
        with mock.patch("builtins.print", side_effect=lambda *a, **k: buffer.append(" ".join(map(str, a)))):
            script.main(self.argv("--dry-run", "--gen-price-in", "1", "--gen-price-out", "2",
                                  "--judge-price-in", "1", "--judge-price-out", "2"))
        self.assertTrue(any("$" in line and "total" in line for line in "\n".join(buffer).splitlines()))

    def test_non_interactive_run_without_yes_aborts_before_any_call(self):
        generator, judge = self.clients()
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(SystemExit):
                script.main(self.argv(), clients=(generator, judge))
        self.assertEqual(generator.calls, 0)

    def test_judge_equal_to_generator_stops_immediately(self):
        generator, judge = self.clients()
        with self.assertRaises(JudgeIndependenceError):
            script.main(self.argv("--yes", "--judge-model", "gen-m"), clients=(generator, judge))
        self.assertEqual(generator.calls, 0)

    def test_changing_the_judge_on_resume_is_refused_even_when_nothing_is_left_to_run(self):
        script.main(self.argv("--yes"), clients=self.clients())
        argv = self.argv("--yes")
        argv[argv.index("--judge-model") + 1] = "other-judge"
        generator, judge = self.clients()
        with self.assertRaises(SystemExit):
            script.main(argv, clients=(generator, judge))
        self.assertEqual(generator.calls, 0)
        with self.assertRaises(SystemExit):
            script.main(argv + ["--retry-errors"], clients=(generator, judge))

    def test_report_names_the_models_that_produced_the_items(self):
        script.main(self.argv("--yes"), clients=self.clients())
        argv = self.argv("--metrics-only")
        argv[argv.index("--judge-model") + 1] = "other-judge"
        script.main(argv)
        metrics = json.loads((self.out / "generation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["run"]["judge_model"], "judge-m")

    def test_missing_retrieval_results_are_skipped_not_invented(self):
        questions = json.loads((self.dir / "bench.json").read_text(encoding="utf-8"))
        questions.append({"question_id": "77", "question": "No retrieval?", "source_cnr": "G"})
        (self.dir / "bench.json").write_text(json.dumps(questions), encoding="utf-8")
        generator, judge = self.clients()
        script.main(self.argv("--yes"), clients=(generator, judge))
        self.assertEqual(generator.calls, 6)

    def test_label_sheet_hides_judge_scores_and_agreement_is_reported(self):
        script.main(self.argv("--yes", "--export-label-sheet", "4"), clients=self.clients())
        sheet = json.loads((self.out / "generation_label_sheet.json").read_text(encoding="utf-8"))
        self.assertEqual(len(sheet), 4)
        self.assertNotIn("scores", sheet[0])
        self.assertNotIn("judge", json.dumps(sheet).lower())
        self.assertIsNone(sheet[0]["faithfulness"])

        for row in sheet:  # a stand-in for the owner's own labels, used only to exercise the code path
            row.update(answer_relevance=4, faithfulness=3, citation_correctness=4, hallucination=False)
        labels = self.dir / "labels.json"
        labels.write_text(json.dumps(sheet), encoding="utf-8")
        script.main(self.argv("--metrics-only", "--human-labels", str(labels)))
        metrics = json.loads((self.out / "generation_metrics.json").read_text(encoding="utf-8"))
        agreement = metrics["judge_agreement"]
        self.assertEqual(agreement["labelled_items"], 4)
        self.assertEqual(agreement["metrics"]["answer_relevance"]["percent_agreement"], 1.0)
        self.assertEqual(agreement["metrics"]["faithfulness"]["percent_agreement"], 0.0)
        self.assertIn("kappa", agreement["metrics"]["hallucination"])
        self.assertIn("Judge agreement", (self.out / "generation_metrics.md").read_text(encoding="utf-8"))

    def test_metrics_only_makes_no_calls(self):
        script.main(self.argv("--yes"), clients=self.clients())
        script.main(self.argv("--metrics-only"))
        self.assertTrue((self.out / "generation_metrics.json").exists())


class TestAgreement(unittest.TestCase):
    def test_skips_unlabelled_and_unknown_items(self):
        records = [{"item_id": "1", "scores": dict(GOOD_SCORES)}, {"item_id": "2", "scores": dict(GOOD_SCORES, faithfulness=2)}]
        human = [
            {"item_id": "1", "faithfulness": 4, "answer_relevance": None},
            {"item_id": "2", "faithfulness": 2},
            {"item_id": "404", "faithfulness": 1},
        ]
        report = script.judge_agreement(records, human)
        self.assertEqual(report["metrics"]["faithfulness"]["n"], 2)
        self.assertNotIn("answer_relevance", report["metrics"])


if __name__ == "__main__":
    unittest.main()

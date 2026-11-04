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

    def test_metrics_only_checks_the_models_too(self):
        script.main(self.argv("--yes"), clients=self.clients())
        other_judge = self.argv("--metrics-only")
        other_judge[other_judge.index("--judge-model") + 1] = "other-judge"
        with self.assertRaises(SystemExit) as ctx:
            script.main(other_judge)
        self.assertIn("judge_model", str(ctx.exception))
        other_generator = self.argv("--metrics-only")
        other_generator[other_generator.index("--generator-model") + 1] = "other-gen"
        with self.assertRaises(SystemExit) as ctx:
            script.main(other_generator)
        self.assertIn("generator_model", str(ctx.exception))
        script.main(self.argv("--metrics-only"))  # the same settings still work

    def test_forced_metrics_only_names_the_models_that_produced_the_items(self):
        script.main(self.argv("--yes"), clients=self.clients())
        argv = self.argv("--metrics-only", "--force-resume")
        argv[argv.index("--judge-model") + 1] = "other-judge"
        script.main(argv)
        metrics = json.loads((self.out / "generation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["run"]["judge_model"], "judge-m")
        self.assertTrue(metrics["run"]["check_skipped_by_force_resume"])

    def test_metrics_only_without_a_saved_config_is_refused(self):
        with self.assertRaises(SystemExit) as ctx:
            script.main(self.argv("--metrics-only"))
        self.assertIn("No saved settings", str(ctx.exception))
        JsonlCheckpoint(self.out / "generation_per_item.jsonl").append(
            {"item_id": "0", "status": "error", "generation_ok": False, "generator_model": "gen-m", "judge_model": "judge-m"}
        )
        with self.assertRaises(SystemExit):
            script.main(self.argv("--metrics-only"))  # a results file alone is not enough
        script.main(self.argv("--metrics-only", "--force-resume"))  # explicit override

    def test_changed_retrieval_results_block_resume_and_metrics(self):
        script.main(self.argv("--yes"), clients=self.clients())
        retrieval = self.dir / "retrieval.jsonl"
        original = retrieval.read_bytes()
        JsonlCheckpoint(retrieval).append(retrieval_record("extra"))  # retrieval was re-run or extended
        for extra in (("--yes",), ("--metrics-only",), ("--dry-run",)):
            with self.subTest(extra=extra):
                generator, judge = self.clients()
                with self.assertRaises(SystemExit) as ctx:
                    script.main(self.argv(*extra), clients=(generator, judge))
                self.assertIn("retrieval_results_sha256", str(ctx.exception))
                self.assertEqual(generator.calls, 0)
        retrieval.write_bytes(original)
        script.main(self.argv("--metrics-only"))  # restored file matches again

    def test_changed_benchmark_blocks_resume(self):
        script.main(self.argv("--yes"), clients=self.clients())
        bench = self.dir / "bench.json"
        questions = json.loads(bench.read_text(encoding="utf-8"))
        questions[0]["question"] = "A different question?"
        bench.write_text(json.dumps(questions), encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            script.main(self.argv("--metrics-only"))
        self.assertIn("benchmark_sha256", str(ctx.exception))

    def test_changed_prompts_block_resume(self):
        script.main(self.argv("--yes"), clients=self.clients())
        with mock.patch.object(script, "SYSTEM_PROMPT", script.SYSTEM_PROMPT + " Be brief."):
            with self.assertRaises(SystemExit) as ctx:
                script.main(self.argv("--metrics-only"))
        self.assertIn("generator_prompt_sha256", str(ctx.exception))
        with mock.patch.object(script, "JUDGE_SYSTEM_PROMPT", script.JUDGE_SYSTEM_PROMPT + " Be harsh."):
            with self.assertRaises(SystemExit) as ctx:
                script.main(self.argv("--metrics-only"))
        self.assertIn("judge_prompt_sha256", str(ctx.exception))

    def test_saved_config_records_every_identity_and_the_retrieval_provenance(self):
        (self.dir / "retrieval_run_config.json").write_text(json.dumps({"artifacts": {"bm25.pkl": "abc"}}), encoding="utf-8")
        script.main(self.argv("--yes"), clients=self.clients())
        saved = json.loads((self.out / "generation_run_config.json").read_text(encoding="utf-8"))
        for key in ("benchmark_sha256", "retrieval_results_sha256", "system", "generator_model", "judge_model",
                    "generator_prompt_sha256", "judge_prompt_sha256"):
            self.assertTrue(saved[key], key)
        self.assertEqual(len(saved["retrieval_results_sha256"]), 64)
        self.assertEqual(saved["retrieval_run_config"], {"artifacts": {"bm25.pkl": "abc"}})
        metrics = json.loads((self.out / "generation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["run"]["identity"]["retrieval_results_sha256"], saved["retrieval_results_sha256"])

    def test_different_system_is_refused(self):
        script.main(self.argv("--yes"), clients=self.clients())
        with self.assertRaises(SystemExit) as ctx:
            script.main(self.argv("--metrics-only", "--system", "bm25"))
        self.assertIn("system", str(ctx.exception))

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
    # Same six items as the hand-computed example in test_stats.py: judge labels 1,1,3,3,4,4 against
    # human labels 1,3,3,4,4,3. Nobody used a 2, yet the 1-4 scale is used: quadratic kappa is 5/8.
    def test_weighted_kappa_uses_the_fixed_one_to_four_scale(self):
        judge = [1, 1, 3, 3, 4, 4]
        human_scores = [1, 3, 3, 4, 4, 3]
        records = [{"item_id": str(i), "scores": dict(GOOD_SCORES, faithfulness=v)} for i, v in enumerate(judge)]
        human = [{"item_id": str(i), "faithfulness": v} for i, v in enumerate(human_scores)]
        entry = script.judge_agreement(records, human)["metrics"]["faithfulness"]
        self.assertAlmostEqual(entry["kappa_quadratic_weighted"]["estimate"], 5 / 8)

    def test_a_human_label_outside_one_to_four_stops_with_a_clear_message(self):
        records = [{"item_id": "1", "scores": dict(GOOD_SCORES)}, {"item_id": "2", "scores": dict(GOOD_SCORES)}]
        human = [{"item_id": "1", "faithfulness": 4}, {"item_id": "2", "faithfulness": 5}]
        with self.assertRaises(SystemExit) as ctx:
            script.judge_agreement(records, human)
        self.assertIn("faithfulness", str(ctx.exception))
        self.assertIn("outside the scale", str(ctx.exception))

    def test_hallucination_labels_need_no_scale(self):
        records = [{"item_id": str(i), "scores": dict(GOOD_SCORES, hallucination=bool(i % 2))} for i in range(6)]
        human = [{"item_id": str(i), "hallucination": bool(i % 2)} for i in range(6)]
        entry = script.judge_agreement(records, human)["metrics"]["hallucination"]
        self.assertAlmostEqual(entry["kappa"]["estimate"], 1.0)

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


class TestSettingsAreRespected(unittest.TestCase):
    """The model and key come from get_settings(), so .env applies (review fix 4)."""

    def setUp(self):
        from legalrag.api.config import get_settings

        self.get_settings = get_settings
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        self.addCleanup(os.chdir, self.cwd)
        os.chdir(self.tmp.name)  # Settings reads ".env" from the working directory
        patched = mock.patch.dict(os.environ, {}, clear=False)
        patched.start()
        self.addCleanup(patched.stop)
        for name in ("GROQ_MODEL_NAME", "GROQ_API_KEY", "GROQ_REQUEST_TIMEOUT", "GROQ_MAX_RETRIES"):
            os.environ.pop(name, None)

    def args(self, *extra):
        return script.parse_args(["--benchmark", "b.json", "--retrieval-results", "r.jsonl",
                                  "--judge-model", "judge-m", *extra])

    def write_env(self, text):
        Path(".env").write_text(text, encoding="utf-8")
        self.get_settings.cache_clear()

    def test_generator_model_comes_from_dot_env_when_the_flag_is_absent(self):
        self.write_env("GROQ_MODEL_NAME=env-file-model\n")
        self.assertEqual(script.resolve_generator_model(self.args()), "env-file-model")

    def test_environment_variable_beats_dot_env(self):
        self.write_env("GROQ_MODEL_NAME=env-file-model\n")
        os.environ["GROQ_MODEL_NAME"] = "process-model"
        self.assertEqual(script.resolve_generator_model(self.args()), "process-model")

    def test_the_flag_beats_settings(self):
        self.write_env("GROQ_MODEL_NAME=env-file-model\n")
        self.assertEqual(script.resolve_generator_model(self.args("--generator-model", "flag-model")), "flag-model")

    def test_default_matches_the_api_default_when_nothing_is_configured(self):
        from legalrag.api.config import Settings

        self.assertEqual(script.resolve_generator_model(self.args()), Settings(_env_file=None).groq_model_name)

    def test_clients_get_key_timeout_and_retries_from_dot_env(self):
        self.write_env("GROQ_API_KEY=key-from-dot-env\nGROQ_REQUEST_TIMEOUT=12\nGROQ_MAX_RETRIES=4\nGROQ_MODEL_NAME=env-file-model\n")
        args = self.args()
        args.generator_model = script.resolve_generator_model(args)
        created = []

        class FakeClient:
            def __init__(self, **kwargs):
                created.append(kwargs)
                self.model_name = kwargs["model_name"]

        with mock.patch("legalrag.generation.client.LegalGenerationClient", FakeClient):
            generator, judge = script.build_clients(args)
        self.assertEqual([c["model_name"] for c in created], ["env-file-model", "judge-m"])
        for kwargs in created:
            self.assertEqual((kwargs["api_key"], kwargs["request_timeout"], kwargs["max_retries"]), ("key-from-dot-env", 12.0, 4))
        self.assertEqual(judge.generator_model, "env-file-model")

    def test_missing_key_gives_a_clear_message_not_a_traceback(self):
        args = self.args("--generator-model", "gen-m")
        with self.assertRaises(SystemExit) as ctx:
            script.build_clients(args)
        self.assertIn("Groq API key is required", str(ctx.exception))

    def test_main_uses_the_dot_env_model_in_the_saved_records_and_independence_check(self):
        self.write_env("GROQ_MODEL_NAME=judge-m\n")  # same as the judge: must be refused
        questions = [{"question_id": "1", "question": "Q?", "source_cnr": "G"}]
        Path("b.json").write_text(json.dumps(questions), encoding="utf-8")
        JsonlCheckpoint(Path("r.jsonl")).append(retrieval_record("1"))
        with self.assertRaises(JudgeIndependenceError):
            script.main(["--benchmark", "b.json", "--retrieval-results", "r.jsonl", "--out-dir", "out",
                         "--judge-model", "judge-m", "--yes"])

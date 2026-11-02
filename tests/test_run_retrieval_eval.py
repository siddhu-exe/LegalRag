"""Tests for scripts/run_retrieval_eval.py (plan task C3). Stub pipeline only; no models or artifacts."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from legalrag.evaluation.metrics import calculate_recall_at_k
from legalrag.evaluation.runtime import JsonlCheckpoint


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_retrieval_eval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


def write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def question(qid, **overrides):
    base = {"question_id": qid, "question": f"Question {qid}?", "source_cnr": f"CNR{qid}"}
    base.update(overrides)
    return base


class TestBenchmarkLoading(unittest.TestCase):
    def load(self, data, as_jsonl=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            if as_jsonl:
                path.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
            else:
                write_json(path, data)
            return script.load_benchmark(path)

    def test_source_cnr_becomes_gold_and_cluster(self):
        item = self.load([question("1")])[0]
        self.assertEqual((item["gold_cnrs"], item["group"]), (["CNR1"], "CNR1"))

    def test_explicit_gold_fields_win(self):
        item = self.load([question("1", gold_cnrs=["A", "B"], gold_chunk_ids=["c1"])])[0]
        self.assertEqual((item["gold_cnrs"], item["gold_chunk_ids"], item["group"]), (["A", "B"], ["c1"], "CNR1"))

    def test_jsonl_and_wrapped_formats(self):
        self.assertEqual(len(self.load([question("1"), question("2")], as_jsonl=True)), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(Path(tmp) / "b.json", {"questions": [question("1")]})
            self.assertEqual(len(script.load_benchmark(path)), 1)

    def test_invalid_benchmarks_are_rejected(self):
        bad = [
            [],
            [question("1"), question("1")],
            [{"question_id": "1", "question": "q"}],  # no gold
            [{"question": "q", "source_cnr": "A"}],  # no id
            [{"question_id": "1", "question": "  ", "source_cnr": "A"}],
            ["not an object"],
        ]
        for data in bad:
            with self.subTest(data=data):
                with self.assertRaises(SystemExit):
                    self.load(data)

    def test_sample_is_seeded_random_and_keeps_order(self):
        items = [{"question_id": str(i)} for i in range(100)]
        first = script.sample_benchmark(items, 10, seed=1)
        self.assertEqual(first, script.sample_benchmark(items, 10, seed=1))
        self.assertNotEqual(first, items[:10])  # not simply the first rows
        ids = [int(i["question_id"]) for i in first]
        self.assertEqual(ids, sorted(ids))
        self.assertIs(script.sample_benchmark(items, None, 1), items)


class TestMetricFunctions(unittest.TestCase):
    def test_first_hit_rank(self):
        self.assertEqual(script.first_hit_rank(["a", "b", "c"], ["c", "z"]), 3)
        self.assertIsNone(script.first_hit_rank(["a"], ["z"]))
        self.assertIsNone(script.first_hit_rank([], ["z"]))

    def test_metric_values(self):
        values = script.metric_values(3, [1, 5, 10])
        self.assertEqual((values["recall@1"], values["recall@5"], values["recall@10"]), (0.0, 1.0, 1.0))
        self.assertAlmostEqual(values["mrr@10"], 1 / 3)
        miss = script.metric_values(None, [1, 5])
        self.assertEqual((miss["recall@5"], miss["mrr@10"]), (0.0, 0.0))
        self.assertEqual(script.metric_values(11, [50])["mrr@10"], 0.0)

    def test_agrees_with_the_existing_v1_recall_function(self):
        ranked = ["c9", "c2", "c7", "c1", "c4", "c8"]
        gold = ["c4", "c3"]
        for k in (1, 3, 5, 6):
            self.assertEqual(
                script.metric_values(script.first_hit_rank(ranked, gold), [k])[f"recall@{k}"],
                calculate_recall_at_k(ranked, gold, k),
            )


def synthetic_records(n=40, bm25_rank=1, hybrid_rank=7):
    """bm25 finds the gold judgment at bm25_rank, hybrid_rrf at hybrid_rank."""
    def ranked(rank, qid):
        cnrs = [f"other{i}" for i in range(1, 51)]
        cnrs[rank - 1] = f"CNR{qid}"
        return {"chunk_ids": [f"ch{i}" for i in range(50)], "cnrs": cnrs}

    return {
        str(q): {"item_id": str(q), "status": "ok",
                 "systems": {"bm25": ranked(bm25_rank, q), "hybrid_rrf": ranked(hybrid_rank, q)}}
        for q in range(n)
    }


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.items = [dict(question("%d" % q, question_type="statute" if q % 2 else "facts"),
                           gold_cnrs=["CNR%d" % q], gold_chunk_ids=[], group="CNR%d" % q)
                      for q in range(40)]

    def run_eval(self, records, **kwargs):
        return script.evaluate(self.items, records, ks=(1, 5, 10), systems=("bm25", "hybrid_rrf"),
                               baseline="hybrid_rrf", n_resamples=500, seed=1, **kwargs)

    def test_known_outcome_and_intervals_for_every_metric(self):
        result = self.run_eval(synthetic_records())
        level = result["levels"]["judgment"]
        bm25, hybrid = level["systems"]["bm25"], level["systems"]["hybrid_rrf"]
        self.assertEqual((bm25["recall@1"]["estimate"], hybrid["recall@1"]["estimate"]), (1.0, 0.0))
        self.assertEqual((bm25["recall@5"]["estimate"], hybrid["recall@5"]["estimate"]), (1.0, 0.0))
        self.assertEqual((bm25["recall@10"]["estimate"], hybrid["recall@10"]["estimate"]), (1.0, 1.0))
        self.assertAlmostEqual(hybrid["mrr@10"]["estimate"], 1 / 7)
        for system in level["systems"].values():
            for metric in system.values():
                self.assertIsNotNone(metric["low"])
                self.assertIsNotNone(metric["high"])
        self.assertNotIn("chunk", result["levels"])

    def test_clear_win_is_reported_and_tie_is_not(self):
        level = self.run_eval(synthetic_records())["levels"]["judgment"]
        by_metric = {c["metric"]: c for c in level["comparisons"]}
        self.assertEqual(by_metric["recall@5"]["verdict"], "bm25 better")
        self.assertEqual(by_metric["recall@10"]["verdict"], "no clear difference")
        self.assertEqual(by_metric["recall@5"]["mcnemar"]["a_only"], 40)
        self.assertGreaterEqual(by_metric["recall@5"]["p_value_holm"], by_metric["recall@5"]["p_value"])

    def test_identical_systems_show_no_difference(self):
        level = self.run_eval(synthetic_records(bm25_rank=3, hybrid_rank=3))["levels"]["judgment"]
        self.assertTrue(all(c["verdict"] == "no clear difference" for c in level["comparisons"]))

    def test_system_errors_are_excluded_and_listed_not_counted_as_misses(self):
        records = synthetic_records()
        records["3"] = {"item_id": "3", "status": "error", "error_type": "RuntimeError"}
        del records["4"]
        result = self.run_eval(records)
        self.assertEqual((result["n_scored"], result["retrieval_errors"], result["not_run"]), (38, ["3"], ["4"]))
        self.assertEqual(result["levels"]["judgment"]["n"], 38)

    def test_breakdown_by_question_type(self):
        result = self.run_eval(synthetic_records(), breakdown_fields=("question_type", "bench"))
        fields = result["breakdown"]["fields"]
        self.assertEqual(set(fields), {"question_type"})  # no 'bench' values in this benchmark
        self.assertEqual(fields["question_type"]["facts"]["n"], 20)
        self.assertEqual(fields["question_type"]["facts"]["systems"]["bm25"]["estimate"], 1.0)

    def test_chunk_level_is_scored_when_chunk_gold_is_given(self):
        items = [dict(i, gold_chunk_ids=["ch3"]) for i in self.items]
        result = script.evaluate(items, synthetic_records(), ks=(1, 5), systems=("bm25", "hybrid_rrf"),
                                 baseline="hybrid_rrf", n_resamples=200)
        self.assertEqual(set(result["levels"]), {"judgment", "chunk"})
        self.assertEqual(result["levels"]["chunk"]["systems"]["bm25"]["recall@5"]["estimate"], 1.0)
        self.assertEqual(result["levels"]["chunk"]["systems"]["bm25"]["recall@1"]["estimate"], 0.0)

    def test_unknown_baseline_raises(self):
        with self.assertRaises(ValueError):
            script.evaluate(self.items, synthetic_records(), systems=("bm25",), baseline="hybrid_rrf")

    def test_markdown_report(self):
        result = self.run_eval(synthetic_records())
        text = script.render_markdown(result)
        self.assertIn("Judgment-level recall", text)
        self.assertIn("p (Holm)", text)
        self.assertIn("bm25 better", text)
        self.assertIn("1.000 [", text)


class TestEndToEndOnStubPipeline(unittest.TestCase):
    def benchmark(self):
        return [
            question("1", question="anticipatory bail Section 438", gold_cnrs=["STUB010000012020_1"], question_type="statute"),
            question("2", question="cheque dishonour notice", gold_cnrs=["STUB010000022021_1"], question_type="statute"),
            question("3", question="a question with an unreachable judgment", gold_cnrs=["NOT-IN-STUB"], question_type="facts"),
        ]

    def test_stub_run_saves_every_item_and_resumes_without_rework(self):
        pipeline, lookup = script.load_pipeline("local_stub")
        items = script.load_benchmark(self._write(self.benchmark()))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = JsonlCheckpoint(Path(tmp) / "r.jsonl")
            script.run_retrieval(items, pipeline, lookup, checkpoint, depth=50, log=lambda *_: None)
            self.assertEqual(len(checkpoint.records), 3)
            record = checkpoint.records["1"]
            self.assertEqual(record["status"], "ok")
            self.assertEqual(set(record["systems"]), set(script.SYSTEMS))
            self.assertEqual(len(record["context"]["hybrid_rerank"]), min(5, len(record["systems"]["hybrid_rerank"]["chunk_ids"])))
            self.assertIn("text", record["context"]["bm25"][0])
            self.assertIn("rerank_ms", record["latency_ms"])

            class Exploding:
                def __getattr__(self, name):
                    raise AssertionError("retrieval was re-run for a finished question")

            script.run_retrieval(items, Exploding(), lookup, JsonlCheckpoint(Path(tmp) / "r.jsonl"), 50, log=lambda *_: None)

    def test_retrieval_fault_is_saved_as_an_error_and_retried_on_request(self):
        pipeline, lookup = script.load_pipeline("local_stub")
        items = script.load_benchmark(self._write(self.benchmark()[:1]))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = JsonlCheckpoint(Path(tmp) / "r.jsonl")

            class Broken:
                bm25 = pipeline.bm25
                dense = pipeline.dense
                class reranker:
                    @staticmethod
                    def rerank(**kwargs):
                        raise RuntimeError("model crashed")

            script.run_retrieval(items, Broken, lookup, checkpoint, 50, log=lambda *_: None)
            self.assertEqual(checkpoint.records["1"], {"item_id": "1", "status": "error", "error_type": "RuntimeError"})
            script.run_retrieval(items, pipeline, lookup, checkpoint, 50, retry_errors=True, log=lambda *_: None)
            self.assertEqual(checkpoint.records["1"]["status"], "ok")

    def test_empty_reranker_result_is_a_system_error_not_a_silent_fallback(self):
        pipeline, lookup = script.load_pipeline("local_stub")

        class EmptyReranker:
            @staticmethod
            def rerank(**kwargs):
                return []

        class Pipeline:
            bm25 = pipeline.bm25
            dense = pipeline.dense
            reranker = EmptyReranker

        with self.assertRaisesRegex(RuntimeError, "reranker returned no results"):
            script.retrieve_all_systems(Pipeline, lookup, "anticipatory bail", 5)

        items = script.load_benchmark(self._write(self.benchmark()[:2]))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = JsonlCheckpoint(Path(tmp) / "r.jsonl")
            script.run_retrieval(items, Pipeline, lookup, checkpoint, 5, log=lambda *_: None)
            self.assertEqual({r["status"] for r in checkpoint.values()}, {"error"})
            result = script.evaluate(items, checkpoint.records, ks=(1, 5), n_resamples=100)
            self.assertEqual(sorted(result["retrieval_errors"]), ["1", "2"])
            self.assertEqual(result["n_scored"], 0)
            self.assertEqual(result["levels"], {})  # nothing is scored as a miss

    def test_no_candidate_metadata_also_raises(self):
        pipeline, lookup = script.load_pipeline("local_stub")

        class UnknownIds:
            @staticmethod
            def search(query, top_k=50):
                return [("id-not-in-the-chunk-table", 1.0)]

        class Pipeline:
            bm25 = UnknownIds
            dense = UnknownIds
            reranker = pipeline.reranker

        with self.assertRaisesRegex(RuntimeError, "reranker returned no results"):
            script.retrieve_all_systems(Pipeline, lookup, "anything", 5)

    def test_main_writes_metrics_files_and_metrics_only_reuses_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bench = write_json(tmp / "bench.json", self.benchmark())
            out = tmp / "out"
            argv = ["--benchmark", str(bench), "--out-dir", str(out), "--pipeline", "local_stub",
                    "--ks", "1", "5", "--depth", "5", "--n-resamples", "200"]
            script.main(argv)
            metrics = json.loads((out / "retrieval_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["n_scored"], 3)
            self.assertTrue((out / "retrieval_metrics.md").exists())
            self.assertTrue((out / "retrieval_run_config.json").exists())
            script.main(argv + ["--metrics-only"])

    def test_resume_with_a_different_benchmark_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "out"
            base = ["--out-dir", str(out), "--pipeline", "local_stub", "--ks", "1", "--depth", "5", "--n-resamples", "100"]
            script.main(["--benchmark", str(write_json(tmp / "a.json", self.benchmark()))] + base)
            with self.assertRaises(SystemExit):
                script.main(["--benchmark", str(write_json(tmp / "b.json", self.benchmark()[:2]))] + base)

    def test_depth_smaller_than_largest_k_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = write_json(Path(tmp) / "b.json", self.benchmark())
            with self.assertRaises(SystemExit):
                script.main(["--benchmark", str(bench), "--out-dir", tmp, "--pipeline", "local_stub",
                             "--ks", "1", "50", "--depth", "10"])

    def _write(self, data):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return write_json(Path(self._tmp.name) / "bench.json", data)


if __name__ == "__main__":
    unittest.main()

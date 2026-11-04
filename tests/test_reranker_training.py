"""
Tests for reranker training (plan task C10): the split by judgment and its leakage guard (pure Python), the
example file checks, the ranking metrics, and training end to end on a tiny randomly initialised BERT created in
a temporary folder (no download, no real data). The training tests skip if PyTorch or transformers is missing.
"""

import contextlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_onnx_models import HAVE_EXPORT_STACK, make_tiny_model

from legalrag.training.reranker import (
    ExampleError,
    TrainingConfig,
    benchmark_judgments,
    load_examples,
    make_groups,
    positive_ranks,
    ranking_metrics,
)
from legalrag.training.splits import (
    LeakageError,
    SplitAssignment,
    SplitData,
    assert_disjoint,
    assert_no_leakage,
    canonical_cnr,
    partition_examples,
    split_judgments,
)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


def example(query_id, cnr, positives=None, negatives=(), **extra):
    positives = positives if positives is not None else (f"pos text {query_id}",)
    return {
        "query_id": query_id, "query": f"question {query_id}", "cnr": cnr,
        "positives": [{"chunk_id": f"{query_id}-p{i}", "text": t} for i, t in enumerate(positives)],
        "negatives": [{"chunk_id": f"{query_id}-n{i}", "cnr": c, "text": f"neg text {query_id} {i} from {c}"} for i, c in enumerate(negatives)],
        **extra,
    }


def synthetic(judgments=24, per_judgment=2, negatives=6, seed=0):
    """Examples whose negatives come from OTHER judgments, the case that causes leakage."""
    rng = random.Random(seed)
    names = [f"J{i:03d}" for i in range(judgments)]
    out = []
    for name in names:
        for k in range(per_judgment):
            others = rng.sample([n for n in names if n != name], negatives)
            out.append(example(f"{name}-q{k}", name, positives=(f"topic {name} passage {k}",), negatives=others))
    return out


class TestSplitJudgments(unittest.TestCase):
    CNRS = [f"C{i:03d}" for i in range(100)]

    def test_exact_counts_and_no_overlap(self):
        a = split_judgments(self.CNRS, test_fraction=0.15, val_fraction=0.10, seed=1)
        self.assertEqual((len(a.train), len(a.val), len(a.test)), (75, 10, 15))
        self.assertEqual(a.train | a.val | a.test, set(self.CNRS))
        assert_disjoint(a)

    def test_reproducible_and_seed_dependent(self):
        first = split_judgments(self.CNRS, seed=7)
        self.assertEqual(first.test, split_judgments(list(reversed(self.CNRS)), seed=7).test)  # input order does not matter
        self.assertNotEqual(first.test, split_judgments(self.CNRS, seed=8).test)

    def test_forced_test_judgments_are_always_test(self):
        forced = ["C005", "C006", "C007", "NOT-A-JUDGMENT"]
        for seed in range(15):
            a = split_judgments(self.CNRS, 0.15, 0.10, seed, forced_test=forced)
            self.assertTrue({"C005", "C006", "C007"} <= a.test)
            self.assertEqual(a.forced_test, {"C005", "C006", "C007"})
            self.assertFalse({"C005", "C006", "C007"} & (a.train | a.val))
            self.assertEqual(len(a.test), 15)  # forced ones count toward the test share

    def test_more_forced_than_the_test_share_makes_the_test_split_bigger(self):
        forced = self.CNRS[:30]
        a = split_judgments(self.CNRS, 0.15, 0.10, 1, forced_test=forced)
        self.assertEqual(len(a.test), 30)
        self.assertTrue(a.train and a.val)

    def test_small_sets_still_get_every_split(self):
        a = split_judgments(["a", "b", "c", "d", "e"], 0.2, 0.2, 3)
        self.assertTrue(a.train and a.val and a.test)
        assert_disjoint(a)

    def test_tiny_sets_never_leave_train_empty(self):
        for n in (1, 2, 3):
            a = split_judgments([f"x{i}" for i in range(n)], 0.3, 0.3, 1)
            self.assertTrue(a.train, n)

    def test_duplicates_in_the_input_are_one_judgment(self):
        a = split_judgments(["a", "a", "b", "b", "c"], 0.3, 0.0, 1)
        self.assertEqual(len(a.train | a.val | a.test), 3)

    def test_invalid_fractions(self):
        for test, val in ((-0.1, 0.1), (1.0, 0.0), (0.6, 0.5), (0.2, 1.2)):
            with self.subTest(test=test, val=val):
                with self.assertRaises(ValueError):
                    split_judgments(["a", "b"], test, val)


class TestCanonicalCnr(unittest.TestCase):
    def test_case_spaces_and_links(self):
        self.assertEqual(canonical_cnr(" hcbm04_1 ", None), "HCBM04_1")
        links = {"dup1": "kept", "DUP2": "dup1", "loop1": "loop2", "loop2": "loop1"}
        self.assertEqual(canonical_cnr("DUP1", links), "KEPT")
        self.assertEqual(canonical_cnr("dup2", links), "KEPT")  # chains are followed
        self.assertIn(canonical_cnr("loop1", links), ("LOOP1", "LOOP2"))  # a cycle terminates


class TestPartition(unittest.TestCase):
    def assignment(self):
        a = SplitAssignment()
        a.train, a.val, a.test = {"A", "B"}, {"C"}, {"D"}
        return a

    def test_each_example_goes_to_its_judgments_split_and_foreign_negatives_are_dropped(self):
        examples = [
            example("q1", "A", negatives=["A", "B", "C", "D"]),
            example("q2", "C", negatives=["C", "A"]),
            example("q3", "D", negatives=["D"]),
        ]
        data = partition_examples(examples, self.assignment())
        self.assertEqual([e["query_id"] for e in data.train], ["q1"])
        self.assertEqual([n["cnr"] for n in data.train[0]["negatives"]], ["A", "B"])  # D and C removed from a train example
        self.assertEqual([n["cnr"] for n in data.val[0]["negatives"]], ["C"])
        self.assertEqual(len(data.test), 1)
        self.assertEqual(data.dropped_negatives, {"train_example_with_val_negative": 1, "train_example_with_test_negative": 1, "val_example_with_train_negative": 1})
        assert_no_leakage(data, self.assignment())

    def test_examples_without_clean_negatives_or_with_unknown_judgments_are_dropped_and_counted(self):
        data = partition_examples([example("q1", "A", negatives=["D"]), example("q2", "ZZZ", negatives=["ZZZ"])], self.assignment())
        self.assertEqual((len(data.train), data.dropped_examples), (0, {"no_negatives_left": 1, "unassigned_judgment": 1}))

    def test_negatives_with_unknown_judgments_are_dropped(self):
        data = partition_examples([example("q1", "A", negatives=["A", "ZZZ"])], self.assignment())
        self.assertEqual([n["cnr"] for n in data.train[0]["negatives"]], ["A"])
        self.assertEqual(data.dropped_negatives, {"train_example_with_unassigned_negative": 1})

    def test_near_duplicate_copies_are_treated_as_the_kept_judgment(self):
        links = {"A-COPY": "D"}  # a copy of test judgment D must not be usable as a train negative
        data = partition_examples([example("q1", "A", negatives=["A", "A-COPY"])], self.assignment(), links)
        self.assertEqual([n["cnr"] for n in data.train[0]["negatives"]], ["A"])


class TestSharedTextBetweenSplits(unittest.TestCase):
    """Stock paragraphs recur across judgments: the copy in the less protected split is removed, not the run stopped."""

    def assignment(self):
        a = SplitAssignment()
        a.train, a.val, a.test = {"A", "A2"}, {"B"}, {"C"}
        return a

    def test_a_negative_whose_text_is_also_in_test_is_dropped_from_train_and_val(self):
        stock = "The petition is dismissed without costs."
        test_example = example("s", "C", negatives=["C"])
        test_example["negatives"][0]["text"] = stock
        train_example = example("t", "A", negatives=["A", "A2"])
        train_example["negatives"][0]["text"] = "  " + stock.upper()
        val_example = example("v", "B", negatives=["B", "B"])
        val_example["negatives"][1]["text"] = stock
        data = partition_examples([train_example, val_example, test_example], self.assignment())
        self.assertEqual(len(data.train[0]["negatives"]), 1)
        self.assertEqual(len(data.val[0]["negatives"]), 1)
        self.assertEqual(len(data.test[0]["negatives"]), 1)  # the test split is never touched
        self.assertEqual(data.dropped_negatives, {"val_negative_text_also_in_a_cleaner_split": 1, "train_negative_text_also_in_a_cleaner_split": 1})
        assert_no_leakage(data, self.assignment())

    def test_a_positive_whose_text_is_in_a_cleaner_split_drops_its_example(self):
        shared = "Identical positive passage."
        test_example = example("s", "C", positives=(shared,), negatives=["C"])
        train_example = example("t", "A", positives=(shared,), negatives=["A"])
        other = example("u", "A2", negatives=["A2"])
        data = partition_examples([train_example, test_example, other], self.assignment())
        self.assertEqual([e["query_id"] for e in data.train], ["u"])
        self.assertEqual(data.dropped_examples, {"positive_text_also_in_a_cleaner_split": 1})
        assert_no_leakage(data, self.assignment())

    def test_val_takes_priority_over_train(self):
        shared = "Shared between val and train."
        val_example = example("v", "B", negatives=["B", "B"])
        val_example["negatives"][0]["text"] = shared
        train_example = example("t", "A", negatives=["A", "A"])
        train_example["negatives"][1]["text"] = shared
        data = partition_examples([val_example, train_example], self.assignment())
        self.assertEqual(len(data.val[0]["negatives"]), 2)
        self.assertEqual(len(data.train[0]["negatives"]), 1)

    def test_an_example_left_with_no_negatives_is_dropped_and_counted(self):
        test_example = example("s", "C", negatives=["C"])
        train_example = example("t", "A", negatives=["A"])
        train_example["negatives"][0]["text"] = test_example["negatives"][0]["text"]
        data = partition_examples([train_example, test_example], self.assignment())
        self.assertEqual((len(data.train), data.dropped_examples), (0, {"no_negatives_left": 1}))

    def test_text_repeated_inside_one_split_is_left_alone(self):
        a, b = example("t1", "A", negatives=["A"]), example("t2", "A2", negatives=["A2"])
        b["negatives"][0]["text"] = a["negatives"][0]["text"]
        data = partition_examples([a, b], self.assignment())
        self.assertEqual((len(data.train), data.dropped_negatives), (2, {}))


class TestLeakageGuard(unittest.TestCase):
    def setUp(self):
        self.a = SplitAssignment()
        self.a.train, self.a.val, self.a.test = {"A"}, {"B"}, {"C"}

    def data(self, train=(), val=(), test=()):
        return SplitData(train=list(train), val=list(val), test=list(test))

    def test_clean_data_passes_and_counts_judgments(self):
        data = self.data([example("t", "A", negatives=["A"])], [example("v", "B", negatives=["B"])], [example("s", "C", negatives=["C"])])
        self.assertEqual(assert_no_leakage(data, self.a), {"train": 1, "val": 1, "test": 1})

    def test_a_passage_from_another_judgment_is_caught(self):
        for split, home in (("train", "A"), ("val", "B"), ("test", "C")):
            for foreign in {"A", "B", "C"} - {home}:
                with self.subTest(split=split, foreign=foreign):
                    kwargs = {split: [example("x", home, negatives=[foreign])]}
                    with self.assertRaisesRegex(LeakageError, "belongs to"):
                        assert_no_leakage(self.data(**kwargs), self.a)

    def test_the_same_chunk_in_two_splits_is_caught(self):
        t, s = example("t", "A", negatives=["A"]), example("s", "C", negatives=["C"])
        s["negatives"][0]["chunk_id"] = t["negatives"][0]["chunk_id"]
        with self.assertRaisesRegex(LeakageError, "chunk"):
            assert_no_leakage(self.data([t], test=[s]), self.a)

    def test_identical_text_under_different_ids_is_caught(self):
        t, s = example("t", "A", negatives=["A"]), example("s", "C", negatives=["C"])
        s["negatives"][0]["text"] = "  " + t["negatives"][0]["text"].upper() + "\n"
        with self.assertRaisesRegex(LeakageError, "same passage text"):
            assert_no_leakage(self.data([t], test=[s]), self.a)

    def test_an_assignment_with_a_judgment_in_two_splits_is_caught(self):
        self.a.test.add("A")
        with self.assertRaisesRegex(LeakageError, "both train and test"):
            assert_no_leakage(self.data(), self.a)

    def test_a_copy_of_a_test_judgment_in_train_is_caught_through_the_links(self):
        data = self.data([example("t", "A", negatives=["A"])], test=[example("s", "C", negatives=["C"])])
        data.train[0]["cnr"] = "C-COPY"  # a duplicate record of judgment C, placed in train by mistake
        assignment = SplitAssignment()
        assignment.train, assignment.test = {"A", "C-COPY"}, {"C"}
        with self.assertRaises(LeakageError):
            assert_no_leakage(data, assignment, {"C-COPY": "C"})

    def test_a_judgment_cannot_appear_in_both_train_and_test_after_any_partition(self):
        """The task's test, over many random datasets whose negatives deliberately cross judgments."""
        for seed in range(25):
            examples = synthetic(judgments=24, per_judgment=2, negatives=6, seed=seed)
            judgments = {e["cnr"] for e in examples}
            assignment = split_judgments(judgments, 0.2, 0.1, seed, forced_test=sorted(judgments)[:2])
            data = partition_examples(examples, assignment)
            counts = assert_no_leakage(data, assignment)
            train = {n["cnr"] for e in data.train for n in e["negatives"]} | {e["cnr"] for e in data.train}
            test = {n["cnr"] for e in data.test for n in e["negatives"]} | {e["cnr"] for e in data.test}
            self.assertEqual(train & test, set(), seed)
            self.assertGreater(counts["train"], 0)
            self.assertGreater(counts["test"], 0)

    def test_without_partitioning_the_synthetic_data_really_does_leak(self):
        # Guard against a vacuous test: raw examples with cross-judgment negatives fail the check.
        examples = synthetic(judgments=24, per_judgment=2, negatives=6, seed=0)
        judgments = {e["cnr"] for e in examples}
        assignment = split_judgments(judgments, 0.2, 0.1, 0)
        raw = SplitData()
        for e in examples:
            getattr(raw, assignment.split_of(e["cnr"])).append(e)
        with self.assertRaises(LeakageError):
            assert_no_leakage(raw, assignment)


class TestSplitFile(unittest.TestCase):
    def test_roundtrip_and_tamper_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "splits.json"
            original = split_judgments([f"C{i}" for i in range(30)], 0.2, 0.1, 5, forced_test=["C1"])
            original.save(path)
            loaded = SplitAssignment.load(path)
            self.assertEqual((loaded.train, loaded.val, loaded.test, loaded.forced_test), (original.train, original.val, original.test, {"C1"}))
            data = json.loads(path.read_text())
            data["test"].remove("C1")
            data["train"].append("C1")  # moving a judgment into train by hand changes the checksum
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(LeakageError, "edited"):
                SplitAssignment.load(path)
            data = json.loads(Path(path).read_text())
            data.pop("sha256")
            data["val"].append(data["train"][0])  # and a judgment in two splits is refused even without a checksum
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(LeakageError, "both"):
                SplitAssignment.load(path)


class TestLoadingExamples(unittest.TestCase):
    def write(self, lines):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "train.jsonl"
        path.write_text("\n".join(json.dumps(l) if not isinstance(l, str) else l for l in lines), encoding="utf-8")
        return path

    def test_valid_file(self):
        examples = load_examples(self.write([example("q1", " J1 ", negatives=["J2", "J3"]), "", example("q2", "J1", positives=("a", "b"), negatives=["J2"])]))
        self.assertEqual([e["query_id"] for e in examples], ["q1", "q2"])
        self.assertEqual(examples[0]["cnr"], "J1")
        self.assertEqual(len(examples[1]["positives"]), 2)

    def test_every_problem_is_listed_with_its_line(self):
        good = example("q1", "J1", negatives=["J2"])
        bad = {k: v for k, v in good.items() if k != "query"}
        no_cnr_negative = example("q3", "J1", negatives=["J2"])
        del no_cnr_negative["negatives"][0]["cnr"]
        wrong_positive_cnr = example("q4", "J1", negatives=["J2"])
        wrong_positive_cnr["positives"][0]["cnr"] = "J9"
        overlap = example("q5", "J1", negatives=["J2"])
        overlap["negatives"][0]["chunk_id"] = overlap["positives"][0]["chunk_id"]
        no_negatives = example("q6", "J1", negatives=[])
        path = self.write([good, bad, "{not json", no_cnr_negative, wrong_positive_cnr, overlap, no_negatives, good, [1, 2]])
        with self.assertRaises(ExampleError) as ctx:
            load_examples(path)
        message = str(ctx.exception)
        for fragment in ("line 2: missing query", "line 3: not valid JSON", "line 4: every negative needs its judgment's cnr",
                         "line 5: positives must come from", "line 6: a passage is listed as both", "line 7: needs at least one negative",
                         "line 8: duplicate query_id", "line 9: not an object"):
            self.assertIn(fragment, message)

    def test_empty_file(self):
        with self.assertRaisesRegex(ExampleError, "no examples"):
            load_examples(self.write([]))

    def test_benchmark_judgments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            path.write_text(json.dumps([
                {"question_id": "1", "question": "q", "gold_cnrs": ["G1", "G2"], "source_cnr": "S1"},
                {"question_id": "2", "question": "q", "source_cnr": "S2"},
            ]))
            self.assertEqual(benchmark_judgments(path), ["G1", "G2", "S1", "S2"])


class TestGroupsAndMetrics(unittest.TestCase):
    def test_one_group_per_positive_with_the_positive_first(self):
        examples = [example("q", "J1", positives=("p1", "p2"), negatives=["J2"] * 5)]
        groups = make_groups(examples, negatives=3, seed=1, shuffle_negatives=False)
        self.assertEqual(len(groups), 2)
        self.assertEqual([g["texts"][0] for g in groups], ["p1", "p2"])
        self.assertTrue(all(len(g["texts"]) == 4 and g["cnr"] == "J1" for g in groups))
        self.assertEqual(groups[0]["texts"][1:], [f"neg text q {i} from J2" for i in range(3)])  # the first ones, in order

    def test_training_negatives_are_a_reproducible_random_subset(self):
        examples = [example("q", "J1", negatives=["J2"] * 10)]
        first = make_groups(examples, 3, seed=5, shuffle_negatives=True)[0]["texts"]
        self.assertEqual(first, make_groups(examples, 3, seed=5, shuffle_negatives=True)[0]["texts"])
        self.assertNotEqual(first, make_groups(examples, 3, seed=6, shuffle_negatives=True)[0]["texts"])
        self.assertEqual(len(first), 4)

    def test_fewer_negatives_than_requested_uses_them_all(self):
        self.assertEqual(len(make_groups([example("q", "J1", negatives=["J2", "J3"])], 7, 1, True)[0]["texts"]), 3)

    def test_ranking_metrics(self):
        scores = [[0.9, 0.1, 0.2], [0.2, 0.9, 0.1], [0.1, 0.9, 0.8, 0.7]]  # positive ranks 1, 2, 4
        self.assertEqual(positive_ranks(scores), [1, 2, 4])
        metrics = ranking_metrics(scores)
        self.assertAlmostEqual(metrics["mrr"], (1 + 0.5 + 0.25) / 3)
        self.assertAlmostEqual(metrics["recall_at_1"], 1 / 3)
        self.assertAlmostEqual(metrics["recall_at_3"], 2 / 3)
        self.assertEqual(metrics["groups"], 3)

    def test_ties_count_against_the_positive(self):
        self.assertEqual(positive_ranks([[0.5, 0.5, 0.5]]), [3])
        self.assertEqual(positive_ranks([[0.0, 0.0]]), [2])

    def test_empty_input(self):
        self.assertEqual(ranking_metrics([])["groups"], 0)

    def test_config_validation(self):
        for kwargs in ({"epochs": 0}, {"batch_groups": 0}, {"learning_rate": 0}, {"warmup_ratio": 1.0}, {"negatives_per_positive": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TrainingConfig(**kwargs)


# --------------------------------------------------------------------------- training on a tiny model


def topic_examples(judgments=10, per=3, negatives=5, seed=0):
    """Each judgment has its own words; the question shares them with its positive, so a model can learn overlap."""
    rng = random.Random(seed)
    names = [f"J{i:02d}" for i in range(judgments)]
    words = {n: [f"w{i}{c}" for c in "abcde"] for i, n in enumerate(names)}
    out = []
    for n in names:
        for k in range(per):
            query = " ".join(rng.sample(words[n], 3))
            positive = " ".join(rng.sample(words[n], 4)) + f" case{k}"
            negs = rng.sample([m for m in names if m != n], negatives)
            ex = example(f"{n}-q{k}", n, positives=(positive,), negatives=negs)
            ex["query"] = query
            for item in ex["negatives"]:
                item["text"] = " ".join(rng.sample(words[item["cnr"]], 4)) + f" neg{rng.randrange(10**6)}"
            out.append(ex)
    return out


def write_jsonl(path: Path, examples):
    path.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")


def all_words(examples):
    texts = []
    for e in examples:
        texts += [e["query"]] + [p["text"] for p in e["positives"]] + [n["text"] for n in e["negatives"]]
    return texts


@unittest.skipUnless(HAVE_TORCH and HAVE_EXPORT_STACK, "needs torch, transformers, tokenizers, onnx and onnxruntime")
class TestTrainingOnATinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from legalrag.training.reranker import train  # noqa: F401

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.examples = topic_examples(judgments=20, per=4, negatives=5)
        make_tiny_model(cls.root / "base", labels=1, texts=all_words(cls.examples))
        write_jsonl(cls.root / "train.jsonl", cls.examples)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def config(self, **overrides):
        values = dict(base_model=str(self.root / "base"), epochs=3, batch_groups=4, grad_accum=1, negatives_per_positive=4,
                      eval_negatives=5, learning_rate=2e-3, max_length=32, fp16=False, seed=3, eval_batch_size=16, warmup_ratio=0.0)
        values.update(overrides)
        return TrainingConfig(**values)

    def data(self):
        examples = load_examples(self.root / "train.jsonl")
        assignment = split_judgments({e["cnr"] for e in examples}, 0.2, 0.2, 1)
        data = partition_examples(examples, assignment)
        assert_no_leakage(data, assignment)
        return data

    def run_training(self, name, **overrides):
        from legalrag.training.reranker import train

        return train(self.data(), self.root / name, self.config(**overrides), device="cpu", log=lambda *_: None)

    def test_training_runs_learns_something_and_saves_a_loadable_best_model(self):
        report = self.run_training("run_a", epochs=8)
        losses = [h["train_loss"] for h in report["history"]]
        self.assertEqual(len(losses), 8)
        self.assertTrue(all(0 < l < 10 for l in losses))
        # Each judgment has its own words and the question shares them with its positive, so this toy task is
        # learnable; the loss measured on seeds 1-3 fell from about 1.34 to about 0.03.
        self.assertLess(losses[-1], 0.5 * losses[0])
        self.assertIn(report["best_epoch"], range(1, 9))
        self.assertTrue(all("val" in h for h in report["history"]))
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(str(self.root / "run_a" / "best"))
        self.assertEqual(model.config.num_labels, 1)
        self.assertGreater(report["baseline"]["test"]["groups"], 0)
        self.assertGreater(report["tuned"]["test"]["groups"], 0)

    def test_report_has_a_paired_comparison_on_the_held_out_judgments(self):
        report = self.run_training("run_b")
        comparison = report["test_comparison"]
        self.assertIn(comparison["verdict"], ("a better", "b better", "no clear difference"))
        self.assertEqual(comparison["difference"]["method"], "cluster_bootstrap")
        self.assertGreater(comparison["judgments"], 0)
        self.assertIsNotNone(comparison["difference"]["low"])
        self.assertTrue(any("scored once" in n for n in report["notes"]))

    def test_same_seed_gives_the_same_history(self):
        first = self.run_training("run_c1")
        second = self.run_training("run_c2")
        self.assertEqual([h["train_loss"] for h in first["history"]], [h["train_loss"] for h in second["history"]])
        self.assertEqual(first["tuned"], second["tuned"])

    def test_test_split_does_not_influence_training(self):
        from legalrag.training.reranker import train

        data = self.data()
        baseline = train(data, self.root / "run_d1", self.config(), device="cpu", log=lambda *_: None)
        changed = SplitData(train=data.train, val=data.val, test=[{**e, "query": "completely different words"} for e in data.test])
        other = train(changed, self.root / "run_d2", self.config(), device="cpu", log=lambda *_: None)
        self.assertEqual([h["train_loss"] for h in baseline["history"]], [h["train_loss"] for h in other["history"]])
        self.assertEqual(baseline["best_epoch"], other["best_epoch"])

    def test_without_a_validation_split_the_last_epoch_is_kept(self):
        data = self.data()
        data.val = []
        from legalrag.training.reranker import train

        report = train(data, self.root / "run_e", self.config(epochs=2), device="cpu", log=lambda *_: None)
        self.assertEqual(report["best_epoch"], 2)
        self.assertIsNone(report["baseline"]["val"])

    def test_no_training_examples_is_an_error(self):
        from legalrag.training.reranker import train

        with self.assertRaisesRegex(ValueError, "no training examples"):
            train(SplitData(), self.root / "run_f", self.config(), device="cpu", log=lambda *_: None)

    def test_a_diverging_loss_stops_with_a_clear_message(self):
        with mock.patch("torch.nn.functional.cross_entropy", return_value=__import__("torch").tensor(float("nan"))):
            with self.assertRaisesRegex(FloatingPointError, "lower the learning rate"):
                self.run_training("run_g")

    def test_a_cpu_run_rejects_a_cuda_request(self):
        if torch.cuda.is_available():
            self.skipTest("this machine has CUDA")
        from legalrag.training.reranker import resolve_device

        with self.assertRaisesRegex(RuntimeError, "no CUDA"):
            resolve_device("cuda")
        self.assertEqual(resolve_device("auto"), "cpu")


# --------------------------------------------------------------------------- the script


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_reranker.py"
    spec = importlib.util.spec_from_file_location("train_reranker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


def run_script(argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        script.main(argv)
    return buffer.getvalue()


class TestScriptWithoutTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.examples = synthetic(judgments=30, per_judgment=2, negatives=6, seed=2)
        write_jsonl(self.root / "train.jsonl", self.examples)
        self.out = self.root / "v2" / "reranker"

    def args(self, *extra):
        return ["train", "--examples", str(self.root / "train.jsonl"), "--out-dir", str(self.out), *extra]

    def test_dry_run_reports_clean_splits_and_writes_nothing(self):
        output = run_script(self.args("--dry-run"))
        summary = json.loads(output[: output.rindex("}") + 1])
        self.assertEqual(summary["leakage_check"], "passed")
        self.assertEqual(sum(summary["judgments"].values()), 30)
        self.assertGreater(summary["examples_per_split"]["train"], 0)
        self.assertGreater(sum(summary["dropped_negatives"].values()), 0)  # the random negatives crossed splits and were removed
        self.assertEqual(len(summary["examples_sha256"]), 64)
        self.assertFalse(self.out.exists())
        self.assertIn("Dry run", output)

    def test_benchmark_judgments_are_forced_into_test(self):
        bench = self.root / "bench.json"
        forced = ["J000", "J001", "J002", "J003"]
        bench.write_text(json.dumps([{"question_id": f"b{i}", "question": "q", "source_cnr": c} for i, c in enumerate(forced)]))
        output = run_script(self.args("--dry-run", "--benchmark", str(bench), "--test-fraction", "0.1"))
        summary = json.loads(output[: output.rindex("}") + 1])
        self.assertEqual(summary["benchmark_judgments_forced_to_test"], 4)
        self.assertGreaterEqual(summary["judgments"]["test"], 4)

    def test_splitting_is_reproducible_and_a_saved_split_can_be_reused(self):
        first = json.loads((lambda o: o[: o.rindex("}") + 1])(run_script(self.args("--dry-run", "--seed", "9"))))
        again = json.loads((lambda o: o[: o.rindex("}") + 1])(run_script(self.args("--dry-run", "--seed", "9"))))
        self.assertEqual(first["split_sha256"], again["split_sha256"])
        self.assertNotEqual(first["split_sha256"], json.loads((lambda o: o[: o.rindex("}") + 1])(run_script(self.args("--dry-run", "--seed", "10"))))["split_sha256"])

    def test_a_reused_split_that_puts_a_benchmark_judgment_in_train_is_refused(self):
        bench = self.root / "bench.json"
        bench.write_text(json.dumps([{"question_id": "b", "question": "q", "source_cnr": "J000"}]))
        assignment = split_judgments([f"J{i:03d}" for i in range(30)], 0.2, 0.1, 1)  # made without the benchmark
        self.assertIn("J000", assignment.train | assignment.val | assignment.test)
        if "J000" in assignment.test:
            assignment.test.remove("J000")
            assignment.train.add("J000")
        assignment.save(self.root / "splits.json")
        with self.assertRaisesRegex(SystemExit, "benchmark judgment"):
            run_script(self.args("--dry-run", "--benchmark", str(bench), "--splits", str(self.root / "splits.json")))

    def test_text_shared_between_splits_is_removed_from_the_cleaner_side_and_reported(self):
        # Every question has the same positive passage text, so it appears in all three splits.
        shared = [dict(e, positives=[{**e["positives"][0], "text": "the very same passage text used twice"}]) for e in self.examples]
        write_jsonl(self.root / "train.jsonl", shared)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaisesRegex(SystemExit, "No training examples remain"):
                script.main(self.args("--dry-run"))
        summary = json.loads(buffer.getvalue()[: buffer.getvalue().rindex("}") + 1])
        self.assertGreater(summary["dropped_examples"]["positive_text_also_in_a_cleaner_split"], 0)
        self.assertEqual(summary["leakage_check"], "passed")  # nothing shared remained to be checked

    def test_output_guards(self):
        with self.assertRaisesRegex(SystemExit, "'v2' part"):
            run_script(["train", "--examples", str(self.root / "train.jsonl"), "--out-dir", str(self.root / "models"), "--dry-run"])
        with self.assertRaisesRegex(SystemExit, "'v2' part"):
            run_script(["export", "--model", str(self.root), "--out-dir", str(self.root / "onnx")])

    def test_a_malformed_example_file_is_reported_before_anything_else(self):
        (self.root / "train.jsonl").write_text('{"query_id": "q"}\nnot json', encoding="utf-8")
        with self.assertRaisesRegex(ExampleError, "line 1.*line 2"):
            run_script(self.args("--dry-run"))

    def test_nothing_left_to_train_on_is_a_clear_error(self):
        # The benchmark covers every judgment, so all of them are forced into test and train is empty.
        bench = self.root / "bench.json"
        bench.write_text(json.dumps([{"question_id": f"b{i}", "question": "q", "source_cnr": f"J{i:03d}"} for i in range(30)]))
        with self.assertRaisesRegex(SystemExit, "No training examples remain"):
            run_script(self.args("--dry-run", "--benchmark", str(bench)))


@unittest.skipUnless(HAVE_TORCH and HAVE_EXPORT_STACK, "needs torch, transformers, tokenizers, onnx and onnxruntime")
class TestScriptTrainingEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.examples = topic_examples(judgments=12, per=3, negatives=5)
        make_tiny_model(cls.root / "base", labels=1, texts=all_words(cls.examples))
        write_jsonl(cls.root / "train.jsonl", cls.examples)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def args(self, out, *extra):
        return ["train", "--examples", str(self.root / "train.jsonl"), "--out-dir", str(self.root / "v2" / out), "--base-model",
                str(self.root / "base"), "--epochs", "2", "--batch-groups", "4", "--grad-accum", "1", "--negatives", "4",
                "--lr", "1e-3", "--max-length", "32", "--no-fp16", "--device", "cpu", "--seed", "3", *extra]

    def test_writes_the_model_splits_report_and_a_verified_onnx_export(self):
        run_script(self.args("full", "--export-onnx"))
        out = self.root / "v2" / "full"
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["best", "onnx", "splits.json", "train_report.json"])
        report = json.loads((out / "train_report.json").read_text())
        self.assertEqual(report["split"]["leakage_check"], "passed")
        self.assertLessEqual(report["onnx_verification"]["max_abs_difference"], 1e-4)
        self.assertEqual(report["split"]["split_sha256"], SplitAssignment.load(out / "splits.json").sha256())
        self.assertIn("torch", report["versions"])
        from legalrag.retrieval.onnx_models import OnnxReranker

        ranked = OnnxReranker.load(out / "onnx").rerank("w1a w1b", [("a", "w1a w1b w1c"), ("b", "w5a w5b w5c")], top_k=2)
        self.assertEqual({c for c, _ in ranked}, {"a", "b"})

    def test_refuses_to_overwrite_and_overwrites_cleanly_when_asked(self):
        run_script(self.args("again"))
        with self.assertRaisesRegex(SystemExit, "already holds"):
            run_script(self.args("again"))
        run_script(self.args("again", "--overwrite"))
        self.assertFalse((self.root / "v2" / "again" / script.STAGING).exists())

    def test_a_failed_run_leaves_the_previous_outputs_untouched(self):
        run_script(self.args("keep"))
        out = self.root / "v2" / "keep"
        before = {p.name: (p.read_bytes() if p.is_file() else None) for p in out.iterdir()}
        with mock.patch("legalrag.training.reranker.train", side_effect=RuntimeError("simulated crash")):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                run_script(self.args("keep", "--overwrite"))
        self.assertEqual({p.name: (p.read_bytes() if p.is_file() else None) for p in out.iterdir()}, before)
        self.assertFalse((out / script.STAGING).exists())

    def test_a_reused_split_file_gives_the_same_split(self):
        run_script(self.args("first"))
        saved = self.root / "v2" / "first" / "splits.json"
        run_script(self.args("second", "--splits", str(saved)))
        self.assertEqual(json.loads(saved.read_text())["sha256"], json.loads((self.root / "v2" / "second" / "splits.json").read_text())["sha256"])

    def test_export_subcommand(self):
        run_script(self.args("toexport"))
        output = run_script(["export", "--model", str(self.root / "v2" / "toexport" / "best"), "--out-dir", str(self.root / "v2" / "onnx_only"), "--max-length", "32"])
        self.assertLessEqual(json.loads(output)["verification"]["max_abs_difference"], 1e-4)
        self.assertTrue((self.root / "v2" / "onnx_only" / "model.onnx").is_file())


if __name__ == "__main__":
    unittest.main()

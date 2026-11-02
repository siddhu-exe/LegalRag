"""Tests for the evaluation statistics (plan task C3)."""

import math
import unittest

import numpy as np

from legalrag.evaluation.stats import (
    Interval,
    bootstrap_ci,
    cohens_kappa,
    holm_adjust,
    kappa_interval,
    mcnemar_exact,
    paired_bootstrap,
    percent_agreement,
    summarize,
    verdict_from_interval,
    wilson_interval,
)


class TestWilson(unittest.TestCase):
    def test_known_values(self):
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=3)
        self.assertAlmostEqual(high, 0.7634, places=3)
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)

    def test_edges_stay_inside_zero_one(self):
        low, high = wilson_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        low, high = wilson_interval(20, 20)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_empty_and_invalid(self):
        self.assertTrue(math.isnan(wilson_interval(0, 0)[0]))
        with self.assertRaises(ValueError):
            wilson_interval(11, 10)
        with self.assertRaises(ValueError):
            wilson_interval(5, 10, confidence=1.5)

    def test_higher_confidence_is_wider(self):
        narrow = wilson_interval(30, 100, 0.90)
        wide = wilson_interval(30, 100, 0.99)
        self.assertGreater(wide[1] - wide[0], narrow[1] - narrow[0])


class TestBootstrap(unittest.TestCase):
    def test_estimate_is_the_sample_mean_and_inside_interval(self):
        values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5, 0.7]
        interval = bootstrap_ci(values, n_resamples=2000, seed=1)
        self.assertAlmostEqual(interval.estimate, float(np.mean(values)))
        self.assertLessEqual(interval.low, interval.estimate)
        self.assertGreaterEqual(interval.high, interval.estimate)
        self.assertEqual((interval.n, interval.method), (8, "bootstrap"))

    def test_constant_values_have_zero_width(self):
        interval = bootstrap_ci([3.0] * 10, n_resamples=500)
        self.assertEqual((interval.low, interval.high), (3.0, 3.0))

    def test_same_seed_same_result_and_different_seed_differs(self):
        values = list(np.random.default_rng(0).random(60))
        first = bootstrap_ci(values, seed=5, n_resamples=1000)
        again = bootstrap_ci(values, seed=5, n_resamples=1000)
        other = bootstrap_ci(values, seed=6, n_resamples=1000)
        self.assertEqual(first, again)
        self.assertNotEqual((first.low, first.high), (other.low, other.high))

    def test_more_data_gives_a_narrower_interval(self):
        rng = np.random.default_rng(3)
        small = bootstrap_ci(rng.random(20), n_resamples=2000)
        large = bootstrap_ci(rng.random(2000), n_resamples=2000)
        self.assertLess(large.high - large.low, small.high - small.low)

    def test_empty_values(self):
        interval = bootstrap_ci([])
        self.assertEqual(interval.n, 0)
        self.assertTrue(math.isnan(interval.estimate))

    def test_clusters_widen_the_interval_when_questions_share_a_judgment(self):
        # 40 judgments x 10 questions; every question in a judgment has the same outcome.
        rng = np.random.default_rng(11)
        per_judgment = rng.integers(0, 2, size=40)
        values = np.repeat(per_judgment, 10)
        groups = np.repeat(np.arange(40), 10)
        naive = bootstrap_ci(values, n_resamples=3000, seed=1)
        clustered = bootstrap_ci(values, groups=groups, n_resamples=3000, seed=1)
        self.assertEqual(clustered.method, "cluster_bootstrap")
        self.assertGreater(clustered.high - clustered.low, 1.8 * (naive.high - naive.low))
        self.assertAlmostEqual(clustered.estimate, naive.estimate)

    def test_groups_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([1, 2, 3], groups=["a", "b"])

    def test_unequal_group_sizes_use_the_ratio_of_sums(self):
        interval = bootstrap_ci([1, 1, 1, 1, 0], groups=["a", "a", "a", "a", "b"], n_resamples=1000)
        self.assertAlmostEqual(interval.estimate, 0.8)
        self.assertGreaterEqual(interval.low, 0.0)
        self.assertLessEqual(interval.high, 1.0)


class TestSummarize(unittest.TestCase):
    def test_binary_without_groups_uses_wilson(self):
        interval = summarize([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
        self.assertEqual(interval.method, "wilson")
        self.assertAlmostEqual(interval.estimate, 0.7)

    def test_binary_with_groups_uses_cluster_bootstrap(self):
        interval = summarize([1, 0, 1, 1], groups=["a", "a", "b", "b"], n_resamples=500)
        self.assertEqual(interval.method, "cluster_bootstrap")

    def test_scores_use_bootstrap(self):
        self.assertEqual(summarize([1, 2, 3, 4, 3, 2], n_resamples=500).method, "bootstrap")

    def test_every_summary_has_an_interval(self):
        interval = summarize([0.5, 0.25, 0.75, 1.0])
        self.assertFalse(math.isnan(interval.low) or math.isnan(interval.high))
        self.assertIn("[", interval.format())

    def test_to_dict_is_json_safe_for_nan(self):
        data = summarize([]).to_dict()
        self.assertIsNone(data["estimate"])


class TestPairedBootstrap(unittest.TestCase):
    def test_clear_improvement_is_detected(self):
        rng = np.random.default_rng(2)
        b = rng.random(200) * 0.5
        a = b + 0.2 + rng.normal(0, 0.02, 200)
        result = paired_bootstrap(a, b, n_resamples=2000, seed=1)
        self.assertEqual(result.verdict, "a better")
        self.assertGreater(result.difference.low, 0.0)
        self.assertLess(result.p_value, 0.01)

    def test_identical_systems_show_no_difference(self):
        values = list(np.random.default_rng(4).random(50))
        result = paired_bootstrap(values, values, n_resamples=500)
        self.assertEqual(result.verdict, "no clear difference")
        self.assertEqual(result.difference.estimate, 0.0)
        self.assertEqual(result.p_value, 1.0)

    def test_small_noisy_difference_is_not_claimed(self):
        rng = np.random.default_rng(8)
        a = rng.integers(0, 2, 30)
        b = rng.integers(0, 2, 30)
        result = paired_bootstrap(a, b, n_resamples=2000, seed=1)
        self.assertEqual(result.verdict, "no clear difference")

    def test_worse_system_is_reported_as_b_better(self):
        b = np.linspace(0.5, 1.0, 100)
        a = b - 0.1
        self.assertEqual(paired_bootstrap(a, b, n_resamples=1000).verdict, "b better")

    def test_clusters_are_resampled_together(self):
        a = np.repeat([1, 0, 1, 1, 0, 1], 10)
        b = np.repeat([0, 0, 1, 0, 0, 1], 10)
        groups = np.repeat(np.arange(6), 10)
        naive = paired_bootstrap(a, b, n_resamples=2000, seed=1)
        clustered = paired_bootstrap(a, b, groups=groups, n_resamples=2000, seed=1)
        self.assertEqual(clustered.p_method, "cluster_bootstrap")
        self.assertGreater(
            clustered.difference.high - clustered.difference.low,
            naive.difference.high - naive.difference.low,
        )

    def test_shape_mismatch_and_empty_raise(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 2], [1])
        with self.assertRaises(ValueError):
            paired_bootstrap([], [])

    def test_verdict_from_interval(self):
        self.assertEqual(verdict_from_interval(Interval(0.1, 0.01, 0.2, 10, "x")), "a better")
        self.assertEqual(verdict_from_interval(Interval(-0.1, -0.2, -0.01, 10, "x")), "b better")
        self.assertEqual(verdict_from_interval(Interval(0.0, -0.1, 0.1, 10, "x")), "no clear difference")


class TestMcNemar(unittest.TestCase):
    def test_only_one_system_ever_wins(self):
        result = mcnemar_exact([1] * 10, [0] * 10)
        self.assertEqual((result["a_only"], result["b_only"], result["discordant"]), (10, 0, 10))
        self.assertAlmostEqual(result["p_value"], 2 / 1024)

    def test_balanced_disagreements_give_p_one(self):
        a = [1, 1, 1, 0, 0, 0, 1, 0]
        b = [0, 0, 0, 1, 1, 1, 1, 0]
        self.assertEqual(mcnemar_exact(a, b)["p_value"], 1.0)

    def test_no_disagreements(self):
        result = mcnemar_exact([1, 0, 1], [1, 0, 1])
        self.assertEqual((result["discordant"], result["p_value"]), (0, 1.0))

    def test_known_small_case(self):
        # 2 vs 8 disagreements: two-sided exact p = 2 * P(X <= 2 | n=10) = 2 * 56/1024
        a = [1] * 2 + [0] * 8
        b = [0] * 2 + [1] * 8
        self.assertAlmostEqual(mcnemar_exact(a, b)["p_value"], 2 * 56 / 1024)


class TestHolm(unittest.TestCase):
    def test_known_adjustment(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_caps_at_one_and_keeps_order(self):
        self.assertEqual(holm_adjust([0.6, 0.9]), [1.0, 1.0])
        self.assertEqual(holm_adjust([]), [])

    def test_adjusted_is_never_smaller(self):
        raw = [0.001, 0.02, 0.2, 0.5]
        for before, after in zip(raw, holm_adjust(raw)):
            self.assertGreaterEqual(after, before)


class TestKappa(unittest.TestCase):
    def test_textbook_example(self):
        # 20 both yes, 5 A yes/B no, 10 A no/B yes, 15 both no -> po=0.7, pe=0.5, kappa=0.4
        a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
        b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        self.assertAlmostEqual(cohens_kappa(a, b), 0.4)
        self.assertAlmostEqual(percent_agreement(a, b), 0.7)

    def test_perfect_agreement_is_one(self):
        labels = [1, 2, 3, 4, 2, 3]
        self.assertAlmostEqual(cohens_kappa(labels, labels), 1.0)
        self.assertAlmostEqual(cohens_kappa(labels, labels, "quadratic"), 1.0)

    def test_chance_level_agreement_is_near_zero(self):
        rng = np.random.default_rng(0)
        a = list(rng.integers(0, 2, 2000))
        b = list(rng.integers(0, 2, 2000))
        self.assertAlmostEqual(cohens_kappa(a, b), 0.0, delta=0.06)

    def test_weighted_kappa_penalises_far_disagreements_more(self):
        truth = [1, 2, 3, 4] * 10
        near = [min(4, v + 1) if i % 4 == 0 else v for i, v in enumerate(truth)]
        far = [4 if v == 1 and i % 4 == 0 else v for i, v in enumerate(truth)]
        self.assertGreater(cohens_kappa(truth, near, "quadratic"), cohens_kappa(truth, far, "quadratic"))

    def test_undefined_when_both_use_one_label(self):
        self.assertTrue(math.isnan(cohens_kappa([1, 1, 1], [1, 1, 1])))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            cohens_kappa([1], [1, 2])
        with self.assertRaises(ValueError):
            cohens_kappa([1, 2], [1, 2], weights="cubic")
        self.assertTrue(math.isnan(cohens_kappa([], [])))

    def test_interval_contains_estimate_and_is_reproducible(self):
        a = [1, 2, 3, 4, 3, 2, 1, 4, 3, 3, 2, 2, 4, 1, 3, 2]
        b = [1, 2, 3, 3, 3, 2, 2, 4, 3, 4, 2, 1, 4, 1, 3, 3]
        interval = kappa_interval(a, b, "linear", n_resamples=500, seed=3)
        self.assertLessEqual(interval.low, interval.estimate)
        self.assertGreaterEqual(interval.high, interval.estimate)
        self.assertEqual(interval, kappa_interval(a, b, "linear", n_resamples=500, seed=3))


if __name__ == "__main__":
    unittest.main()

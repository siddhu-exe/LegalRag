"""
Tests for the data quality report (plan task C2): text measures, the accumulator, and the script
end to end on a corpus built from small fixtures. No network, models or real artifacts.
"""

import csv
import datetime as dt
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from legalrag.preprocessing.corpus_v2 import BuildConfig
from legalrag.preprocessing.quality import (
    OcrThresholds,
    has_repeated_4gram,
    in_sample,
    is_ocr_suspect,
    ocr_flags,
    ocr_noise_metrics,
    year_period,
)

CLEAN = ("The learned counsel for the petitioner submitted that the impugned order dated 12 March 2015 "
         "passed by the Collector under Section 12 of the Act is contrary to law and the writ petition "
         "should be allowed with costs. ") * 5
GARBLED_SPACED = "t h e c o u r t h e l d t h a t t h e o r d e r i s b a d " * 20
GARBLED_VOWELLESS = "xcvbn qwrtp zxcvb mnbvc " * 40
GARBLED_SYMBOLS = "~~^^|| ##@@ <<>> \\\\ ~~^^ " * 40
GARBLED_UNDECODABLE = "The court ������� held " * 5


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report_script = _load("data_quality_report")
build_script = _load("build_corpus_v2")


class TestOcrMeasures(unittest.TestCase):
    def test_clean_prose_is_not_flagged(self):
        metrics = ocr_noise_metrics(CLEAN)
        self.assertFalse(is_ocr_suspect(metrics))
        self.assertEqual(metrics["replacement_chars"], 0)
        self.assertLess(metrics["symbol_share"], 0.02)

    def test_each_kind_of_noise_trips_its_own_flag(self):
        cases = [
            (GARBLED_SPACED, "spaced_letters"),
            (GARBLED_VOWELLESS, "vowelless_words"),
            (GARBLED_SYMBOLS, "symbols"),
            (GARBLED_UNDECODABLE, "replacement_chars"),
            ("धारा ४३८ के अंतर्गत " * 30, "non_ascii"),
        ]
        for text, flag in cases:
            with self.subTest(flag=flag):
                flags = ocr_flags(ocr_noise_metrics(text))
                self.assertTrue(flags[flag], flags)
                self.assertTrue(is_ocr_suspect(ocr_noise_metrics(text)))

    def test_capitalised_abbreviations_and_a_i_are_not_noise(self):
        metrics = ocr_noise_metrics("A CRPC SCRRB petition I filed " * 30)
        self.assertEqual((metrics["spaced_letter_share"], metrics["vowelless_share"]), (0.0, 0.0))

    def test_legal_enumerations_and_section_references_are_not_noise(self):
        text = ("The conditions in clauses (a) (b) (c) (d) (e) (f) under Section 12A and Article 14(1)(c) "
                "apply, see s. 438 and Rule 3 (a), (b), (c). ") * 20
        metrics = ocr_noise_metrics(text)
        self.assertEqual(metrics["spaced_letter_share"], 0.0)
        self.assertFalse(is_ocr_suspect(metrics))

    def test_spaced_letters_are_measured_as_a_share_of_text(self):
        metrics = ocr_noise_metrics("t h e c o u r t " + "x" * 0 + "held")
        self.assertGreater(metrics["spaced_letter_share"], 0.5)

    def test_thresholds_can_be_moved(self):
        metrics = ocr_noise_metrics(GARBLED_SPACED)
        self.assertFalse(is_ocr_suspect(metrics, OcrThresholds(max_spaced_letter_share=1.0)))

    def test_empty_text(self):
        metrics = ocr_noise_metrics("")
        self.assertEqual(metrics["chars"], 0)
        self.assertFalse(is_ocr_suspect(metrics))


class TestRepeatedFourGram(unittest.TestCase):
    def test_detects_a_phrase_repeated_three_times(self):
        self.assertTrue(has_repeated_4gram("CORAM A B SHARMA J " * 3 + "body of the judgment"))

    def test_two_repeats_are_not_enough(self):
        self.assertFalse(has_repeated_4gram("CORAM A B SHARMA J " * 2 + "body of the judgment follows here"))

    def test_only_the_head_counts(self):
        text = "x" * 4000 + " " + "CORAM A B SHARMA J " * 5
        self.assertFalse(has_repeated_4gram(text))
        self.assertFalse(has_repeated_4gram(""))


class TestSamplingAndPeriods(unittest.TestCase):
    def test_sampling_is_deterministic_and_close_to_the_rate(self):
        keys = [f"CNR{i}" for i in range(20000)]
        chosen = [k for k in keys if in_sample(k, 0.1, "s")]
        self.assertEqual(chosen, [k for k in keys if in_sample(k, 0.1, "s")])
        self.assertAlmostEqual(len(chosen) / len(keys), 0.1, delta=0.01)
        self.assertNotEqual(chosen, [k for k in keys if in_sample(k, 0.1, "other")])

    def test_rate_extremes(self):
        self.assertTrue(in_sample("a", 1.0))
        self.assertFalse(in_sample("a", 0.0))

    def test_year_periods_match_the_profile_bands(self):
        expected = {2009: "before 2010", 2010: "2010-14", 2014: "2010-14", 2015: "2015-19",
                    2019: "2015-19", 2020: "2020-23", 2023: "2020-23", 2024: "2024+", None: "unknown"}
        for year, period in expected.items():
            self.assertEqual(year_period(year), period)


def row(cnr, text=CLEAN, searched=True, **overrides):
    base = {
        "cnr": cnr, "source_config": "high_courts", "bench_name": "Nagpur", "in_search_index": searched,
        "judgment_date": "2015-06-01", "judgment_year": 2015, "date_source": "decision_date",
        "decision_date_status": "ok", "cnr_date_status": "ok", "date_conflict": False, "year_conflict": False,
        "judge": "A B Sharma", "title": "Ram v. State", "case_type_source": "title", "outcome": "allowed",
        "clean_chars": len(text), "repeat_chars_removed": 0, "text": text,
    }
    base.update(overrides)
    return base


def accumulator(**kwargs):
    settings = {"thresholds": OcrThresholds(), "sample_rate": 1.0}
    settings.update(kwargs)
    return report_script.QualityAccumulator(**settings)


class TestAccumulator(unittest.TestCase):
    def test_coverage_by_bench_and_period(self):
        acc = accumulator()
        acc.add(row("A1", text="x " * 3000, clean_chars=6000))
        acc.add(row("A2", text="y " * 4000, clean_chars=8000, judgment_year=2022))
        acc.add(row("A3", text="short order", searched=False, clean_chars=11))
        acc.add(row("B1", bench_name="Goa", judgment_year=2001))
        result = acc.result()
        nagpur = next(r for r in result["coverage_by_bench"] if r["bench_name"] == "Nagpur")
        self.assertEqual((nagpur["records"], nagpur["search_records"], nagpur["search_chars"]), (3, 2, 14000))
        self.assertEqual(nagpur["median_search_chars"], 7000)
        self.assertEqual((nagpur["first_year"], nagpur["last_year"]), (2015, 2022))
        self.assertEqual(result["records"], {"all": 4, "search": 3, "lookup_only": 1})
        self.assertEqual(result["coverage_by_period"]["before 2010"], {"records": 1, "search_records": 1})

    def test_missing_dates_and_conflicts(self):
        acc = accumulator()
        acc.add(row("A1", judgment_date=None, date_source=None, decision_date_status="missing", cnr_date_status="implausible"))
        acc.add(row("A2", date_conflict=True, year_conflict=True))
        acc.add(row("A3"))
        dates = acc.result()["missing_and_odd_dates"]
        self.assertEqual(dates["missing_judgment_date"]["count"], 1)
        self.assertEqual((dates["date_conflict"]["count"], dates["year_conflict"]["count"]), (1, 1))
        self.assertEqual(dates["cnr_date_status"], {"implausible": 1, "ok": 2})
        self.assertEqual(dates["missing_by_bench"]["Nagpur"]["count"], 1)
        self.assertAlmostEqual(dates["missing_judgment_date"]["share"], 1 / 3)

    def test_ocr_sample_flags_noise_and_gives_intervals(self):
        acc = accumulator()
        for i in range(8):
            acc.add(row(f"C{i}", text=CLEAN + f" unique{i}"))
        acc.add(row("N1", text=GARBLED_SPACED))
        acc.add(row("N2", text=GARBLED_UNDECODABLE))
        quality = acc.result()["text_quality"]
        self.assertEqual(quality["sampled_records"], 10)
        self.assertEqual(quality["ocr_suspect"]["count"], 2)
        self.assertLess(quality["ocr_suspect"]["ci_low"], 0.2)
        self.assertGreater(quality["ocr_suspect"]["ci_high"], 0.2)
        self.assertIn("spaced_letters", quality["ocr_flag_reasons"])
        self.assertEqual(quality["ocr_suspect_by_bench"]["Nagpur"]["total"], 10)
        self.assertEqual(sum(1 for r in acc.review_rows() if r["ocr_suspect"]), 2)

    def test_text_sample_respects_the_rate(self):
        acc = accumulator(sample_rate=0.0)
        acc.add(row("A1"))
        quality = acc.result()["text_quality"]
        self.assertEqual(quality["sampled_records"], 0)
        self.assertIsNone(quality["ocr_suspect"]["share"])
        self.assertEqual(acc.result()["coverage_by_bench"][0]["records"], 1)

    def test_residual_duplicates_and_repeated_cnrs_are_detected(self):
        acc = accumulator()
        acc.add(row("A1", text=CLEAN))
        acc.add(row("A2", text=CLEAN.upper()))
        # Same CNR again, and a text that differs from CLEAN inside the 2,000-character prefix: not a duplicate.
        acc.add(row("A2", text=CLEAN + "different tail " * 3000))
        dups = acc.result()["duplicates"]
        self.assertEqual(dups["repeated_cnr_rows_in_output"], 1)
        self.assertEqual(dups["residual_near_duplicates_in_search_index"], 1)

    def test_lookup_only_rows_are_not_checked_for_duplicates(self):
        acc = accumulator()
        acc.add(row("S1", text="Adjourned.", searched=False))
        acc.add(row("S2", text="Adjourned.", searched=False))
        self.assertEqual(acc.result()["duplicates"]["residual_near_duplicates_in_search_index"], 0)

    def test_residual_check_can_be_skipped(self):
        acc = accumulator(residual_duplicate_check=False)
        acc.add(row("A1"))
        acc.add(row("A2"))
        dups = acc.result()["duplicates"]
        self.assertIsNone(dups["residual_near_duplicates_in_search_index"])
        self.assertFalse(dups["residual_check_ran"])

    def test_metadata_and_repeat_cleaning_counts(self):
        acc = accumulator()
        acc.add(row("A1", judge=None, outcome="other", repeat_chars_removed=40, case_type_source="none"))
        acc.add(row("A2", title="", outcome=""))
        acc.add(row("A3"))
        result = acc.result()
        self.assertEqual(result["metadata"]["blank_judge"]["count"], 1)
        self.assertEqual(result["metadata"]["blank_title"]["count"], 1)
        self.assertEqual(result["metadata"]["blank_or_other_outcome"]["count"], 2)
        self.assertEqual(result["metadata"]["case_type_source"], {"none": 1, "title": 2})
        self.assertEqual(result["text_quality"]["records_where_repeat_cleaning_removed_text"]["count"], 1)

    def test_empty_corpus_does_not_crash(self):
        result = accumulator().result()
        self.assertEqual(result["records"], {})
        self.assertEqual(result["coverage_by_bench"], [])
        report_script.render_markdown({**result, "build": None})


def hc_record(i, text, **overrides):
    base = {
        "cnr": f"HCBM04_{i}_2015-06-0{(i % 8) + 1}", "court_code": "27_1", "bench": "testcase", "year": 2015,
        "full_text": text, "title": "WP/10/2015", "judge": "A B Sharma", "petitioner": "Ram", "respondent": "State",
        "decision_date": f"2015-06-0{(i % 8) + 1}", "disposal_nature_normalized": "allowed", "case_type": "",
    }
    base.update(overrides)
    return base


class TestReportEndToEnd(unittest.TestCase):
    def build_corpus(self, tmp: Path, max_rows=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [hc_record(i, f"Judgment {i} " + " ".join(f"word{i}x{j}" for j in range(80))) for i in range(12)]
        rows.append(hc_record(20, GARBLED_SPACED * 3))
        rows.append(hc_record(21, "Rule discharged."))
        rows.append(hc_record(22, rows[0]["full_text"]))  # near duplicate of the first
        rows.append(hc_record(23, "Sunk", court_code="10_8"))
        rows.append(hc_record(24, "x y z " * 100, decision_date="3021-03-24"))
        source = tmp / "hc.parquet"
        pq.write_table(pa.Table.from_pylist(rows), source)
        out = tmp / "v2" / "corpus"
        build_script.build({"high_courts": [source], "supreme_court": []}, out,
                           BuildConfig(min_chars=200, today=dt.date(2026, 11, 2)), max_rows=max_rows)
        return out

    def run_report(self, corpus, out, *extra):
        report_script.main(["--corpus-dir", str(corpus), "--out-dir", str(out), "--sample-rate", "1.0",
                            "--profile", str(corpus / "no-profile.json"), *extra])
        return json.loads((out / "data_quality_report.json").read_text(encoding="utf-8")), \
            (out / "data_quality_report.md").read_text(encoding="utf-8")

    def test_report_covers_ocr_duplicates_dates_and_bench(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self.build_corpus(tmp)
            result, text = self.run_report(corpus, tmp / "report")

            # 17 input rows minus 1 other-court row and 1 near duplicate.
            self.assertEqual(result["records"]["all"], 15)
            for heading in ("## What the build dropped", "## Coverage by bench", "## Missing and odd dates",
                            "## OCR noise and repeated text (sample)", "## Duplicates", "## Metadata"):
                self.assertIn(heading, text)
            self.assertIn("court_not_selected | 1", text)
            self.assertIn("near_duplicate | 1", text)
            self.assertIn("Nagpur", text)
            self.assertEqual(result["duplicates"]["residual_near_duplicates_in_search_index"], 0)
            self.assertEqual(result["duplicates"]["repeated_cnr_rows_in_output"], 0)
            self.assertTrue(result["near_duplicate_links"]["available"])
            self.assertEqual(result["near_duplicate_links"]["removed_records"], 1)
            self.assertGreaterEqual(result["text_quality"]["ocr_suspect"]["count"], 1)
            self.assertEqual(result["missing_and_odd_dates"]["cnr_date_status"].get("ok"), 15)

            with (tmp / "report" / "ocr_review_sample.csv").open(encoding="utf-8") as handle:
                review = list(csv.DictReader(handle))
            self.assertTrue(any(r["ocr_suspect"] == "True" for r in review))
            self.assertTrue(any(r["ocr_suspect"] == "False" for r in review))

    def test_partial_build_is_flagged_in_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self.build_corpus(tmp, max_rows=3)
            _, text = self.run_report(corpus, tmp / "report")
            self.assertIn("partial build", text)

    def test_profile_comparison_is_added_when_a_profile_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self.build_corpus(tmp)
            profile = tmp / "profile.json"
            profile.write_text(json.dumps({"reasoned_min_chars": 5000, "benches": [{"reasoned_records": 100}],
                                           "supreme_court": {"reasoned_records": 50}}), encoding="utf-8")
            report_script.main(["--corpus-dir", str(corpus), "--out-dir", str(tmp / "r"), "--profile", str(profile)])
            text = (tmp / "r" / "data_quality_report.md").read_text(encoding="utf-8")
            self.assertIn("## Against the earlier profile", text)
            self.assertIn("| Bombay HC | 100 |", text)

    def test_missing_corpus_gives_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                report_script.main(["--corpus-dir", tmp, "--out-dir", tmp])
            self.assertIn("build_corpus_v2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

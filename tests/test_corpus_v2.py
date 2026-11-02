"""
Tests for the v2 corpus rules (plan task C1).

Each rule has a fixture, and dropped records are checked by reason. Small parquet files are
written to a temporary folder to test the script; no network, models or real artifacts.
"""

import datetime as dt
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from legalrag.preprocessing.corpus_v2 import (
    BuildConfig,
    CorpusBuilder,
    DROP_REASONS,
    ReservoirSampler,
    case_type_name,
    collapse_repeated_phrases,
    default_sources,
    parse_case_reference,
    parse_iso_date,
    resolve_date,
    text_fingerprint,
)

TODAY = dt.date(2026, 11, 2)
SOURCES = default_sources()
HC = SOURCES["high_courts"]
SC = SOURCES["supreme_court"]


def body(label: str, words: int = 120) -> str:
    """Distinct filler text: the label makes every body differ from the others."""
    return " ".join(f"{label}{i}" for i in range(words))


def record(cnr="HCBM010000012015_x", text=None, **overrides):
    base = {
        "cnr": cnr,
        "court_code": "27_1",
        "bench": "newas",
        "year": 2015,
        "full_text": text if text is not None else body(cnr),
        "title": "Ram v. State",
        "judge": "A B Sharma",
        "petitioner": "Ram",
        "respondent": "State",
        "decision_date": "2015-06-01",
        "disposal_nature_normalized": "allowed",
        "case_type": "",
    }
    base.update(overrides)
    return base


def builder(**overrides):
    settings = {"min_chars": 200, "today": TODAY}
    settings.update(overrides)
    return CorpusBuilder(BuildConfig(**settings))


class TestCaseType(unittest.TestCase):
    def test_parses_reference_from_title(self):
        self.assertEqual(
            parse_case_reference("WP/1519/2000"), {"code": "WP", "number": "1519", "year": "2000"}
        )

    def test_parses_spaces_and_suffix(self):
        ref = parse_case_reference("Ram v. State  CRWP / 12 / 2015 decided")
        self.assertEqual((ref["code"], ref["number"], ref["year"]), ("CRWP", "12", "2015"))
        self.assertEqual(parse_case_reference("WP(L)/45/2019")["code"], "WP(L)")

    def test_multi_word_code_at_the_start_is_kept_whole(self):
        ref = parse_case_reference("CR. MISC./57199/2022 of A Vs B")
        self.assertEqual(ref, {"code": "CRMISC", "number": "57199", "year": "2022"})
        self.assertEqual(parse_case_reference("CR. MISC. APPLN./12/2019")["code"], "CRMISCAPPLN")

    def test_single_codes_at_the_start_are_unchanged(self):
        self.assertEqual(parse_case_reference("WP/1519/2000")["code"], "WP")
        self.assertEqual(parse_case_reference("WP(L)/12/2010")["code"], "WP(L)")
        self.assertEqual(parse_case_reference("APP/694/1994"), {"code": "APP", "number": "694", "year": "1994"})

    def test_reference_in_the_middle_of_a_title_parses_as_before(self):
        self.assertEqual(parse_case_reference("Order in WP/5/2001")["code"], "WP")
        self.assertEqual(parse_case_reference("Ram v. WP/5/2001")["code"], "WP")
        self.assertEqual(parse_case_reference("Ram Kumar Vs State of Maharashtra CRA/7/2011")["code"], "CRA")

    def test_more_than_three_leading_tokens_is_not_one_code(self):
        self.assertEqual(parse_case_reference("A B C D/9/2001")["code"], "D")

    def test_no_reference_returns_none(self):
        self.assertIsNone(parse_case_reference("Ram v. State of Maharashtra"))
        self.assertIsNone(parse_case_reference(""))
        self.assertIsNone(parse_case_reference(None))
        self.assertIsNone(parse_case_reference("12/2015"))

    def test_names_only_for_bombay_and_known_codes(self):
        self.assertEqual(case_type_name("WP"), "Writ Petition")
        self.assertEqual(case_type_name("WP(L)"), "Writ Petition")
        self.assertIsNone(case_type_name("ZZZ"))
        self.assertIsNone(case_type_name("WP", is_supreme_court=True))


class TestDates(unittest.TestCase):
    def test_parse_iso_date(self):
        self.assertEqual(parse_iso_date("2015-06-01"), dt.date(2015, 6, 1))
        self.assertEqual(parse_iso_date("2015-06-01T10:00:00"), dt.date(2015, 6, 1))
        self.assertIsNone(parse_iso_date("2015-13-01"))
        self.assertIsNone(parse_iso_date("not a date"))
        self.assertIsNone(parse_iso_date(None))

    def test_agreeing_dates(self):
        res = resolve_date("2015-06-01", "HCBM01_9_2015-06-01", today=TODAY)
        self.assertEqual(res.judgment_date, "2015-06-01")
        self.assertFalse(res.conflict)

    def test_prefer_decision_uses_decision_date_on_conflict(self):
        res = resolve_date("2015-06-01", "HCBM01_9_2016-02-03", "prefer_decision", TODAY)
        self.assertEqual((res.judgment_date, res.date_source), ("2015-06-01", "decision_date"))
        self.assertTrue(res.conflict)
        self.assertTrue(res.year_conflict)

    def test_prefer_cnr_uses_cnr_date_on_conflict(self):
        res = resolve_date("2015-06-01", "HCBM01_9_2015-06-09", "prefer_cnr", TODAY)
        self.assertEqual((res.judgment_date, res.date_source), ("2015-06-09", "cnr"))
        self.assertTrue(res.conflict)
        self.assertFalse(res.year_conflict)

    def test_agree_only_leaves_conflicts_empty(self):
        conflict = resolve_date("2015-06-01", "HCBM01_9_2015-06-09", "agree_only", TODAY)
        self.assertIsNone(conflict.judgment_date)
        agree = resolve_date("2015-06-01", "HCBM01_9_2015-06-01", "agree_only", TODAY)
        self.assertEqual(agree.judgment_date, "2015-06-01")

    def test_impossible_cnr_date_is_rejected(self):
        res = resolve_date("2015-06-01", "HCBM01_9_3021-03-24", today=TODAY)
        self.assertEqual(res.cnr_date_status, "implausible")
        self.assertEqual(res.judgment_date, "2015-06-01")
        self.assertFalse(res.conflict)

    def test_impossible_decision_date_falls_back_to_cnr(self):
        res = resolve_date("3021-03-24", "HCBM01_9_2015-06-01", today=TODAY)
        self.assertEqual(res.decision_date_status, "implausible")
        self.assertEqual((res.judgment_date, res.date_source), ("2015-06-01", "cnr"))

    def test_dates_before_1862_and_in_the_future_are_rejected(self):
        self.assertIsNone(resolve_date("1700-01-01", None, today=TODAY).judgment_date)
        self.assertIsNone(resolve_date("2026-11-03", None, today=TODAY).judgment_date)

    def test_missing_and_unparseable(self):
        res = resolve_date("", "no-date-here", today=TODAY)
        self.assertEqual((res.decision_date_status, res.cnr_date_status), ("missing", "missing"))
        self.assertIsNone(res.judgment_date)
        res = resolve_date("garbage", "HCBM01_9_garbage", today=TODAY)
        self.assertEqual((res.decision_date_status, res.cnr_date_status), ("unparseable", "unparseable"))

    def test_unknown_rule_raises(self):
        with self.assertRaises(ValueError):
            resolve_date("2015-06-01", None, rule="newest", today=TODAY)


class TestRepeatedText(unittest.TestCase):
    def test_collapses_consecutive_repeats_and_keeps_one_copy(self):
        unit = "CORAM: A B SHARMA, J."
        text = f"{unit} {unit} {unit} The petition is allowed for the reasons below."
        cleaned, removed = collapse_repeated_phrases(text)
        self.assertEqual(cleaned, f"{unit} The petition is allowed for the reasons below.")
        self.assertEqual(removed, len(text) - len(cleaned))

    def test_many_repeats_collapse_to_one_copy(self):
        unit = "CORAM A B SHARMA J"
        cleaned, _ = collapse_repeated_phrases(" ".join([unit] * 20) + " ok")
        self.assertEqual(cleaned, f"{unit} ok")

    def test_ignores_case_and_punctuation_when_matching(self):
        text = "Coram A B Sharma J now. CORAM: a b sharma, j. now! Heard the matter today."
        cleaned, removed = collapse_repeated_phrases(text, min_words=4)
        self.assertGreater(removed, 0)
        self.assertEqual(cleaned.lower().count("sharma"), 1)

    def test_non_consecutive_repeats_are_kept(self):
        text = "the learned counsel for the petitioner argued. Then the court heard the learned counsel for the petitioner again."
        cleaned, removed = collapse_repeated_phrases(text)
        self.assertEqual((cleaned, removed), (text, 0))

    def test_short_repeats_below_min_words_are_kept(self):
        text = "It was held that that the order stands. Very very good indeed."
        self.assertEqual(collapse_repeated_phrases(text), (text, 0))

    def test_separator_lines_are_not_treated_as_repeats(self):
        text = "Heading - - - - - - - - - - - - Body of the judgment follows here in full."
        self.assertEqual(collapse_repeated_phrases(text), (text, 0))

    def test_only_the_head_is_cleaned(self):
        unit = "ALPHA BETA GAMMA DELTA"
        head = body("filler", 400)
        text = f"{head} {unit} {unit} end"
        cleaned, removed = collapse_repeated_phrases(text, head_chars=500)
        self.assertEqual((cleaned, removed), (text, 0))

    def test_text_after_the_head_is_unchanged(self):
        unit = "ALPHA BETA GAMMA DELTA"
        tail = f"{unit} {unit} untouched"
        text = f"{unit} {unit} {unit} " + body("x", 30) + " " + body("pad", 600) + " " + tail
        cleaned, _ = collapse_repeated_phrases(text, head_chars=300)
        self.assertTrue(cleaned.endswith(tail))

    def test_empty_and_short_inputs(self):
        self.assertEqual(collapse_repeated_phrases(""), ("", 0))
        self.assertEqual(collapse_repeated_phrases("a b"), ("a b", 0))


class TestFingerprint(unittest.TestCase):
    def test_ignores_spacing_case_and_punctuation(self):
        self.assertEqual(text_fingerprint("The Court, held."), text_fingerprint("the  court held"))

    def test_prefix_mode_ignores_differences_after_the_prefix(self):
        a, b = "x" * 3000 + "ending one", "x" * 3000 + "ending two"
        self.assertEqual(text_fingerprint(a, "prefix", 2000), text_fingerprint(b, "prefix", 2000))
        self.assertNotEqual(text_fingerprint(a, "exact"), text_fingerprint(b, "exact"))

    def test_prefix_window_gives_the_same_result_as_normalising_everything(self):
        import hashlib
        import re

        # Mostly punctuation and spaces, so many raw characters are needed for 2,000 kept ones.
        text = ("a ... b ,,, c --- " * 3000) + "tail"
        everything = re.sub(r"\W+", "", text).lower()[:2000]
        self.assertEqual(text_fingerprint(text, "prefix", 2000), hashlib.sha1(everything.encode()).hexdigest())
        short = "tiny text"
        self.assertEqual(text_fingerprint(short, "prefix", 2000), hashlib.sha1(b"tinytext").hexdigest())
        self.assertEqual(text_fingerprint("", "prefix", 2000), hashlib.sha1(b"").hexdigest())

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            text_fingerprint("a", "fuzzy")


class TestReservoirSampler(unittest.TestCase):
    def test_size_and_seen(self):
        sampler = ReservoirSampler(5, seed=1)
        for i in range(100):
            sampler.add(i)
        self.assertEqual((len(sampler.items), sampler.seen), (5, 100))

    def test_small_streams_are_kept_whole(self):
        sampler = ReservoirSampler(10)
        for i in range(3):
            sampler.add(i)
        self.assertEqual(sampler.items, [0, 1, 2])

    def test_same_seed_same_sample(self):
        def run(seed):
            sampler = ReservoirSampler(5, seed)
            for i in range(200):
                sampler.add(i)
            return sampler.items

        self.assertEqual(run(7), run(7))
        self.assertNotEqual(run(7), run(8))


class TestBuilderRules(unittest.TestCase):
    def test_reasoned_record_goes_to_search_index_with_cleaned_text(self):
        b = builder()
        result = b.process(record(), HC)
        self.assertEqual(result.route, "search")
        row = result.row
        self.assertTrue(row["in_search_index"])
        self.assertEqual(row["court_name"], "Bombay High Court")
        self.assertEqual(row["bench_name"], "Bombay (Appellate Side)")
        self.assertTrue(row["bench_name_inferred"])
        self.assertEqual(row["clean_chars"], len(row["text"]))
        self.assertEqual(row["judgment_year"], 2015)

    def test_short_record_is_routed_to_lookup_only_not_dropped(self):
        b = builder()
        result = b.process(record(cnr="HCBM010000022015_x", text="Rule discharged."), HC)
        self.assertEqual(result.route, "lookup_only")
        self.assertFalse(result.row["in_search_index"])
        self.assertEqual(b.stats.kept_lookup_only, 1)
        self.assertEqual(sum(b.stats.dropped.values()), 0)

    def test_cut_uses_cleaned_length(self):
        unit = "CORAM A B SHARMA J"
        padded = " ".join([unit] * 20) + " ok"
        b = builder(min_chars=150)
        self.assertGreaterEqual(len(padded), 150)
        self.assertEqual(b.process(record(text=padded), HC).route, "lookup_only")

    def test_each_drop_reason_is_counted(self):
        b = builder()
        b.process(record(cnr="A1", court_code="10_8"), HC)  # other court
        b.process(record(cnr="  "), HC)  # missing cnr
        b.process(record(cnr="A2"), HC)
        b.process(record(cnr="A2"), HC)  # repeated cnr
        b.process(record(cnr="A3", text="   "), HC)  # empty text
        b.process(record(cnr="A4", full_text=None), HC)  # null text
        b.process(record(cnr="A5", text="\x01" * 50 + "a" * 100), HC)  # control characters
        counts = b.stats.dropped
        self.assertEqual(counts["court_not_selected"], 1)
        self.assertEqual(counts["missing_cnr"], 1)
        self.assertEqual(counts["duplicate_cnr"], 1)
        self.assertEqual(counts["empty_text"], 2)
        self.assertEqual(counts["corrupted_text"], 1)
        self.assertTrue(b.stats.identity_holds())

    def test_near_duplicate_is_dropped_and_linked_to_kept_cnr(self):
        b = builder()
        first = b.process(record(cnr="K1", text=body("same")), HC)
        second = b.process(record(cnr="K2", text=body("same").upper()), HC)
        self.assertEqual(first.route, "search")
        self.assertEqual((second.route, second.drop_reason), ("dropped", "near_duplicate"))
        self.assertEqual((second.duplicate_of, second.cnr), ("K1", "K2"))
        self.assertEqual(b.stats.near_duplicate_links, 1)

    def test_distinct_texts_are_not_treated_as_duplicates(self):
        b = builder()
        self.assertEqual(b.process(record(cnr="K1", text=body("one")), HC).route, "search")
        self.assertEqual(b.process(record(cnr="K2", text=body("two")), HC).route, "search")

    def test_dedup_can_be_switched_off(self):
        b = builder(dedup_mode="off")
        b.process(record(cnr="K1", text=body("same")), HC)
        self.assertEqual(b.process(record(cnr="K2", text=body("same")), HC).route, "search")

    def test_identical_short_orders_are_not_deduplicated(self):
        b = builder()
        a = b.process(record(cnr="S1", text="Adjourned to next week."), HC)
        c = b.process(record(cnr="S2", text="Adjourned to next week."), HC)
        self.assertEqual((a.route, c.route), ("lookup_only", "lookup_only"))
        self.assertEqual(b.stats.near_duplicate_links, 0)

    def test_repeat_cleaning_is_counted(self):
        unit = "CORAM A B SHARMA J"
        b = builder()
        result = b.process(record(text=f"{unit} {unit} {unit} " + body("w", 200)), HC)
        self.assertEqual(b.stats.repeat_cleaned_records, 1)
        self.assertGreater(result.row["repeat_chars_removed"], 0)
        self.assertEqual(result.row["raw_chars"] - result.row["clean_chars"] >= result.row["repeat_chars_removed"], True)

    def test_repeat_cleaning_can_be_switched_off(self):
        unit = "CORAM A B SHARMA J"
        b = builder(clean_repeats=False)
        result = b.process(record(text=f"{unit} {unit} {unit} " + body("w", 200)), HC)
        self.assertEqual(result.row["repeat_chars_removed"], 0)
        self.assertEqual(b.stats.repeat_cleaned_records, 0)

    def test_require_date_drops_reasoned_records_without_a_date(self):
        b = builder(require_date=True)
        undated = record(cnr="D1", decision_date="", text=body("d1"))
        result = b.process(undated, HC)
        self.assertEqual(result.drop_reason, "no_valid_date")
        short = b.process(record(cnr="D2", decision_date="", text="Rule discharged."), HC)
        self.assertEqual(short.route, "lookup_only")

    def test_date_conflict_is_flagged_and_counted_for_kept_rows(self):
        b = builder()
        result = b.process(record(cnr="HCBM01_5_2014-01-01", decision_date="2015-06-01"), HC)
        self.assertTrue(result.row["date_conflict"])
        self.assertTrue(result.row["year_conflict"])
        self.assertEqual((b.stats.date_conflicts, b.stats.year_conflicts), (1, 1))
        self.assertEqual(result.row["judgment_date"], "2015-06-01")

    def test_impossible_cnr_date_does_not_become_the_judgment_date(self):
        b = builder(date_rule="prefer_cnr")
        result = b.process(record(cnr="HCBM01_5_3021-03-24", decision_date="2015-06-01"), HC)
        self.assertEqual(result.row["judgment_date"], "2015-06-01")
        self.assertEqual(b.stats.cnr_date_status["implausible"], 1)

    def test_bench_names_and_unknown_bench(self):
        b = builder()
        self.assertEqual(b.process(record(cnr="B1", bench="hcaurdb"), HC).row["bench_name"], "Aurangabad")
        self.assertEqual(b.process(record(cnr="B2", bench="testcase"), HC).row["bench_name"], "Nagpur")
        unknown = b.process(record(cnr="B3", bench="mystery"), HC).row
        self.assertEqual(unknown["bench_name"], "Unknown (mystery)")
        self.assertEqual(b.stats.unknown_bench["mystery"], 1)

    def test_unknown_bench_falls_back_to_cnr_prefix(self):
        b = builder()
        row = b.process(record(cnr="HCBM03_1_2015-01-01", bench=""), HC).row
        self.assertEqual(row["bench_name"], "Aurangabad")
        self.assertTrue(row["bench_name_inferred"])

    def test_bench_that_disagrees_with_cnr_prefix_is_counted(self):
        b = builder()
        b.process(record(cnr="HCBM03_1_2015-01-01", bench="newas"), HC)
        self.assertEqual(b.stats.bench_cnr_prefix_mismatch, 1)

    def test_case_type_comes_from_dataset_then_title(self):
        b = builder()
        from_title = b.process(record(cnr="C1", title="WP/1519/2000", case_type=""), HC).row
        self.assertEqual(
            (from_title["case_type"], from_title["case_type_source"], from_title["case_type_name"]),
            ("WP", "title", "Writ Petition"),
        )
        self.assertEqual((from_title["case_number"], from_title["case_year"]), ("1519", "2000"))
        from_dataset = b.process(record(cnr="C2", title="WP/1/2000", case_type="Criminal Appeal"), HC).row
        self.assertEqual((from_dataset["case_type"], from_dataset["case_type_source"]), ("Criminal Appeal", "dataset"))
        neither = b.process(record(cnr="C3", title="Ram v. State", case_type=None), HC).row
        self.assertEqual((neither["case_type"], neither["case_type_source"]), (None, "none"))
        self.assertEqual(b.stats.case_type_source, {"title": 1, "dataset": 1, "none": 1})

    def test_multi_word_code_flows_through_to_the_row_and_the_unmapped_list(self):
        b = builder()
        row = b.process(record(cnr="C9", title="CR. MISC./57199/2022 of A Vs B"), HC).row
        self.assertEqual((row["case_type"], row["case_type_code"], row["case_type_name"]), ("CRMISC", "CRMISC", None))
        self.assertEqual((row["case_number"], row["case_year"]), ("57199", "2022"))
        self.assertEqual(b.stats.unmapped_case_codes["CRMISC"], 1)

    def test_unmapped_case_codes_are_listed(self):
        b = builder()
        b.process(record(cnr="C1", title="ZZQ/9/2001"), HC)
        self.assertEqual(b.stats.unmapped_case_codes["ZZQ"], 1)

    def test_supreme_court_records_keep_all_courts_and_skip_case_type_names(self):
        b = builder()
        row = b.process(
            record(cnr="SC1", court_code=None, bench=None, title="CA/100/1990"), SC
        ).row
        self.assertEqual(row["court_name"], "Supreme Court of India")
        self.assertEqual(row["court_code"], "SC")
        self.assertEqual(row["bench_name"], "Supreme Court of India")
        self.assertIsNone(row["case_type_name"])
        self.assertEqual(b.stats.unknown_bench, {})

    def test_blank_metadata_becomes_none(self):
        b = builder()
        row = b.process(record(judge="  ", petitioner="", title=None), HC).row
        self.assertEqual((row["judge"], row["petitioner"], row["title"]), (None, None, None))

    def test_prefiltered_rows_reconcile(self):
        b = builder()
        b.note_prefiltered("high_courts", 40)
        b.process(record(), HC)
        self.assertEqual(b.stats.scanned, 41)
        self.assertEqual(b.stats.dropped["court_not_selected"], 40)
        self.assertTrue(b.stats.identity_holds())

    def test_report_lists_every_drop_reason(self):
        report = builder().stats.to_dict()
        self.assertEqual(list(report["dropped_by_reason"]), list(DROP_REASONS))

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            BuildConfig(date_rule="newest")
        with self.assertRaises(ValueError):
            BuildConfig(dedup_mode="fuzzy")


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_corpus_v2.py"
    spec = importlib.util.spec_from_file_location("build_corpus_v2_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBuildScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pyarrow  # noqa: F401
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("pyarrow is not installed")
        cls.script = _load_script()

    def _write_parquet(self, path: Path, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), path)

    def _hc_rows(self):
        return [
            record(cnr="HCBM01_1_2015-06-01", text=body("a")),
            record(cnr="HCBM01_2_2015-06-02", text=body("a")),  # near duplicate of the first
            record(cnr="HCBM01_3_2015-06-03", text="Rule discharged."),
            record(cnr="HCBM01_4_2015-06-04", text="   "),
            record(cnr="HCBM01_5_2014-01-01", text=body("b"), decision_date="2015-06-05"),  # date conflict
            record(cnr="HCBM01_6_2015-06-06", text=body("c"), court_code="10_8"),  # other court
        ]

    def test_build_writes_outputs_and_reconciles_counts(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hc_file = tmp / "hc.parquet"
            self._write_parquet(hc_file, self._hc_rows())
            out_dir = tmp / "artifacts" / "v2" / "corpus"
            report = self.script.build(
                sources={"high_courts": [hc_file], "supreme_court": []},
                out_dir=out_dir,
                config=BuildConfig(min_chars=200, today=TODAY),
            )

            stats = report["stats"]
            self.assertEqual(stats["scanned"], 6)
            self.assertTrue(stats["identity_holds"])
            self.assertEqual(stats["dropped_by_reason"]["court_not_selected"], 1)
            self.assertEqual(stats["dropped_by_reason"]["near_duplicate"], 1)
            self.assertEqual(stats["dropped_by_reason"]["empty_text"], 1)
            self.assertEqual((stats["kept_search_index"], stats["kept_lookup_only"]), (2, 1))

            search = pq.read_table(out_dir / "search_corpus.parquet").to_pylist()
            lookup = pq.read_table(out_dir / "lookup_only.parquet").to_pylist()
            links = pq.read_table(out_dir / "duplicate_links.parquet").to_pylist()
            self.assertEqual(len(search), 2)
            self.assertEqual([row["cnr"] for row in lookup], ["HCBM01_3_2015-06-03"])
            self.assertEqual(
                links, [{"duplicate_cnr": "HCBM01_2_2015-06-02", "kept_cnr": "HCBM01_1_2015-06-01", "source_config": "high_courts"}]
            )
            self.assertTrue(all(row["in_search_index"] for row in search))
            self.assertEqual(set(report["outputs"]), {"search_corpus_rows", "lookup_only_rows", "duplicate_links_rows"})

            saved = json.loads((out_dir / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["stats"]["scanned"], 6)
            self.assertFalse(saved["partial_build"])
            # The second kept row and the fifth row both have decision dates that differ from the CNR.
            self.assertEqual(saved["samples"]["date_disagreements_seen"], 2)
            self.assertTrue((out_dir / "date_disagreements_sample.csv").exists())
            self.assertIn("your_label", (out_dir / "short_orders_sample.csv").read_text(encoding="utf-8"))

    def test_supreme_court_without_court_code_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sc_file = tmp / "sc.parquet"
            rows = [
                {"cnr": "SC_1", "full_text": body("s1"), "title": "CA/1/1990", "decision_date": "1990-01-01", "year": 1990},
                {"cnr": "SC_2", "full_text": body("s2"), "title": "X v. Y", "decision_date": "", "year": 1991},
            ]
            self._write_parquet(sc_file, rows)
            report = self.script.build(
                sources={"high_courts": [], "supreme_court": [sc_file]},
                out_dir=tmp / "v2" / "out",
                config=BuildConfig(min_chars=200, today=TODAY),
            )
            self.assertEqual(report["stats"]["kept_search_index"], 2)
            self.assertEqual(report["stats"]["by_source"]["supreme_court"]["kept_search"], 2)

    def test_max_rows_marks_the_build_as_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hc_file = tmp / "hc.parquet"
            self._write_parquet(hc_file, self._hc_rows())
            report = self.script.build(
                sources={"high_courts": [hc_file], "supreme_court": []},
                out_dir=tmp / "v2" / "smoke",
                config=BuildConfig(min_chars=200, today=TODAY),
                max_rows=2,
            )
            self.assertTrue(report["partial_build"])
            self.assertEqual(report["stats"]["kept_search_index"], 1)
            self.assertTrue(report["stats"]["identity_holds"])

    def test_output_directory_must_be_a_v2_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hc_file = tmp / "hc.parquet"
            self._write_parquet(hc_file, self._hc_rows())
            with self.assertRaises(SystemExit):
                self.script.build(
                    sources={"high_courts": [hc_file], "supreme_court": []},
                    out_dir=tmp / "artifacts",
                    config=BuildConfig(min_chars=200, today=TODAY),
                )
            self.assertFalse((tmp / "artifacts").exists())

    def test_existing_outputs_are_not_overwritten_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hc_file = tmp / "hc.parquet"
            self._write_parquet(hc_file, self._hc_rows())
            kwargs = dict(
                sources={"high_courts": [hc_file], "supreme_court": []},
                out_dir=tmp / "v2" / "corpus",
                config=BuildConfig(min_chars=200, today=TODAY),
            )
            self.script.build(**kwargs)
            with self.assertRaises(SystemExit):
                self.script.build(**kwargs)
            self.script.build(overwrite=True, **kwargs)

    def test_expand_inputs_handles_files_folders_and_globs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "sub").mkdir()
            for name in ("b.parquet", "a.parquet", "sub/c.parquet", "notes.txt"):
                (tmp / name).write_bytes(b"")
            by_dir = self.script.expand_inputs([str(tmp)])
            self.assertEqual([p.name for p in by_dir], ["a.parquet", "b.parquet", "c.parquet"])
            by_glob = self.script.expand_inputs([str(tmp / "*.parquet")])
            self.assertEqual([p.name for p in by_glob], ["a.parquet", "b.parquet"])
            by_file = self.script.expand_inputs([str(tmp / "a.parquet"), str(tmp / "a.parquet")])
            self.assertEqual(len(by_file), 1)


class TestReportedRates(unittest.TestCase):
    def test_rates_are_computed_from_the_counts(self):
        b = builder()
        for i in range(3):  # three distinct reasoned records, the first with a date conflict
            conflict = {"cnr": f"HCBM01_{i}_2014-01-01", "decision_date": "2015-06-01"} if i == 0 else {}
            b.process(record(cnr=conflict.get("cnr", f"R{i}"), text=body(f"r{i}"), **{k: v for k, v in conflict.items() if k != "cnr"}), HC)
        b.process(record(cnr="DUP", text=body("r1")), HC)  # near duplicate of R1: dropped
        b.process(record(cnr="SHORT", text="Rule discharged."), HC)  # lookup only
        stats = b.stats.to_dict()
        self.assertEqual((stats["kept_search_index"], stats["dropped_by_reason"]["near_duplicate"], stats["kept_lookup_only"]), (3, 1, 1))
        self.assertEqual(stats["near_duplicate_share_of_reasoned"], 1 / 4)  # 1 / (3 kept + 1 dropped)
        self.assertEqual(stats["date_conflicts"], 1)
        self.assertEqual(stats["date_conflict_share_of_kept"], 1 / 4)  # 1 / (3 search + 1 lookup)

    def test_a_dropped_near_duplicate_does_not_count_as_a_kept_date_conflict(self):
        b = builder()
        b.process(record(cnr="K1", text=body("same")), HC)
        b.process(record(cnr="HCBM01_9_2014-01-01", text=body("same"), decision_date="2015-06-01"), HC)  # dropped, conflicting
        stats = b.stats.to_dict()
        self.assertEqual(stats["date_conflicts"], 0)
        self.assertEqual(stats["date_conflict_share_of_kept"], 0.0)
        self.assertEqual(stats["near_duplicate_share_of_reasoned"], 1 / 2)

    def test_division_by_zero_gives_none(self):
        empty = builder().stats.to_dict()
        self.assertIsNone(empty["near_duplicate_share_of_reasoned"])
        self.assertIsNone(empty["date_conflict_share_of_kept"])

        only_short = builder()
        only_short.process(record(cnr="S1", text="Adjourned."), HC)
        stats = only_short.stats.to_dict()
        self.assertIsNone(stats["near_duplicate_share_of_reasoned"])  # nothing reasoned
        self.assertEqual(stats["date_conflict_share_of_kept"], 0.0)

        nothing_kept = builder()
        nothing_kept.process(record(cnr="E1", text="   "), HC)  # dropped as empty text
        self.assertIsNone(nothing_kept.stats.to_dict()["date_conflict_share_of_kept"])

    def test_rates_are_json_serialisable_with_null(self):
        self.assertIn('"near_duplicate_share_of_reasoned": null', json.dumps(builder().stats.to_dict()))


class TestRatesInBuildOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pyarrow  # noqa: F401
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("pyarrow is not installed")
        cls.script = _load_script()

    def build(self, tmp, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq

        source = tmp / "hc.parquet"
        pq.write_table(pa.Table.from_pylist(rows), source)
        return self.script.build({"high_courts": [source], "supreme_court": []}, tmp / "v2" / "o",
                                 BuildConfig(min_chars=200, today=TODAY))

    def test_rates_are_in_build_report_json_and_the_printed_summary(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [
                record(cnr="HCBM01_1_2015-06-01", text=body("a")),
                record(cnr="HCBM01_2_2015-06-01", text=body("a")),  # near duplicate
                record(cnr="HCBM01_3_2014-01-01", text=body("b"), decision_date="2015-06-05"),  # date conflict
                record(cnr="HCBM01_4_2015-06-01", text="Rule discharged."),  # lookup only
            ]
            report = self.build(tmp, rows)
            saved = json.loads((tmp / "v2" / "o" / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["stats"]["near_duplicate_share_of_reasoned"], 1 / 3)  # 1 / (2 kept + 1 dropped)
            self.assertEqual(saved["stats"]["date_conflict_share_of_kept"], 1 / 3)  # 1 / (2 search + 1 lookup)
            self.assertEqual(report["stats"]["date_conflict_share_of_kept"], 1 / 3)

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.script.print_summary(report)
            text = buffer.getvalue()
            self.assertIn("near-duplicate share of reasoned", text)
            self.assertIn("0.3333 (33.33%)", text)
            self.assertIn("date-conflict share of kept", text)

    def test_summary_prints_n_a_when_a_rate_is_undefined(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            report = self.build(tmp, [record(cnr="HCBM01_1_2015-06-01", text="   ")])
            self.assertIsNone(report["stats"]["date_conflict_share_of_kept"])
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.script.print_summary(report)
            self.assertEqual(buffer.getvalue().count("n/a"), 2)


class TestSchemaChecks(unittest.TestCase):
    """The first run on the Supreme Court config must fail clearly, or work with the columns it has."""

    @classmethod
    def setUpClass(cls):
        try:
            import pyarrow  # noqa: F401
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("pyarrow is not installed")
        cls.script = _load_script()

    def write(self, path: Path, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def sc_rows(self):
        # No bench, court_code, judge or case_type columns at all.
        return [
            {"cnr": "SC_1", "full_text": body("s1"), "title": "Ram v. State", "decision_date": "1990-01-01", "year": 1990},
            {"cnr": "SC_2", "full_text": "Dismissed.", "title": "X v. Y", "decision_date": "", "year": 1991},
        ]

    def run_check(self, *argv):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as ctx:
                self.script.main(list(argv))
        return ctx.exception.code, buffer.getvalue()

    def test_supreme_court_file_without_optional_columns_still_builds(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sc = self.write(tmp / "sc.parquet", self.sc_rows())
            out = tmp / "v2" / "out"
            report = self.script.build(
                {"high_courts": [], "supreme_court": [sc]}, out, BuildConfig(min_chars=200, today=TODAY)
            )
            stats = report["stats"]
            self.assertEqual((stats["scanned"], stats["kept_search_index"], stats["kept_lookup_only"]), (2, 1, 1))
            self.assertTrue(stats["identity_holds"])
            rows = pq.read_table(out / "search_corpus.parquet").to_pylist() + pq.read_table(out / "lookup_only.parquet").to_pylist()
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["bench_name"], "Supreme Court of India")
                self.assertEqual((row["court_code"], row["court_name"]), ("SC", "Supreme Court of India"))
                self.assertIsNone(row["bench"])
                self.assertIsNone(row["judge"])
                self.assertIsNone(row["case_type_name"])
            self.assertEqual({r["case_type_source"] for r in rows}, {"none"})
            self.assertEqual(stats["unknown_bench"], {})

    def test_file_without_cnr_fails_clearly_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [{k: v for k, v in r.items() if k != "cnr"} for r in self.sc_rows()]
            sc = self.write(tmp / "sc.parquet", rows)
            out = tmp / "v2" / "out"
            with self.assertRaises(SystemExit) as ctx:
                self.script.build({"high_courts": [], "supreme_court": [sc]}, out, BuildConfig(min_chars=200, today=TODAY))
            message = str(ctx.exception)
            self.assertIn("supreme_court", message)
            self.assertIn("'cnr'", message)
            self.assertIn("sc.parquet", message)
            self.assertIn("Columns found: full_text, title, decision_date, year", message)
            self.assertFalse(out.exists())  # no empty outputs left behind to block the next run

    def test_high_courts_file_without_court_code_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hc = self.write(tmp / "hc.parquet", [{"cnr": "HCBM01_1_2015-01-01", "full_text": body("a")}])
            with self.assertRaises(SystemExit) as ctx:
                self.script.build({"high_courts": [hc], "supreme_court": []}, tmp / "v2" / "o", BuildConfig(min_chars=200, today=TODAY))
            self.assertIn("'court_code'", str(ctx.exception))
            self.assertIn("Columns found: cnr, full_text", str(ctx.exception))

    def test_reader_message_lists_the_columns_it_found(self):
        # The reader keeps its own check as a second line of defence.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hc = self.write(tmp / "hc.parquet", [{"cnr": "HCBM01_1_2015-01-01", "full_text": body("a")}])
            builder_ = CorpusBuilder(BuildConfig(min_chars=200, today=TODAY))
            with self.assertRaises(SystemExit) as ctx:
                list(self.script.iter_source_records([hc], HC, builder_, None))
            self.assertIn("Columns found: cnr, full_text", str(ctx.exception))

    def test_check_schema_reports_files_rows_columns_and_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sc = self.write(tmp / "sc.parquet", self.sc_rows())
            code, text = self.run_check("--supreme-court", str(sc), "--check-schema")
            self.assertEqual(code, 0)
            self.assertIn("[supreme_court] 1 file(s), 2 rows", text)
            self.assertIn(str(sc), text)
            self.assertIn("cnr: string", text)
            self.assertIn("year: int64", text)
            missing_line = next(l for l in text.splitlines() if "expected columns not present" in l)
            for column in ("court_code", "bench", "judge", "case_type"):
                self.assertIn(column, missing_line)
            self.assertNotIn("cnr", missing_line)
            self.assertIn("required 'cnr': present", text)
            self.assertIn("required 'full_text': present", text)
            self.assertNotIn("PROBLEM", text)

    def test_check_schema_flags_missing_required_columns_with_a_failing_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            hc = self.write(tmp / "hc.parquet", [{"cnr": "HCBM01_1_2015-01-01", "full_text": body("a")}])
            sc = self.write(tmp / "sc.parquet", [{"full_text": "x"}])
            code, text = self.run_check("--high-courts", str(hc), "--supreme-court", str(sc), "--check-schema")
            self.assertEqual(code, 1)
            self.assertIn("required 'court_code': MISSING", text)
            self.assertIn("required 'cnr': MISSING", text)
            self.assertIn("PROBLEM: high_courts: required column 'court_code'", text)

    def test_check_schema_notes_columns_that_differ_between_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = self.write(tmp / "a.parquet", self.sc_rows())
            second = self.write(tmp / "b.parquet", [{k: v for k, v in r.items() if k != "title"} for r in self.sc_rows()])
            code, text = self.run_check("--supreme-court", str(first), str(second), "--check-schema")
            self.assertEqual(code, 0)
            self.assertIn("2 file(s), 4 rows", text)
            self.assertIn("columns missing from some files: title", text)

    def test_check_schema_writes_nothing_and_reads_no_text(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sc = self.write(tmp / "sc.parquet", self.sc_rows())
            before = sorted(p.name for p in tmp.rglob("*"))
            with mock.patch.object(pq, "read_table", side_effect=AssertionError("text was read")), \
                    mock.patch.object(pq.ParquetFile, "read", side_effect=AssertionError("text was read")), \
                    mock.patch.object(pq.ParquetFile, "iter_batches", side_effect=AssertionError("text was read")):
                code, _ = self.run_check("--supreme-court", str(sc), "--check-schema", "--out-dir", str(tmp / "v2" / "never"))
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in tmp.rglob("*")), before)


if __name__ == "__main__":
    unittest.main()

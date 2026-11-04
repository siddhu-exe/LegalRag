"""Tests for case lookup across all records (plan task C7): the index, its builder and GET /lookup."""

import ast
import contextlib
import datetime as dt
import importlib.util
import io
import json
import re
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from legalrag.api import lookup as lookup_api
from legalrag.api.config import Settings, get_settings
from legalrag.lookup import CaseLookupIndex, LookupError_, LookupQuery, parse_case_query
from legalrag.lookup.index import DISCLAIMER, MAX_LIMIT, NOT_FOUND
from legalrag.preprocessing.corpus_v2 import BuildConfig

# Words that would say or imply a case does not exist. A "not found" answer must never use them.
FORBIDDEN = re.compile(r"exist|no such case|not a real|non-?existent|fake|invalid case|never (?:happened|decided|filed)", re.IGNORECASE)


def row(cnr, **overrides):
    base = {
        "cnr": cnr, "title": "WP/1519/2015", "petitioner": "Ram Kumar Patil", "respondent": "State of Maharashtra",
        "judge": "A B Sharma", "judgment_date": "2015-06-01", "judgment_year": 2015, "court_name": "Bombay High Court",
        "bench_name": "Nagpur", "case_type_code": "WP", "case_type_name": "Writ Petition", "case_number": "1519",
        "case_year": "2015", "outcome": "allowed", "in_search_index": True, "date_conflict": False,
        "date_source": "decision_date", "source_config": "high_courts",
    }
    base.update(overrides)
    return base


ROWS = [
    row("HCBM04_1_2015-06-01"),
    row("HCBM04_2_2016-02-10", title="WP(L)/1519/2015", case_type_code="WP(L)", petitioner="Meera Joshi", judge="C D Rao",
        judgment_date="2016-02-10", judgment_year=2016),
    row("HCBM01_3_2020-01-05", title="CRA/7/2020", case_type_code="CRA", case_type_name="Criminal Appeal", case_number="7",
        case_year="2020", petitioner="Sita Devi", respondent="State", judge=None, judgment_date="2020-01-05",
        judgment_year=2020, bench_name="Bombay (Appellate Side)", in_search_index=False, date_conflict=True),
    row("HCBM03_4_2018-09-30", title="FA/88/2018", case_type_code="FA", case_type_name="First Appeal", case_number="88",
        case_year="2018", petitioner="Abdul Rahman", respondent="Pune Municipal Corporation", judge="E F Kulkarni",
        judgment_date="2018-09-30", judgment_year=2018, bench_name="Aurangabad", outcome="dismissed"),
    row("SC_5", title="CA/100/1990", court_name="Supreme Court of India", bench_name="Supreme Court of India",
        case_type_code=None, case_type_name=None, case_number="100", case_year="1990", petitioner="Union of India",
        respondent="K Venkat", judge="G H Reddy", judgment_date="1990-01-01", judgment_year=1990, source_config="supreme_court"),
]
LINKS = [{"duplicate_cnr": "HCBM04_9_2015-06-02", "kept_cnr": "HCBM04_1_2015-06-01"}]


def build(tmp, rows=ROWS, links=LINKS):
    return CaseLookupIndex.build(Path(tmp) / "v2" / "lookup.sqlite", rows, links, batch_rows=2)


def find(index, **criteria):
    return index.search(LookupQuery(**criteria))


def cnrs(result):
    return [r["cnr"] for r in result.results]


class TestSearching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index = build(self.tmp.name)
        self.addCleanup(self.index.close)

    def test_by_cnr_ignoring_case_and_spaces(self):
        for text in ("HCBM04_1_2015-06-01", "hcbm04_1_2015-06-01", "  HCBM04_1_2015-06-01  ", "HCBM04_1_ 2015-06-01"):
            with self.subTest(text=text):
                result = find(self.index, cnr=text)
                self.assertEqual((result.status, cnrs(result)), ("found", ["HCBM04_1_2015-06-01"]))

    def test_record_fields(self):
        record = find(self.index, cnr="HCBM04_1_2015-06-01").results[0]
        self.assertEqual((record["case_reference"], record["bench_name"], record["case_type_name"]), ("WP/1519/2015", "Nagpur", "Writ Petition"))
        self.assertEqual((record["in_search_index"], record["date_conflict"], record["judge_display"]), (True, False, "A B Sharma"))
        short = find(self.index, cnr="HCBM01_3_2020-01-05").results[0]
        self.assertEqual((short["in_search_index"], short["date_conflict"], short["judge"], short["judge_display"]), (False, True, None, "not recorded"))
        self.assertNotIn("text", record)  # metadata only

    def test_by_case_reference_in_the_forms_a_lawyer_writes_it(self):
        for text in ("WP/1519/2015", "wp / 1519 / 2015", "WP 1519 of 2015", "Writ Petition No. 1519 of 2015", "writ petition 1519/2015"):
            with self.subTest(text=text):
                self.assertEqual(cnrs(find(self.index, case_ref=text)), ["HCBM04_1_2015-06-01"])

    def test_an_exact_case_type_is_preferred_and_sub_types_are_only_a_fallback(self):
        # WP and WP(L) are separate series: a search for one does not return the other when the exact type exists.
        self.assertEqual(cnrs(find(self.index, case_ref="WP/1519/2015")), ["HCBM04_1_2015-06-01"])
        self.assertEqual(cnrs(find(self.index, case_ref="WP(L)/1519/2015")), ["HCBM04_2_2016-02-10"])
        self.assertEqual(find(self.index, case_ref="WP/1519/2015").warnings, [])
        # With no type given both series match.
        self.assertEqual(sorted(cnrs(find(self.index, case_ref="1519 of 2015"))), ["HCBM04_1_2015-06-01", "HCBM04_2_2016-02-10"])

    def test_sub_types_are_tried_when_nothing_matches_the_exact_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = build(tmp, [row("L1", case_type_code="WP(L)", case_number="55", case_year="2012", title="WP(L)/55/2012")], [])
            result = find(index, case_ref="WP/55/2012")
            self.assertEqual(cnrs(result), ["L1"])
            self.assertTrue(any("sub-types" in w for w in result.warnings))
            self.assertEqual(find(index, case_ref="WP(L)/55/2012").warnings, [])
            self.assertEqual(find(index, case_ref="FA/55/2012").status, "not_in_our_data")  # an unrelated type is not widened
            index.close()

    def test_leading_zeros_in_the_data_or_the_query_do_not_matter(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = build(tmp, [row("A1", case_number="01519"), row("A2", case_number=" 2 ", case_year="2015", title="WP/2/2015")], [])
            self.assertEqual(cnrs(find(index, case_ref="WP/001519/2015")), ["A1"])
            self.assertEqual(cnrs(find(index, case_ref="WP/1519/2015")), ["A1"])
            self.assertEqual(cnrs(find(index, case_ref="WP/2/2015")), ["A2"])  # stray spaces in the data are stripped
            index.close()

    def test_unrecognised_case_type_still_matches_number_and_year_with_a_warning(self):
        result = find(self.index, case_ref="XYZ 7 of 2020")
        self.assertEqual(cnrs(result), ["HCBM01_3_2020-01-05"])
        self.assertTrue(any("case type" in w for w in result.warnings))

    def test_unreadable_case_reference_is_a_request_error(self):
        for bad in ("hello", "1519", "WP/abc/2015"):
            with self.subTest(bad=bad):
                with self.assertRaises(LookupError_):
                    find(self.index, case_ref=bad)

    def test_by_names_with_prefixes_and_every_word_required(self):
        self.assertEqual(cnrs(find(self.index, title="ram kumar")), ["HCBM04_1_2015-06-01"])
        self.assertEqual(cnrs(find(self.index, title="Ram Ku")), ["HCBM04_1_2015-06-01"])  # a word prefix
        self.assertEqual(cnrs(find(self.index, title="Ram Kumar v. State of Maharashtra")), ["HCBM04_1_2015-06-01"])
        self.assertEqual(cnrs(find(self.index, title="ram sita")), [])  # both words are required
        self.assertEqual(cnrs(find(self.index, party="Pune Municipal")), ["HCBM03_4_2018-09-30"])
        self.assertEqual(cnrs(find(self.index, party="rahman")), ["HCBM03_4_2018-09-30"])
        self.assertEqual(cnrs(find(self.index, title="WP/1519")), cnrs(find(self.index, title="WP 1519")))  # punctuation is just a separator

    def test_party_search_ignores_the_title_and_judge_columns(self):
        self.assertEqual(cnrs(find(self.index, party="Sharma")), [])  # a judge, not a party
        self.assertEqual(cnrs(find(self.index, judge="Patil")), [])  # a party, not a judge

    def test_by_judge(self):
        self.assertEqual(cnrs(find(self.index, judge="sharma")), ["HCBM04_1_2015-06-01"])
        self.assertEqual(cnrs(find(self.index, judge="A B Sharma")), ["HCBM04_1_2015-06-01"])
        self.assertEqual(find(self.index, judge="nobody").status, "not_in_our_data")

    def test_by_date_range_and_year(self):
        self.assertEqual(cnrs(find(self.index, date="2018-09-30")), ["HCBM03_4_2018-09-30"])
        self.assertEqual(cnrs(find(self.index, date_from="2016-01-01", date_to="2018-12-31")), ["HCBM03_4_2018-09-30", "HCBM04_2_2016-02-10"])
        self.assertEqual(cnrs(find(self.index, date_from="2020-01-05")), ["HCBM01_3_2020-01-05"])
        self.assertEqual(cnrs(find(self.index, year=1990)), ["SC_5"])
        self.assertEqual(find(self.index, date="2001-01-01").status, "not_in_our_data")

    def test_by_court_and_bench_with_wildcard_characters_taken_literally(self):
        self.assertEqual(cnrs(find(self.index, court="supreme")), ["SC_5"])
        self.assertEqual(len(find(self.index, court="bombay high").results), 4)
        self.assertEqual(cnrs(find(self.index, bench="aurangabad")), ["HCBM03_4_2018-09-30"])
        for wildcard in ("%", "_", "b%y", "\\"):
            with self.subTest(wildcard=wildcard):
                self.assertEqual(find(self.index, bench=wildcard).status, "not_in_our_data")

    def test_criteria_combine_with_and(self):
        self.assertEqual(cnrs(find(self.index, court="bombay", judge="sharma", year=2015)), ["HCBM04_1_2015-06-01"])
        self.assertEqual(find(self.index, judge="sharma", year=2020).status, "not_in_our_data")
        self.assertEqual(cnrs(find(self.index, case_ref="CRA/7/2020", title="sita", bench="appellate")), ["HCBM01_3_2020-01-05"])

    def test_results_come_newest_first_and_paginate(self):
        everything = cnrs(find(self.index, court="bombay"))
        self.assertEqual(everything, ["HCBM01_3_2020-01-05", "HCBM03_4_2018-09-30", "HCBM04_2_2016-02-10", "HCBM04_1_2015-06-01"])
        page = find(self.index, court="bombay", limit=2, offset=1)
        self.assertEqual((page.total, cnrs(page)), (4, everything[1:3]))
        self.assertEqual(cnrs(find(self.index, court="bombay", limit=2, offset=3)), everything[3:])
        self.assertEqual(find(self.index, court="bombay", offset=50).results, [])

    def test_records_without_a_date_sort_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = build(tmp, [row("N1", judgment_date=None, judgment_year=None), row("D1")], [])
            self.assertEqual(cnrs(find(index, court="bombay")), ["D1", "N1"])
            index.close()

    def test_a_request_needs_a_criterion_and_valid_values(self):
        for kwargs, pattern in (
            ({}, "at least one"),
            ({"cnr": "   "}, "at least one"),
            ({"date": "15/06/2015"}, "YYYY-MM-DD"),
            ({"date_from": "2020-02-30"}, "YYYY-MM-DD"),
            ({"date_from": "2020-02-01", "date_to": "2020-01-01"}, "later than"),
            ({"year": 20, "cnr": "x"}, "year"),
            ({"cnr": "x", "limit": 0}, "limit"),
            ({"cnr": "x", "limit": MAX_LIMIT + 1}, "limit"),
            ({"cnr": "x", "offset": -1}, "offset"),
            ({"title": "x" * 300}, "longer than"),
            ({"title": "v. vs"}, "two or more letters"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(LookupError_, pattern):
                    find(self.index, **kwargs)

    def test_hostile_text_is_just_text(self):
        hostile = ['" OR 1=1 --', "'; DROP TABLE cases; --", "ram* NEAR(kumar patil, 1)", "(((", '"unbalanced', "title : x", "AND OR NOT", "ram OR sita", "\x00", "ram\"kumar"]
        for text in hostile:
            for field in ("title", "party", "judge", "cnr", "court", "case_ref"):
                with self.subTest(text=text, field=field):
                    try:
                        result = find(self.index, **{field: text})
                    except LookupError_:
                        continue  # a clean request error is fine; a database error is not
                    self.assertIn(result.status, ("found", "not_in_our_data"))
        self.assertEqual(len(self.index), 5)  # nothing was dropped
        self.assertEqual(find(self.index, title="ram OR sita").status, "not_in_our_data")  # OR is a word, not an operator

    def test_non_latin_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = build(tmp, [row("H1", petitioner="राम कुमार पाटील"), row("H2", petitioner="Sita Devi", respondent="State")], [])
            # Indic words are indexed as letter pieces, so matching is looser than for English names.
            self.assertEqual(cnrs(find(index, party="राम कुमार")), ["H1"])
            self.assertEqual(find(index, party="सीता").status, "not_in_our_data")
            index.close()

    def test_safe_to_use_from_several_threads(self):
        outcomes, errors = [], []

        def worker(n):
            try:
                outcomes.append(find(self.index, year=2015 + n % 2).total)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(12)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(sorted(set(outcomes)), [1])  # one 2015 record, one 2016 record

    def test_the_index_is_read_only(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.index._connection().execute("DELETE FROM cases")


class TestDuplicateCnrs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index = build(self.tmp.name)
        self.addCleanup(self.index.close)

    def test_a_merged_cnr_resolves_to_the_kept_record_and_says_so(self):
        result = find(self.index, cnr="hcbm04_9_2015-06-02")
        self.assertEqual((result.status, cnrs(result)), ("found", ["HCBM04_1_2015-06-01"]))
        self.assertEqual(result.results[0]["matched_via_duplicate_cnr"], "HCBM04_9_2015-06-02")
        self.assertIn("merged into a near-identical judgment", result.message)
        self.assertIsNone(find(self.index, cnr="HCBM04_1_2015-06-01").results[0]["matched_via_duplicate_cnr"])

    def test_other_criteria_still_apply_through_the_link(self):
        self.assertEqual(find(self.index, cnr="HCBM04_9_2015-06-02", judge="sharma").status, "found")
        self.assertEqual(find(self.index, cnr="HCBM04_9_2015-06-02", judge="rao").status, "not_in_our_data")
        self.assertEqual(find(self.index, cnr="HCBM04_9_2015-06-02", year=2015).status, "found")
        self.assertEqual(find(self.index, cnr="HCBM04_9_2015-06-02", year=2001).status, "not_in_our_data")

    def test_an_unknown_cnr_is_not_in_our_data(self):
        self.assertEqual(find(self.index, cnr="HCBM99_1_2000-01-01").status, "not_in_our_data")


class TestBuild(unittest.TestCase):
    def test_counts_coverage_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = build(tmp)
            self.assertEqual(len(index), 5)
            coverage = {c["court"]: c for c in index.coverage()}
            self.assertEqual(coverage["Bombay High Court"]["records"], 4)
            self.assertEqual((coverage["Bombay High Court"]["first_date"], coverage["Bombay High Court"]["last_date"]), ("2015-06-01", "2020-01-05"))
            self.assertEqual(coverage["Supreme Court of India"]["records"], 1)
            self.assertEqual(index.meta["links"], "1")
            index.close()

    def test_bad_input_is_refused_without_leaving_files(self):
        for rows, links, pattern in (
            ([row("A"), row("a")], [], "twice"),  # the same CNR in another case
            ([{"title": "no cnr"}], [], "no cnr"),
            ([], [], "no rows"),
        ):
            with self.subTest(pattern=pattern):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaisesRegex(ValueError, pattern):
                        build(tmp, rows, links)
                    self.assertEqual([p for p in Path(tmp).rglob("*") if p.is_file()], [])

    def test_opening_something_else_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                CaseLookupIndex(Path(tmp) / "none.sqlite")
            other = Path(tmp) / "other.sqlite"
            sqlite3.connect(other).execute("CREATE TABLE cases (x)").connection.commit()
            with self.assertRaisesRegex(ValueError, "not a case lookup index"):
                CaseLookupIndex(other)


# --------------------------------------------------------------------------- wording


class TestNotInOurDataWording(unittest.TestCase):
    """A search with no match must never say or imply that a case does not exist."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index = build(self.tmp.name)
        self.addCleanup(self.index.close)

    def test_no_match_says_not_in_our_data_and_states_the_coverage(self):
        misses = [dict(cnr="HCBM99_1_2000-01-01"), dict(case_ref="WP/9999/2030"), dict(title="zzzz qqqq"), dict(judge="nobody"),
                  dict(date="2040-01-01"), dict(year=2099), dict(court="privy council"), dict(party="x" * 20)]
        for criteria in misses:
            with self.subTest(criteria=criteria):
                result = find(self.index, **criteria)
                self.assertEqual((result.status, result.total, result.results), ("not_in_our_data", 0, []))
                self.assertTrue(result.message.startswith(NOT_FOUND + ":"), result.message)
                self.assertIn("Bombay High Court (2015-06-01 to 2020-01-05)", result.message)
                self.assertIn("Supreme Court of India", result.message)
                self.assertIn("check the court's own records", result.message)
                self.assertEqual(result.disclaimer, DISCLAIMER)
                for text in [result.message, result.disclaimer, *result.warnings]:
                    self.assertIsNone(FORBIDDEN.search(text), text)

    def test_found_results_and_warnings_avoid_the_forbidden_wording_too(self):
        for criteria in (dict(cnr="HCBM04_9_2015-06-02"), dict(judge="sharma"), dict(case_ref="XYZ 7 of 2020"), dict(year=2020)):
            result = find(self.index, **criteria)
            for text in [result.message, result.disclaimer, *result.warnings]:
                self.assertIsNone(FORBIDDEN.search(text), text)

    def test_no_user_facing_string_in_the_lookup_code_uses_the_forbidden_wording(self):
        """Reads the source: every string constant that is not a docstring, in the modules that build responses."""
        root = Path(__file__).resolve().parents[1] / "src" / "legalrag"
        checked = 0
        for relative in ("lookup/index.py", "api/lookup.py"):
            tree = ast.parse((root / relative).read_text(encoding="utf-8"))
            docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                          if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body
                          and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    checked += 1
                    self.assertIsNone(FORBIDDEN.search(node.value), f"{relative}: {node.value!r}")
        self.assertGreater(checked, 50)  # the scan really looked at the code


# --------------------------------------------------------------------------- the script


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


corpus_script = load_script("build_corpus_v2")
lookup_script = load_script("build_lookup_index")


def hc_record(i, text, **overrides):
    base = {"cnr": f"HCBM04_{i}_2015-06-0{(i % 8) + 1}", "court_code": "27_1", "bench": "testcase", "year": 2015, "full_text": text,
            "title": "WP/10/2015", "judge": "A B Sharma", "petitioner": "Ram", "respondent": "State",
            "decision_date": f"2015-06-0{(i % 8) + 1}", "disposal_nature_normalized": "allowed", "case_type": ""}
    base.update(overrides)
    return base


class TestBuildScript(unittest.TestCase):
    def build_corpus(self, tmp: Path, max_rows=None):
        long_text = " ".join(f"word{j}" for j in range(80))
        rows = [hc_record(i, f"Judgment {i} {long_text} {i}") for i in range(6)]
        rows.append(hc_record(10, "Rule discharged.", title="CRA/5/2014", judge=""))
        rows.append(hc_record(11, rows[0]["full_text"]))  # a near duplicate of the first
        source = tmp / "hc.parquet"
        pd.DataFrame(rows).to_parquet(source, index=False)
        out = tmp / "v2" / "corpus"
        corpus_script.build({"high_courts": [source], "supreme_court": []}, out, BuildConfig(min_chars=200, today=dt.date(2026, 11, 2)), max_rows=max_rows)
        return out

    def run_build(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return lookup_script.build(*args, **kwargs)

    def test_every_kept_record_is_looked_up_including_short_orders_and_merged_cnrs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self.build_corpus(tmp)
            out = tmp / "v2" / "lookup.sqlite"
            report = self.run_build(corpus, out)
            self.assertEqual((report["records"], report["links"], report["checked_against_build_report"]), (7, 1, True))
            index = CaseLookupIndex(out)
            short = find(index, case_ref="CRA/5/2014")
            self.assertEqual((short.status, short.results[0]["in_search_index"], short.results[0]["judge_display"]), ("found", False, "not recorded"))
            self.assertEqual(find(index, cnr="HCBM04_5_2015-06-06").results[0]["in_search_index"], True)
            merged = find(index, cnr="HCBM04_11_2015-06-04")
            self.assertEqual(cnrs(merged), ["HCBM04_0_2015-06-01"])
            self.assertEqual(find(index, bench="nagpur", year=2015).total, 7)
            index.close()

    def test_no_judgment_text_is_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "v2" / "lookup.sqlite"
            self.run_build(self.build_corpus(tmp), out)
            connection = sqlite3.connect(out)
            columns = {r[1] for r in connection.execute("PRAGMA table_info(cases)")}
            connection.close()
            self.assertNotIn("text", columns)
            self.assertNotIn(b"word40", out.read_bytes())

    def test_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self.build_corpus(tmp)
            out = tmp / "v2" / "lookup.sqlite"
            with self.assertRaisesRegex(SystemExit, "'v2' directory part"):
                self.run_build(corpus, tmp / "lookup.sqlite")
            self.run_build(corpus, out)
            with self.assertRaisesRegex(SystemExit, "already exists"):
                self.run_build(corpus, out)
            self.run_build(corpus, out, overwrite=True)
            with self.assertRaisesRegex(SystemExit, "build_corpus_v2"):
                self.run_build(tmp / "empty", tmp / "v2" / "other.sqlite")

    def test_partial_builds_are_refused_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self.build_corpus(tmp, max_rows=3)
            with self.assertRaisesRegex(SystemExit, "partial"):
                self.run_build(corpus, tmp / "v2" / "lookup.sqlite")
            self.assertEqual(self.run_build(corpus, tmp / "v2" / "lookup.sqlite", allow_partial=True)["records"], 3)

    def test_a_row_count_that_disagrees_with_the_build_report_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = self.build_corpus(tmp)
            report_path = corpus / "build_report.json"
            report = json.loads(report_path.read_text())
            report["outputs"]["search_corpus_rows"] += 5
            report_path.write_text(json.dumps(report))
            out = tmp / "v2" / "lookup.sqlite"
            with self.assertRaisesRegex(SystemExit, "build_report.json says"):
                self.run_build(corpus, out)
            self.assertEqual([p.name for p in out.parent.iterdir() if p.name.startswith("lookup")], [])


# --------------------------------------------------------------------------- the endpoint


class TestEndpoint(unittest.TestCase):
    def setUp(self):
        lookup_api.reset_lookup_cache()
        self.addCleanup(lookup_api.reset_lookup_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index = build(self.tmp.name)
        self.addCleanup(self.index.close)
        self.app = FastAPI()
        self.app.include_router(lookup_api.router)
        self.app.dependency_overrides[lookup_api.get_lookup_index] = lambda: self.index
        self.client = TestClient(self.app)

    def get(self, **params):
        return self.client.get("/lookup", params=params)

    def test_found(self):
        response = self.get(cnr="HCBM04_1_2015-06-01")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["status"], body["total"], len(body["results"])), ("found", 1, 1))
        record = body["results"][0]
        self.assertEqual((record["cnr"], record["case_reference"], record["judge_display"]), ("HCBM04_1_2015-06-01", "WP/1519/2015", "A B Sharma"))
        self.assertEqual(body["criteria"], {"cnr": "HCBM04_1_2015-06-01"})
        self.assertEqual(body["disclaimer"], DISCLAIMER)
        self.assertTrue(body["coverage"])
        self.assertNotIn("text", record)

    def test_not_in_our_data_is_a_normal_200_answer(self):
        response = self.get(case_ref="WP/9999/2030")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["status"], body["results"], body["total"]), ("not_in_our_data", [], 0))
        self.assertIsNone(FORBIDDEN.search(response.text))
        self.assertIn("Not in our data", body["message"])

    def test_search_by_names_dates_and_pagination(self):
        self.assertEqual(self.get(title="ram kumar").json()["results"][0]["cnr"], "HCBM04_1_2015-06-01")
        body = self.get(court="bombay", limit=2, offset=1).json()
        self.assertEqual((body["total"], len(body["results"]), body["offset"], body["limit"]), (4, 2, 1, 2))
        self.assertEqual(self.get(date_from="2018-01-01", date_to="2018-12-31").json()["total"], 1)

    def test_invalid_requests_are_422_with_a_reason(self):
        cases = [
            ({}, "at least one"),
            ({"date": "2020-13-45"}, None),
            ({"date_from": "2021-01-01", "date_to": "2020-01-01"}, "later than"),
            ({"cnr": "x", "limit": 0}, None),
            ({"cnr": "x", "limit": 101}, None),
            ({"cnr": "x", "offset": -1}, None),
            ({"year": 20}, None),
            ({"title": "x" * 201}, None),
            ({"date": "June 1"}, None),
            ({"case_ref": "hello"}, "case number"),
            ({"title": "v"}, "two or more letters"),
        ]
        for params, fragment in cases:
            with self.subTest(params=params):
                response = self.get(**params)
                self.assertEqual(response.status_code, 422, response.text)
                if fragment:
                    self.assertIn(fragment, json.dumps(response.json()))

    def test_unknown_parameters_are_ignored(self):
        self.assertEqual(self.get(cnr="HCBM04_1_2015-06-01", sql="DROP TABLE cases").status_code, 200)
        self.assertEqual(len(self.index), 5)

    def test_unexpected_failures_are_500_and_sanitized(self):
        class Broken:
            def search(self, query):
                raise RuntimeError("secret path /srv/lookup.sqlite and password hunter2")

        self.app.dependency_overrides[lookup_api.get_lookup_index] = lambda: Broken()
        response = self.get(cnr="x")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": lookup_api.LOOKUP_FAILED_DETAIL})
        self.assertNotIn("hunter2", response.text)

    def test_the_route_is_documented(self):
        spec = self.client.get("/openapi.json").json()
        self.assertIn("/lookup", spec["paths"])
        names = {p["name"] for p in spec["paths"]["/lookup"]["get"]["parameters"]}
        self.assertTrue({"cnr", "case_ref", "title", "party", "judge", "date", "date_from", "date_to", "year", "court", "bench", "limit", "offset"} <= names)


class TestEnablingTheEndpoint(unittest.TestCase):
    def setUp(self):
        lookup_api.reset_lookup_cache()
        self.addCleanup(lookup_api.reset_lookup_cache)
        self.app = FastAPI()
        self.app.include_router(lookup_api.router)
        self.client = TestClient(self.app)

    def use(self, **flags):
        settings = Settings(_env_file=None, **flags)
        self.app.dependency_overrides[get_settings] = lambda: settings

    def test_off_by_default(self):
        self.assertFalse(Settings(_env_file=None).lookup_enabled)
        self.use(environment="production")
        response = self.client.get("/lookup", params={"cnr": "x"})
        self.assertEqual((response.status_code, response.json()), (503, {"detail": lookup_api.LOOKUP_UNAVAILABLE_DETAIL}))

    def test_enabled_but_the_file_is_missing_is_503_without_leaking_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.use(environment="production", lookup_enabled=True, artifact_dir=Path(tmp) / "secret-location")
            response = self.client.get("/lookup", params={"cnr": "x"})
        self.assertEqual((response.status_code, response.json()), (503, {"detail": lookup_api.LOOKUP_UNAVAILABLE_DETAIL}))
        self.assertNotIn("secret-location", response.text)

    def test_enabled_in_production_reads_the_configured_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = CaseLookupIndex.build(Path(tmp) / "lookup.sqlite", ROWS, LINKS)
            index.close()
            self.use(environment="production", lookup_enabled=True, artifact_dir=Path(tmp))
            response = self.client.get("/lookup", params={"judge": "reddy"})
            self.assertEqual(response.json()["results"][0]["cnr"], "SC_5")
            lookup_api.reset_lookup_cache()

    def test_local_stub_mode_serves_a_tiny_built_in_index(self):
        self.use(environment="local_stub", lookup_enabled=True)
        body = self.client.get("/lookup", params={"case_ref": "WP/100/2020"}).json()
        self.assertEqual((body["status"], body["results"][0]["cnr"]), ("found", "STUB010000012020_1"))
        self.assertEqual(self.client.get("/lookup", params={"cnr": "nope"}).json()["status"], "not_in_our_data")

    def test_the_real_app_has_the_route_and_lists_it_at_the_root(self):
        from legalrag.api.main import create_app

        client = TestClient(create_app())
        self.assertEqual(client.get("/").json()["lookup_url"], "/lookup")
        self.assertIn("/lookup", client.get("/openapi.json").json()["paths"])


class TestCaseQueryParser(unittest.TestCase):
    def test_forms(self):
        cases = {
            "WP/1519/2000": ("WP", "1519", "2000"),
            "Writ Petition No. 1519 of 2000": ("WP", "1519", "2000"),
            "first appeal 88 of 2018": ("FA", "88", "2018"),
            "WP(L) 12 of 2010": ("WP(L)", "12", "2010"),
            "1519 of 2000": (None, "1519", "2000"),
            "No. 7/2020": (None, "7", "2020"),
            "CR. MISC./57199/2022": ("CRMISC", "57199", "2022"),
            "Mystery Petition 5 of 2001": (None, "5", "2001"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_case_query(text), expected)

    def test_unreadable(self):
        for text in ("", "hello", "WP/xx/2000", "12"):
            with self.subTest(text=text):
                with self.assertRaises(LookupError_):
                    parse_case_query(text)


if __name__ == "__main__":
    unittest.main()

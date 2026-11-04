"""
Tests for the statute mapping engine, its builder and its wiring (plan task C6).

The fixture rows below use INVENTED section numbers (700s-900s). They exist so the engine can be checked
without any real data; they are not a transcription of the official table, and nothing in this repository
contains one until the owner builds and reviews it with scripts/build_statute_map.py.
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from test_artifact_contracts import CORPUS, FakeEmbedder, build_chunks_dataframe  # noqa: F401  (shared fixtures)

from legalrag.api.config import Settings
from legalrag.api.dependencies import (
    PipelineComponents,
    StubCrossEncoderReranker,
    StubGenerator,
    create_stub_chunks,
    load_retrieval_components,
)
from legalrag.retrieval.bm25 import BM25Retriever
from legalrag.retrieval.statute_map import (
    ACTS,
    PAIRS,
    StatuteMap,
    StatuteMapError,
    base_section,
    find_references,
    normalize_section,
    validate_row,
)

FIXTURE_ROWS = [
    # (pair, old, new, relationship)
    ("IPC-BNS", "901", "701(1)", "changed"),
    ("IPC-BNS", "902", "702", "same"),
    ("IPC-BNS", "903", "703(1)", "split"),
    ("IPC-BNS", "903", "703(2)", "split"),
    ("IPC-BNS", "904", None, "omitted"),
    ("IPC-BNS", None, "705", "new"),
    ("IPC-BNS", "906A", "706", "merged"),
    ("IPC-BNS", "907", "707", "merged"),
    ("IPC-BNS", "908", "707", "merged"),
    ("CRPC-BNSS", "911", "811", "changed"),
    ("CRPC-BNSS", "912(2)", "812(1)", "changed"),
    ("CRPC-BNSS", "913", "813", "same"),
    ("IEA-BSA", "921A", "821", "changed"),
    ("IEA-BSA", "922", "822", "same"),
]


def fixture_map(reviewed_by="Test Reviewer"):
    rows = [{"pair": p, "old": o, "new": n, "relationship": r} for p, o, n, r in FIXTURE_ROWS]
    return StatuteMap.from_dict({"format": 1, "reviewed_by": reviewed_by, "rows": rows})


def sections(expansion):
    return [m.ref.section for m in expansion.matches]


class TestNormalising(unittest.TestCase):
    def test_section_forms(self):
        cases = {"498a": "498A", "498A": "498A", "438( 2 )": "438(2)", "85(A)": "85(a)", "103(1)(a)": "103(1)(a)", "0302": "302", " 65 b ": "65B"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_section(raw), expected)
        for bad in ("", "abc", "12-3", "§12"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    normalize_section(bad)

    def test_base_section(self):
        self.assertEqual(base_section("103(1)(a)"), "103")
        self.assertEqual(base_section("498A"), "498A")


class TestFindingReferences(unittest.TestCase):
    def found(self, text):
        return [(r.act, r.section) for r in find_references(text)]

    def test_each_law_in_each_common_spelling(self):
        cases = {
            "Section 901 IPC": [("IPC", "901")],
            "section 901 of the I.P.C.": [("IPC", "901")],
            "s. 901 of the Indian Penal Code, 1860": [("IPC", "901")],
            "u/s 901 ipc": [("IPC", "901")],
            "Sec. 701(1) BNS": [("BNS", "701(1)")],
            "section 701 of the Bharatiya Nyaya Sanhita, 2023": [("BNS", "701")],
            "Section 911 CrPC": [("CRPC", "911")],
            "Section 911 Cr.P.C.": [("CRPC", "911")],
            "section 911 of the Code of Criminal Procedure, 1973": [("CRPC", "911")],
            "Section 811 BNSS": [("BNSS", "811")],
            "s.811 of the Bharatiya Nagarik Suraksha Sanhita": [("BNSS", "811")],
            "Section 921A of the Evidence Act": [("IEA", "921A")],
            "Section 921A of the Indian Evidence Act, 1872": [("IEA", "921A")],
            "section 922 I.E.A.": [("IEA", "922")],
            "Section 821 BSA": [("BSA", "821")],
            "section 821 of the Bharatiya Sakshya Adhiniyam": [("BSA", "821")],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.found(text), expected)

    def test_bnss_is_not_read_as_bns(self):
        self.assertEqual(self.found("Section 811 BNSS and Section 701 BNS"), [("BNSS", "811"), ("BNS", "701")])

    def test_act_before_the_number(self):
        for text, expected in {"IPC 901": [("IPC", "901")], "BNS section 701(1)": [("BNS", "701(1)")], "Cr.P.C. s. 912(2)": [("CRPC", "912(2)")],
                               "BNSS, 2023 s. 813": [("BNSS", "813")], "IPC, 1860 section 902": [("IPC", "902")]}.items():
            with self.subTest(text=text):
                self.assertEqual(self.found(text), expected)

    def test_lists_of_sections_share_the_law(self):
        self.assertEqual(self.found("Sections 901 and 902 IPC"), [("IPC", "901"), ("IPC", "902")])
        self.assertEqual(self.found("ss. 901, 902 & 903 of the IPC"), [("IPC", "901"), ("IPC", "902"), ("IPC", "903")])
        self.assertEqual(self.found("Sections 901/902 IPC"), [("IPC", "901"), ("IPC", "902")])
        self.assertEqual(self.found("Section 901 or 902 IPC"), [("IPC", "901"), ("IPC", "902")])

    def test_two_laws_in_one_question(self):
        self.assertEqual(self.found("Section 901 IPC read with Section 911 CrPC"), [("IPC", "901"), ("CRPC", "911")])
        self.assertEqual(self.found("Section 901 IPC and Section 902 IPC"), [("IPC", "901"), ("IPC", "902")])

    def test_a_section_with_no_named_law_is_never_guessed(self):
        for text in ("Section 901", "what does section 911 say about bail", "Section 901 murder", "ss. 901 and 902", "u/s 901"):
            with self.subTest(text=text):
                self.assertEqual(self.found(text), [])

    def test_other_laws_and_noise_are_ignored(self):
        for text in ("Section 138 of the Negotiable Instruments Act", "Section 901 NI Act", "Rs. 901 paid under the IPC", "the IPC, 1860",
                     "IPC 1860", "BNS 2023", "passed in 2023 under the BNS", "Article 226 of the Constitution", ""):
            with self.subTest(text=text):
                self.assertEqual(self.found(text), [])

    def test_ranges_are_not_expanded(self):
        self.assertEqual(self.found("Sections 901 to 905 IPC"), [])

    def test_positions_point_at_the_matched_text(self):
        text = "Bail under Section 911 CrPC please"
        (ref,) = find_references(text)
        self.assertEqual(text[ref.start:ref.end], ref.text)
        self.assertTrue(ref.text.startswith("Section 911"))

    def test_sub_sections_and_letters_are_kept(self):
        self.assertEqual(self.found("Section 912(2) CrPC"), [("CRPC", "912(2)")])
        self.assertEqual(self.found("Section 906a IPC"), [("IPC", "906A")])
        self.assertEqual(self.found("Section 701(1)(a) BNS"), [("BNS", "701(1)(a)")])


class TestRowValidation(unittest.TestCase):
    def row(self, **overrides):
        base = {"pair": "IPC-BNS", "old": "901", "new": "701", "relationship": "same"}
        base.update(overrides)
        return base

    def test_good_rows_are_normalised(self):
        row = validate_row(self.row(old="906a", new="706 ", relationship=None))
        self.assertEqual((row.old, row.new, row.relationship), ("906A", "706", "same"))

    def test_missing_side_defaults_to_new_or_omitted(self):
        self.assertEqual(validate_row(self.row(old=None, relationship=None)).relationship, "new")
        self.assertEqual(validate_row(self.row(new="", relationship=None)).relationship, "omitted")

    def test_bad_rows_are_refused_with_a_reason(self):
        cases = [
            (dict(pair="IPC-BSA"), "pair"),
            (dict(old=None, new=None), "needs an old or a new"),
            (dict(old="abc"), "not a section"),
            (dict(relationship="moved"), "relationship"),
            (dict(new=None, relationship="same"), "omitted"),
            (dict(old=None, relationship="changed"), "new"),
            (dict(relationship="omitted"), "does not fit"),
        ]
        for overrides, pattern in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(StatuteMapError, pattern):
                    validate_row(self.row(**overrides))


class TestBothFormsFindEachOther(unittest.TestCase):
    """The task's test: old and new section numbers each find both forms."""

    def setUp(self):
        self.map = fixture_map()

    def query_for(self, act, section):
        return f"Section {section} {ACTS[act][0]} and what the court held"

    def test_every_row_with_both_sides_works_in_both_directions(self):
        checked = 0
        for pair, old, new, _ in FIXTURE_ROWS:
            if old is None or new is None:
                continue
            old_act, new_act = PAIRS[pair]
            forward = self.map.expand(self.query_for(old_act, old))
            self.assertIn(f"Section {new} {ACTS[new_act][0]}", forward.additions, (pair, old, new))
            backward = self.map.expand(self.query_for(new_act, new))
            self.assertIn(f"Section {old} {ACTS[old_act][0]}", backward.additions, (pair, old, new))
            checked += 1
        self.assertEqual(checked, 12)  # 14 fixture rows, minus the omitted one and the new one

    def test_the_retrieval_query_contains_both_forms(self):
        expansion = self.map.expand("Is anticipatory relief available under Section 911 CrPC?")
        self.assertEqual(expansion.retrieval_query, "Is anticipatory relief available under Section 911 CrPC? Section 811 BNSS")
        self.assertTrue(expansion.expanded)
        reverse = self.map.expand("Section 811 BNSS")
        self.assertEqual(reverse.retrieval_query, "Section 811 BNSS Section 911 CrPC")

    def test_a_search_for_either_form_finds_a_chunk_written_in_the_other(self):
        chunks = {
            "old_law": "The accused was charged under Section 901 of the IPC and convicted after trial.",
            "new_law": "The accused was charged under Section 701(1) of the BNS and convicted after trial.",
            "other": "A writ petition under Article 226 was dismissed for delay.",
        }
        index = BM25Retriever(chunk_ids=list(chunks), corpus_texts=list(chunks.values()))
        old_question, new_question = "What is the punishment under Section 901 IPC?", "What is the punishment under Section 701(1) BNS?"

        # Without expansion each question ranks its own form first.
        self.assertEqual(index.search(old_question, 1)[0][0], "old_law")
        self.assertEqual(index.search(new_question, 1)[0][0], "new_law")
        # With expansion, each question reaches both chunks, and the other-law chunk now scores above the unrelated one.
        for question in (old_question, new_question):
            expanded = self.map.expand(question).retrieval_query
            scores = dict(index.search(expanded, 3))
            self.assertGreater(scores["old_law"], scores["other"], question)
            self.assertGreater(scores["new_law"], scores["other"], question)
            # and the chunk in the other law's wording scores higher than it did for the plain question
            other_form = "new_law" if question is old_question else "old_law"
            self.assertGreater(scores[other_form], dict(index.search(question, 3)).get(other_form, 0.0), question)


class TestLookupBehaviour(unittest.TestCase):
    def setUp(self):
        self.map = fixture_map()

    def test_one_to_many_gives_every_counterpart(self):
        status, counterparts = self.map.counterparts("IPC", "903")
        self.assertEqual(status, "mapped")
        self.assertEqual([c.section for c in counterparts], ["703(1)", "703(2)"])
        self.assertEqual({c.relationship for c in counterparts}, {"split"})

    def test_many_to_one_works_from_the_new_side(self):
        status, counterparts = self.map.counterparts("BNS", "707")
        self.assertEqual((status, [c.section for c in counterparts]), ("mapped", ["907", "908"]))

    def test_a_search_for_the_whole_section_finds_its_sub_sections_and_a_sub_section_prefers_its_own_row(self):
        self.assertEqual([c.section for c in self.map.counterparts("BNS", "703")[1]], ["903"])
        self.assertEqual([c.section for c in self.map.counterparts("IPC", "903")[1]], ["703(1)", "703(2)"])
        self.assertEqual([c.section for c in self.map.counterparts("CRPC", "912(2)")[1]], ["812(1)"])
        self.assertEqual([c.section for c in self.map.counterparts("CRPC", "912")[1]], ["812(1)"])  # whole section -> its sub-section row
        self.assertEqual([c.section for c in self.map.counterparts("BNSS", "812")[1]], ["912(2)"])
        # a sub-section the table does not list falls back to the whole section
        self.assertEqual([c.section for c in self.map.counterparts("CRPC", "912(7)")[1]], ["812(1)"])

    def test_omitted_and_new_provisions_are_reported_not_expanded(self):
        self.assertEqual(self.map.counterparts("IPC", "904"), ("no_counterpart", []))
        self.assertEqual(self.map.counterparts("BNS", "705"), ("no_counterpart", []))
        expansion = self.map.expand("Section 904 IPC")
        self.assertEqual((expansion.additions, expansion.matches[0].status), ([], "no_counterpart"))
        self.assertEqual(expansion.retrieval_query, "Section 904 IPC")

    def test_a_section_missing_from_the_table_is_reported(self):
        self.assertEqual(self.map.counterparts("IPC", "999"), ("not_in_table", []))
        self.assertEqual(self.map.expand("Section 999 IPC").matches[0].status, "not_in_table")

    def test_the_same_number_in_different_laws_does_not_cross_over(self):
        # 911 is a CrPC section here; an IPC 911 must not pick up the CrPC row.
        self.assertEqual(self.map.counterparts("IPC", "911"), ("not_in_table", []))
        self.assertEqual(self.map.counterparts("BNS", "811"), ("not_in_table", []))

    def test_letter_sections_and_the_evidence_act(self):
        expansion = self.map.expand("Admissibility under section 921a of the Evidence Act")
        self.assertEqual(expansion.additions, ["Section 821 BSA"])
        self.assertEqual(self.map.expand("Section 821 BSA").additions, ["Section 921A Evidence Act"])

    def test_additions_are_unique_and_capped(self):
        text = "Sections 907 and 908 IPC"
        self.assertEqual(self.map.expand(text).additions, ["Section 707 BNS"])  # both map to 707: added once
        wide = self.map.expand("Sections 901, 902, 903 and 906A IPC", max_additions=2)
        self.assertEqual(len(wide.additions), 2)

    def test_act_names_can_be_appended(self):
        expansion = self.map.expand("Section 911 CrPC", include_act_names=True)
        self.assertEqual(expansion.additions, ["Section 811 BNSS", "Bharatiya Nagarik Suraksha Sanhita, 2023"])

    def test_text_without_references_is_returned_untouched(self):
        for text in ("anticipatory bail principles", "", "Section 911"):
            expansion = self.map.expand(text)
            self.assertEqual((expansion.retrieval_query, expansion.expanded), (text, False))

    def test_summary_counts_rows(self):
        summary = self.map.summary()
        self.assertEqual(summary["rows"], len(FIXTURE_ROWS))
        self.assertEqual(summary["rows_by_pair"], {"IPC-BNS": 9, "CRPC-BNSS": 3, "IEA-BSA": 2})


class TestLoading(unittest.TestCase):
    def write(self, data):
        path = Path(self.tmp.name) / "map.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rows = [{"pair": "IPC-BNS", "old": "901", "new": "701"}]

    def test_a_reviewed_file_loads(self):
        loaded = StatuteMap.load(self.write({"format": 1, "reviewed_by": "A. Reviewer", "rows": self.rows}))
        self.assertEqual(loaded.reviewed_by, "A. Reviewer")

    def test_an_unreviewed_file_is_refused_unless_testing(self):
        path = self.write({"format": 1, "reviewed_by": None, "rows": self.rows})
        with self.assertRaisesRegex(StatuteMapError, "not been reviewed"):
            StatuteMap.load(path)
        for blank in ("", "   "):
            with self.assertRaises(StatuteMapError):
                StatuteMap.load(self.write({"format": 1, "reviewed_by": blank, "rows": self.rows}))
        self.assertEqual(len(StatuteMap.load(path, require_reviewed=False).rows), 1)

    def test_bad_files_are_refused_with_plain_messages(self):
        with self.assertRaisesRegex(StatuteMapError, "not found.*build_statute_map"):
            StatuteMap.load(Path(self.tmp.name) / "absent.json")
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(StatuteMapError, "not valid JSON"):
            StatuteMap.load(bad)
        for data, pattern in (
            ({"format": 2, "reviewed_by": "x", "rows": self.rows}, "format"),
            ({"format": 1, "reviewed_by": "x", "rows": []}, "no rows"),
            ({"format": 1, "reviewed_by": "x"}, "no rows"),
            ({"format": 1, "reviewed_by": "x", "rows": self.rows + self.rows}, "duplicate"),
            ({"format": 1, "reviewed_by": "x", "rows": [{"pair": "IPC-BNS", "old": "x1", "new": "2"}]}, "row 0"),
        ):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(StatuteMapError, pattern):
                    StatuteMap.load(self.write(data))


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_statute_map.py"
    spec = importlib.util.spec_from_file_location("build_statute_map_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script()


class TestBuilder(unittest.TestCase):
    HEADER = "pair,old_section,new_section,relationship,note\n"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "v2" / "statute_map.json"

    def csv(self, text, name="a.csv"):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def run_build(self, paths, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            report = builder.build(paths, self.out, **kwargs)
        return report, out.getvalue()

    def test_csv_becomes_a_json_file_the_engine_loads(self):
        path = self.csv(self.HEADER + "IPC-BNS,901,701(1),changed,note one\nIPC-BNS,904,,omitted,\nIPC-BNS,,705,new,\nCRPC-BNSS,911,811,,\n")
        report, _ = self.run_build([path], reviewed_by="A. Reviewer", reviewed_on="2026-11-10", source_urls=["https://example.invalid/table"], retrieved_on="2026-11-04")
        self.assertEqual(report["rows"], 4)
        document = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual((document["reviewed_by"], document["reviewed_on"], document["retrieved_on"]), ("A. Reviewer", "2026-11-10", "2026-11-04"))
        self.assertEqual(document["source_urls"], ["https://example.invalid/table"])
        self.assertEqual(len(document["csv_files"][0]["sha256"]), 64)
        loaded = StatuteMap.load(self.out)
        self.assertEqual(loaded.expand("Section 901 IPC").additions, ["Section 701(1) BNS"])
        self.assertEqual(loaded.counterparts("CRPC", "911")[1][0].relationship, "same")  # derived default

    def test_a_file_with_no_reviewer_is_written_but_the_search_refuses_it(self):
        path = self.csv(self.HEADER + "IPC-BNS,901,701,,\n")
        _, printed = self.run_build([path])
        self.assertIn("NOT REVIEWED", printed)
        self.assertIsNone(json.loads(self.out.read_text())["reviewed_by"])
        with self.assertRaisesRegex(StatuteMapError, "not been reviewed"):
            StatuteMap.load(self.out)

    def test_every_problem_is_listed_with_its_line_number(self):
        path = self.csv(self.HEADER + "IPC-BNS,901,701,,\nIPC-BNS,x9,702,,\nIPC-XYZ,903,703,,\nIPC-BNS,904,,same,\nIPC-BNS,901,701,,\n")
        with self.assertRaises(SystemExit) as ctx:
            self.run_build([path])
        message = str(ctx.exception)
        for fragment in ("a.csv line 3", "a.csv line 4", "a.csv line 5", "duplicate row"):
            self.assertIn(fragment, message)
        self.assertFalse(self.out.exists())

    def test_pair_can_come_from_the_command_line(self):
        path = self.csv("old_section,new_section\n901,701\n902,702\n")
        report, _ = self.run_build([path], default_pair="IPC-BNS")
        self.assertEqual(report["rows_by_pair"], {"IPC-BNS": 2})
        with self.assertRaisesRegex(SystemExit, "no 'pair' column"):
            self.run_build([path], overwrite=True)

    def test_header_bom_blank_lines_and_extra_columns_are_tolerated(self):
        path = self.dir / "bom.csv"
        path.write_bytes(("﻿" + "pair,old_section,new_section,source_page\n" + "IPC-BNS,901,701,12\n\n,,,\n").encode("utf-8"))
        report, _ = self.run_build([path])
        self.assertEqual(report["rows"], 1)

    def test_wrong_headers_and_empty_files_are_refused(self):
        with self.assertRaisesRegex(SystemExit, "header must include"):
            self.run_build([self.csv("pair,old\nIPC-BNS,901\n")])
        with self.assertRaisesRegex(SystemExit, "no rows"):
            self.run_build([self.csv(self.HEADER)], overwrite=True)

    def test_several_files_are_merged_and_sorted(self):
        a = self.csv(self.HEADER + "IPC-BNS,902,702,,\n", "a.csv")
        b = self.csv(self.HEADER + "IPC-BNS,901,701,,\nIEA-BSA,921A,821,,\n", "b.csv")
        report, _ = self.run_build([a, b])
        self.assertEqual(report["rows_by_pair"], {"IPC-BNS": 2, "IEA-BSA": 1})
        rows = json.loads(self.out.read_text())["rows"]
        self.assertEqual([(r["pair"], r["old"]) for r in rows], [("IEA-BSA", "921A"), ("IPC-BNS", "901"), ("IPC-BNS", "902")])

    def test_output_guards(self):
        path = self.csv(self.HEADER + "IPC-BNS,901,701,,\n")
        with self.assertRaisesRegex(SystemExit, "'v2' directory part"):
            builder.build([path], self.dir / "statute_map.json")
        self.run_build([path])
        with self.assertRaisesRegex(SystemExit, "already exists"):
            self.run_build([path])
        self.run_build([path], overwrite=True)
        self.assertEqual([p.name for p in self.out.parent.iterdir()], ["statute_map.json"])  # no temporary file left


# --------------------------------------------------------------------------- wiring


def pipeline_with(mapper):
    class RecordingBm25:
        def __init__(self):
            self.queries = []
            self.chunk_ids = create_stub_chunks()["chunk_id"].tolist()

        def search(self, query, top_k=50):
            self.queries.append(query)
            return [(cid, 1.0 / (i + 1)) for i, cid in enumerate(self.chunk_ids[:top_k])]

    class RecordingDense(RecordingBm25):
        pass

    chunks = create_stub_chunks()
    return PipelineComponents(RecordingBm25(), RecordingDense(), StubCrossEncoderReranker(), chunks, StubGenerator(),
                              "local_stub", statute_mapper=mapper)


class TestWiring(unittest.TestCase):
    def test_flag_defaults_to_off_and_the_path_follows_the_artifact_directory(self):
        s = Settings(_env_file=None, artifact_dir=Path("artifacts/v2"))
        self.assertEqual((s.statute_expansion, s.statute_map_allow_unreviewed), (False, False))
        self.assertEqual(s.statute_map_path, Path("artifacts/v2/statute_map.json"))

    def test_without_a_mapper_the_query_is_unchanged(self):
        pipeline = pipeline_with(None)
        self.assertEqual(pipeline.retrieval_query("Section 911 CrPC"), "Section 911 CrPC")

    def test_the_api_expands_the_bm25_query_only(self):
        from legalrag.api.routes import query_endpoint
        from legalrag.api.schemas import QueryRequest

        pipeline = pipeline_with(fixture_map())
        question = "Is bail available under Section 911 CrPC?"
        response = query_endpoint(QueryRequest.model_validate({"question": question}), pipeline)
        self.assertEqual(response.status, "ok")
        self.assertEqual(pipeline.bm25.queries, [question + " Section 811 BNSS"])
        self.assertEqual(pipeline.dense.queries, [question])  # the dense query is the question as asked

    def test_the_api_without_a_mapper_searches_the_question_as_asked(self):
        from legalrag.api.routes import query_endpoint
        from legalrag.api.schemas import QueryRequest

        pipeline = pipeline_with(None)
        query_endpoint(QueryRequest.model_validate({"question": "Section 911 CrPC bail"}), pipeline)
        self.assertEqual(pipeline.bm25.queries, ["Section 911 CrPC bail"])

    def test_the_retrieval_evaluation_expands_bm25_the_same_way(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
        spec = importlib.util.spec_from_file_location("run_retrieval_eval_statute", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        pipeline = pipeline_with(fixture_map())
        lookup = script.ChunkLookup(pipeline)
        script.retrieve_all_systems(pipeline, lookup, "Section 901 IPC punishment", depth=5)
        self.assertEqual(pipeline.bm25.queries, ["Section 901 IPC punishment Section 701(1) BNS"])
        self.assertEqual(pipeline.dense.queries, ["Section 901 IPC punishment"])


class TestLoaderFlag(unittest.TestCase):
    """load_retrieval_components with statute_expansion on (v1 artifacts, so no other flag is needed)."""

    def setUp(self):
        import numpy as np

        from legalrag.retrieval.dense import DenseRetriever

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "v2"
        self.dir.mkdir()
        ids = [cid for cid, _ in CORPUS]
        BM25Retriever(chunk_ids=ids, corpus_texts=[t for _, t in CORPUS]).save(self.dir / "bm25.pkl")
        build_chunks_dataframe().to_parquet(self.dir / "legal_chunks.parquet", index=False)
        dense = DenseRetriever()
        dense.build_index(np.ones((len(ids), 4), dtype=np.float32), ids)
        dense.save_index(self.dir / "dense.index")

    def settings(self, **flags):
        return Settings(_env_file=None, environment="production", artifact_dir=self.dir, model_warmup_enabled=False, **flags)

    def test_flag_off_loads_no_mapper_even_if_a_file_exists(self):
        (self.dir / "statute_map.json").write_text("{broken", encoding="utf-8")
        self.assertIsNone(load_retrieval_components(self.settings()).statute_mapper)

    def test_flag_on_loads_the_reviewed_map(self):
        (self.dir / "statute_map.json").write_text(json.dumps({"format": 1, "reviewed_by": "Reviewer", "rows": [{"pair": "IPC-BNS", "old": "901", "new": "701"}]}), encoding="utf-8")
        components = load_retrieval_components(self.settings(statute_expansion=True))
        self.assertEqual(components.retrieval_query("Section 901 IPC"), "Section 901 IPC Section 701 BNS")

    def test_flag_on_with_a_missing_file_stops_startup_and_says_how_to_build_it(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_retrieval_components(self.settings(statute_expansion=True))
        self.assertIn("scripts/build_statute_map.py", str(ctx.exception))

    def test_flag_on_with_an_unreviewed_map_fails_closed_unless_overridden_for_testing(self):
        (self.dir / "statute_map.json").write_text(json.dumps({"format": 1, "reviewed_by": None, "rows": [{"pair": "IPC-BNS", "old": "901", "new": "701"}]}), encoding="utf-8")
        with self.assertRaisesRegex(StatuteMapError, "not been reviewed"):
            load_retrieval_components(self.settings(statute_expansion=True))
        components = load_retrieval_components(self.settings(statute_expansion=True, statute_map_allow_unreviewed=True))
        self.assertIsNotNone(components.statute_mapper)

    def test_flag_on_with_a_corrupt_map_fails_closed(self):
        (self.dir / "statute_map.json").write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(StatuteMapError, "not valid JSON"):
            load_retrieval_components(self.settings(statute_expansion=True))


if __name__ == "__main__":
    unittest.main()

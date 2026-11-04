"""
Case lookup over all records (plan task C7).

The search index holds only reasoned judgments (5,000 characters and up). Case lookup is different: it
covers EVERY record the corpus build kept (search index and lookup-only), by CNR, case number, title and
party names, judge, date, court and bench. It stores metadata only, no judgment text, in a SQLite file
with an FTS5 index for names.

Wording rule, enforced by tests: a search that finds nothing says "Not in our data" and states what the
data covers. It never says or implies that a case does not exist, because the dataset has no Bombay data
after May 2026 and no Supreme Court data after 2024, and a missing record can be a gap in the data, a
different spelling or a wrong date. Nothing here says whether a judgment is still good law.

A CNR that the corpus build removed as a near-duplicate of another judgment resolves to the record that was
kept, and the result says so.
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from legalrag.preprocessing.corpus_v2 import BOMBAY_CASE_TYPE_NAMES, parse_case_reference

SCHEMA_VERSION = 1
MAX_LIMIT = 100
MAX_OFFSET = 10_000
MAX_TEXT_CHARS = 200
_VARIABLE_BATCH = 500

DISCLAIMER = (
    "Research aid, not legal advice. Check any judgment against the official copy. "
    "'Not in our data' means only that no matching record is in our data."
)
NOT_FOUND = "Not in our data"

COLUMNS = (
    "cnr", "title", "petitioner", "respondent", "judge", "judgment_date", "judgment_year", "court_name",
    "bench_name", "case_type_code", "case_type_name", "case_number", "case_year", "outcome",
    "in_search_index", "date_conflict", "date_source", "source_config",
)


class LookupError_(ValueError):
    """The lookup request is not valid (for example: no criteria, a bad date). Safe to show to the caller."""


def normalize_cnr(text: str) -> str:
    return re.sub(r"\s+", "", text or "").upper()


def _clean_number(value: Any) -> Optional[str]:
    """'01519' -> '1519'; anything that is not a whole number gives None."""
    text = str(value).strip() if value is not None else ""
    return str(int(text)) if text.isdigit() else None


_NAME_TO_CODE = {name.lower(): code for code, name in BOMBAY_CASE_TYPE_NAMES.items()}
_WORDY_REFERENCE = re.compile(
    r"^\s*(?P<type>[A-Za-z][A-Za-z.&()\s\-]*?)\s*(?:no\.?|number)?\s*(?P<number>\d{1,7})\s*(?:of|/)\s*(?P<year>(?:18|19|20)\d{2})\s*$",
    re.IGNORECASE,
)
_NUMBER_YEAR = re.compile(r"^\s*(?:no\.?\s*)?(?P<number>\d{1,7})\s*(?:of|/)\s*(?P<year>(?:18|19|20)\d{2})\s*$", re.IGNORECASE)


def parse_case_query(text: str) -> Tuple[Optional[str], str, str]:
    """Reads 'WP/1519/2000', 'Writ Petition No. 1519 of 2000', 'WP(L) 12 of 2010' or '1519 of 2000'.

    Returns ``(case type code or None, number, year)``. A type that is not a known abbreviation or name is
    ignored (the number and year still match), and ``LookupError_`` is raised if there is no number and year.
    """
    reference = parse_case_reference(text)
    if reference:
        return reference["code"], _clean_number(reference["number"]) or reference["number"], reference["year"]
    plain = _NUMBER_YEAR.match(text or "")
    if plain:
        return None, str(int(plain["number"])), plain["year"]
    wordy = _WORDY_REFERENCE.match(text or "")
    if wordy:
        words = re.sub(r"\s+", " ", wordy["type"]).strip(" .").lower()
        code = _NAME_TO_CODE.get(words)
        if code is None:
            compact = re.sub(r"[^A-Za-z()]", "", wordy["type"]).upper()
            code = compact if compact in BOMBAY_CASE_TYPE_NAMES or re.sub(r"\(.*\)$", "", compact) in BOMBAY_CASE_TYPE_NAMES else None
        return code, str(int(wordy["number"])), wordy["year"]
    raise LookupError_("Could not read a case number and year; write it like WP/1519/2000 or '1519 of 2000'.")


_STOP_TOKENS = {"v", "vs", "versus", "the", "and", "&"}


def _word_tokens(text: str) -> List[str]:
    """Search words: lower-cased word characters, without filler words and one-letter Latin words.

    A single character is kept when it is not ASCII. The index's tokenizer splits Indic words at vowel signs
    (so a Devanagari name is stored as letter pieces), and the query is split the same way to match.
    """
    tokens = []
    for token in re.findall(r"\w+", text.lower()):
        if token in _STOP_TOKENS:
            continue
        if len(token) >= 2 or token.isdigit() or not token.isascii():
            tokens.append(token)
    return tokens


def fts_expression(columns: str, text: str) -> Optional[str]:
    """An FTS5 expression for the words in ``text``, built only from plain word tokens (no query syntax)."""
    tokens = _word_tokens(text)
    if not tokens:
        return None
    return "{" + columns + "} : (" + " ".join(f'"{t}"*' for t in tokens) + ")"


def _check_date(value: Optional[str], name: str) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        raise LookupError_(f"{name} must be a date written YYYY-MM-DD.") from None


@dataclass
class LookupQuery:
    cnr: Optional[str] = None
    case_ref: Optional[str] = None
    title: Optional[str] = None
    party: Optional[str] = None
    judge: Optional[str] = None
    date: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None
    bench: Optional[str] = None
    limit: int = 20
    offset: int = 0

    CRITERIA = ("cnr", "case_ref", "title", "party", "judge", "date", "date_from", "date_to", "year", "court", "bench")

    def normalized(self) -> "LookupQuery":
        """A validated copy with blanks removed. Raises LookupError_ for anything the caller must fix."""
        values: Dict[str, Any] = {}
        for name in ("cnr", "case_ref", "title", "party", "judge", "court", "bench"):
            raw = getattr(self, name)
            text = re.sub(r"\s+", " ", raw).strip() if isinstance(raw, str) else None
            if text and len(text) > MAX_TEXT_CHARS:
                raise LookupError_(f"{name} is longer than {MAX_TEXT_CHARS} characters.")
            values[name] = text or None
        values["cnr"] = normalize_cnr(values["cnr"]) if values["cnr"] else None
        for name in ("date", "date_from", "date_to"):
            values[name] = _check_date(getattr(self, name), name)
        if values["date_from"] and values["date_to"] and values["date_from"] > values["date_to"]:
            raise LookupError_("date_from must not be later than date_to.")
        if self.year is not None and not 1800 <= int(self.year) <= 2200:
            raise LookupError_("year must be a four-digit year.")
        values["year"] = int(self.year) if self.year is not None else None
        if not 1 <= int(self.limit) <= MAX_LIMIT:
            raise LookupError_(f"limit must be between 1 and {MAX_LIMIT}.")
        if not 0 <= int(self.offset) <= MAX_OFFSET:
            raise LookupError_(f"offset must be between 0 and {MAX_OFFSET}.")
        result = LookupQuery(**values, limit=int(self.limit), offset=int(self.offset))
        if not any(getattr(result, name) is not None for name in self.CRITERIA):
            raise LookupError_("Give at least one of: " + ", ".join(self.CRITERIA) + ".")
        return result

    def echo(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.CRITERIA if getattr(self, name) is not None}


@dataclass
class LookupResult:
    status: str  # "found" or "not_in_our_data"
    message: str
    total: int
    offset: int
    limit: int
    results: List[Dict[str, Any]]
    criteria: Dict[str, Any]
    coverage: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


def _record(row: sqlite3.Row, via: Optional[str] = None) -> Dict[str, Any]:
    data = {name: row[name] for name in COLUMNS}
    data["in_search_index"] = bool(data["in_search_index"])
    data["date_conflict"] = bool(data["date_conflict"])
    data["judge_display"] = data["judge"] or "not recorded"
    code, number, year = data["case_type_code"], data["case_number"], data["case_year"]
    data["case_reference"] = f"{code}/{number}/{year}" if code and number and year else None
    data["matched_via_duplicate_cnr"] = via
    return data


class CaseLookupIndex:
    """Read-only lookup over the SQLite file written by ``build`` (safe to use from several threads)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Case lookup index not found: {self.path}")
        self._local = threading.local()
        meta = dict(self._connection().execute("SELECT key, value FROM meta").fetchall()) if self._has("meta") else {}
        if meta.get("schema_version") != str(SCHEMA_VERSION):
            raise ValueError(f"{self.path} is not a case lookup index of version {SCHEMA_VERSION}.")
        self.meta = meta

    def _has(self, table: str) -> bool:
        row = self._connection().execute("SELECT 1 FROM sqlite_master WHERE type IN ('table') AND name = ?", (table,)).fetchone()
        return row is not None

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def __len__(self) -> int:
        return int(self._connection().execute("SELECT COUNT(*) FROM cases").fetchone()[0])

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        path: Union[str, Path],
        rows: Iterable[Mapping[str, Any]],
        links: Iterable[Mapping[str, Any]] = (),
        batch_rows: int = 20_000,
    ) -> "CaseLookupIndex":
        """Writes a new index from corpus rows (metadata columns) and near-duplicate links.

        Rows need ``cnr``; every other column is optional. A repeated CNR raises ``ValueError``. The file is
        written to a temporary name and renamed, so a failed build leaves nothing behind.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".building")
        for stale in (temporary, Path(str(temporary) + "-journal")):
            stale.unlink(missing_ok=True)
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute(
                "CREATE TABLE cases (cnr_norm TEXT NOT NULL UNIQUE, cnr TEXT NOT NULL, title TEXT, petitioner TEXT, "
                "respondent TEXT, judge TEXT, judgment_date TEXT, judgment_year INTEGER, court_name TEXT, "
                "bench_name TEXT, case_type_code TEXT, case_type_name TEXT, case_number TEXT, case_year TEXT, "
                "outcome TEXT, in_search_index INTEGER, date_conflict INTEGER, date_source TEXT, source_config TEXT)"
            )
            connection.execute("CREATE TABLE links (duplicate_norm TEXT PRIMARY KEY, duplicate_cnr TEXT, kept_norm TEXT NOT NULL)")
            connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            insert = f"INSERT INTO cases VALUES ({', '.join('?' * (len(COLUMNS) + 1))})"
            batch: List[tuple] = []
            count = 0

            def flush() -> None:
                nonlocal batch
                if batch:
                    connection.executemany(insert, batch)
                    batch = []

            for row in rows:
                cnr = str(row.get("cnr") or "").strip()
                if not cnr:
                    raise ValueError(f"Row {count} has no cnr.")
                number, year = _clean_number(row.get("case_number")), row.get("case_year")
                values = {c: row.get(c) for c in COLUMNS}
                values.update(cnr=cnr, case_number=number, case_year=str(year) if year not in (None, "") else None,
                              in_search_index=int(bool(row.get("in_search_index"))), date_conflict=int(bool(row.get("date_conflict"))))
                batch.append((normalize_cnr(cnr), *(values[c] for c in COLUMNS)))
                count += 1
                if len(batch) >= batch_rows:
                    flush()
            flush()
            if count == 0:
                raise ValueError("Cannot build a lookup index with no rows.")
            link_rows = [(normalize_cnr(l["duplicate_cnr"]), l["duplicate_cnr"], normalize_cnr(l["kept_cnr"])) for l in links]
            connection.executemany("INSERT OR REPLACE INTO links VALUES (?, ?, ?)", link_rows)
            connection.execute("CREATE INDEX cases_case ON cases (case_number, case_year)")
            connection.execute("CREATE INDEX cases_date ON cases (judgment_date)")
            connection.execute("CREATE INDEX cases_year ON cases (judgment_year)")
            connection.execute(
                "CREATE VIRTUAL TABLE cases_fts USING fts5(title, petitioner, respondent, judge, content='cases', "
                "content_rowid='rowid', tokenize='unicode61 remove_diacritics 2')"
            )
            connection.execute("INSERT INTO cases_fts(cases_fts) VALUES ('rebuild')")
            coverage = connection.execute(
                "SELECT court_name, COUNT(*), MIN(judgment_date), MAX(judgment_date) FROM cases GROUP BY court_name ORDER BY court_name"
            ).fetchall()
            import json

            connection.executemany("INSERT INTO meta VALUES (?, ?)", [
                ("schema_version", str(SCHEMA_VERSION)),
                ("records", str(count)),
                ("links", str(len(link_rows))),
                ("built_at_utc", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")),
                ("coverage", json.dumps([{"court": c or "unknown", "records": n, "first_date": lo, "last_date": hi} for c, n, lo, hi in coverage])),
            ])
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise ValueError("A cnr appears twice in the input; every record must have a unique cnr.") from exc
        except BaseException:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        connection.close()
        temporary.replace(path)
        return cls(path)

    # ------------------------------------------------------------------ search

    def coverage(self) -> List[Dict[str, Any]]:
        import json

        return json.loads(self.meta.get("coverage", "[]"))

    def _coverage_text(self) -> str:
        parts = []
        for item in self.coverage():
            span = f" ({item['first_date']} to {item['last_date']})" if item.get("first_date") else ""
            parts.append(f"{item['court']}{span}: {item['records']:,} records")
        return "; ".join(parts) if parts else "the records in this index"

    def search(self, query: LookupQuery) -> LookupResult:
        """Finds records matching ALL the given criteria. Raises LookupError_ for an invalid request."""
        q = query.normalized()
        warnings: List[str] = []
        where: List[Tuple[str, List[Any]]] = []  # (SQL condition, its parameters), joined with AND
        fts: List[str] = []

        def escape(text: str) -> str:
            return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        if q.cnr:
            where.append(("cases.cnr_norm = ?", [q.cnr]))
        sub_type_clause: Optional[Tuple[str, List[Any]]] = None
        if q.case_ref:
            code, number, year = parse_case_query(q.case_ref)
            where.append(("cases.case_number = ? AND cases.case_year = ?", [number, year]))
            if code:
                # The exact type first. Only if nothing matches are its sub-types tried (WP -> WP(L)), because
                # WP(L) is a separate numbering series and a search for WP/12/2010 should not mix the two.
                where.append(("cases.case_type_code = ?", [code]))
                if "(" not in code:
                    sub_type_clause = ("cases.case_type_code LIKE ? ESCAPE '\\'", [escape(code) + "(%"])
            else:
                warnings.append("The case type was not recognised or not given, so only the number and year were matched.")
        for text, columns in ((q.title, "title petitioner respondent"), (q.party, "petitioner respondent"), (q.judge, "judge")):
            if text:
                expression = fts_expression(columns, text)
                if expression is None:
                    raise LookupError_("Names need at least one word of two or more letters.")
                fts.append(expression)
        if q.date:
            where.append(("cases.judgment_date = ?", [q.date]))
        if q.date_from:
            where.append(("cases.judgment_date >= ?", [q.date_from]))
        if q.date_to:
            where.append(("cases.judgment_date <= ?", [q.date_to]))
        if q.year is not None:
            where.append(("cases.judgment_year = ?", [q.year]))
        for text, column in ((q.court, "court_name"), (q.bench, "bench_name")):
            if text:
                where.append((f"lower(cases.{column}) LIKE ? ESCAPE '\\'", ["%" + escape(text.lower()) + "%"]))

        rows, total, via = self._attempt(where, fts, q)
        if not rows and sub_type_clause is not None:
            widened = [w for w in where if not w[0].startswith("cases.case_type_code = ")] + [sub_type_clause]
            rows, total, via = self._attempt(widened, fts, q)
            if rows:
                warnings.append("No record has exactly that case type, so records of its sub-types are shown.")
        coverage = self.coverage()
        if rows:
            records = [_record(r, via) for r in rows]
            note = f" The CNR {q.cnr} was merged into a near-identical judgment; the record shown is that one." if via else ""
            message = f"{total:,} record{'s' if total != 1 else ''} in our data match.{note}"
            if q.judge or q.date or q.date_from or q.date_to or q.year is not None:
                warnings.append("Dates and judge names are as recorded in the data and can differ from the judgment text.")
            return LookupResult("found", message, total, q.offset, q.limit, records, q.echo(), coverage, warnings)
        message = (
            f"{NOT_FOUND}: no record matching these details was found. Our data covers {self._coverage_text()}. "
            "A judgment outside that coverage, or recorded under different details or spelling, would not appear here; "
            "check the court's own records."
        )
        return LookupResult("not_in_our_data", message, 0, q.offset, q.limit, [], q.echo(), coverage, warnings)

    def _attempt(self, where: List[Tuple[str, List[Any]]], fts: List[str], q: LookupQuery) -> Tuple[List[sqlite3.Row], int, Optional[str]]:
        rows, total, via = self._run(where, fts, q)
        if not rows and q.cnr:
            rows, total, via = self._through_duplicate_link(where, fts, q)
        return rows, total, via

    def _run(self, where: List[Tuple[str, List[Any]]], fts: List[str], q: LookupQuery) -> Tuple[List[sqlite3.Row], int, Optional[str]]:
        values: List[Any] = []
        conditions = ""
        select_cols = ", ".join(f"cases.{c}" for c in COLUMNS)
        if fts:
            base = "FROM cases_fts JOIN cases ON cases.rowid = cases_fts.rowid WHERE cases_fts MATCH ?"
            values.append(" AND ".join(f"({e})" for e in fts))
            order = "ORDER BY cases_fts.rank, cases.judgment_date DESC, cases.cnr_norm"
        else:
            base = "FROM cases WHERE 1 = 1"
            order = "ORDER BY cases.judgment_date IS NULL, cases.judgment_date DESC, cases.cnr_norm"
        for clause, clause_params in where:
            conditions += f" AND ({clause})"
            values.extend(clause_params)
        connection = self._connection()
        total = connection.execute(f"SELECT COUNT(*) {base}{conditions}", values).fetchone()[0]
        rows = connection.execute(f"SELECT {select_cols} {base}{conditions} {order} LIMIT ? OFFSET ?", values + [q.limit, q.offset]).fetchall()
        return rows, int(total), None

    def _through_duplicate_link(self, where: List[Tuple[str, List[Any]]], fts: List[str], q: LookupQuery) -> Tuple[List[sqlite3.Row], int, Optional[str]]:
        """A CNR the corpus build merged into another judgment resolves to the record that was kept."""
        link = self._connection().execute("SELECT duplicate_cnr, kept_norm FROM links WHERE duplicate_norm = ?", (q.cnr,)).fetchone()
        if link is None:
            return [], 0, None
        swapped = [(c, [link["kept_norm"]] if c == "cases.cnr_norm = ?" else p) for c, p in where]
        rows, total, _ = self._run(swapped, fts, q)
        return rows, total, link["duplicate_cnr"] if rows else None

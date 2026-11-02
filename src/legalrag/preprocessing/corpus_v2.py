"""
Pure building blocks for the v2 corpus (Bombay High Court plus Supreme Court).

Used by ``scripts/build_corpus_v2.py`` (C1) and ``scripts/data_quality_report.py`` (C2).
Nothing here touches the network, models or the filesystem, so every rule can be tested
with small fixtures.

Rules implemented, in the order they are applied to each record:

1. Select the court (Bombay High Court is ``court_code = 27_1``).
2. Drop records with no CNR, a repeated CNR, no text, or corrupted text.
3. Clean the text: strip control characters (the v1 cleaner), then collapse extraction
   repeats such as ``CORAM ... CORAM ... CORAM`` in the first part of the text.
4. Route the record: reasoned judgments (``clean_chars >= min_chars``) go to the search
   corpus; shorter records go to the lookup-only set. Neither is dropped.
5. Remove near duplicates from the search corpus only, and keep a link from each removed
   CNR to the CNR that was kept, so case lookup can still resolve it.

Metadata rules: readable bench names, case type parsed from the title, and a date rule
that rejects impossible dates (the source has CNR dates in the year 3021).

Assumption not yet checked against the real data: a CNR looks like
``<PREFIX>_<serial>_<YYYY-MM-DD>``. This is taken from ``scripts/profile_bombay_hc.py``,
which reads the CNR date with ``split_part(cnr, '_', 3)``. If it is wrong, the build
report shows a very high ``cnr_date_status`` count of ``unparseable``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from legalrag.preprocessing.cleaner import clean_judgment_text, is_corrupted_document

BOMBAY_COURT_CODE = "27_1"
# The Bombay High Court was established in 1862; earlier dates are not plausible.
MIN_PLAUSIBLE_YEAR = 1862

# ---------------------------------------------------------------------------
# Courts and benches
# ---------------------------------------------------------------------------

BOMBAY_COURT_NAME = "Bombay High Court"
SUPREME_COURT_NAME = "Supreme Court of India"

# bench code -> (readable name, expected CNR prefix, name is inferred rather than confirmed)
# Source: docs/DATA_PROFILE_BOMBAY.md. "Inferred" means the profile could not confirm the
# name from judgment headers; the build report repeats this so it is not forgotten.
BENCH_NAMES: Dict[str, Tuple[str, str, bool]] = {
    "newas": ("Bombay (Appellate Side)", "HCBM01", True),
    "newos": ("Bombay (Original Side)", "HCBM02", True),
    "hcaurdb": ("Aurangabad", "HCBM03", False),
    "testcase": ("Nagpur", "HCBM04", False),
    "hcbgoa": ("Goa", "HCBM05", False),
    "kolhcdb": ("Kolhapur", "HCBM07", False),
}
_PREFIX_TO_BENCH = {prefix: bench for bench, (_n, prefix, _i) in BENCH_NAMES.items()}

# ---------------------------------------------------------------------------
# Case type
# ---------------------------------------------------------------------------

# Starting set only, limited to abbreviations that are unambiguous at the Bombay High Court.
# The data quality report lists the most common codes left unmapped so this table can be
# extended after reading real output. Never applied to Supreme Court records, where the same
# letters mean something else (for example "CA" is a Civil Appeal there).
BOMBAY_CASE_TYPE_NAMES: Dict[str, str] = {
    "WP": "Writ Petition",
    "PIL": "Public Interest Litigation",
    "CRWP": "Criminal Writ Petition",
    "FA": "First Appeal",
    "SA": "Second Appeal",
    "CRA": "Criminal Appeal",
    "ARBP": "Arbitration Petition",
    "LPA": "Letters Patent Appeal",
    "AO": "Appeal from Order",
}

_NUMBER_AND_YEAR = r"\s*/\s*(?P<number>\d{1,7})\s*/\s*(?P<year>(?:18|19|20)\d{2})(?!\d)"

# A title that STARTS with the reference: a code of up to three space-separated tokens, such as
# "CR. MISC./57199/2022". Tokens must be capital letters and dots (no lowercase), so a title like
# "Order in WP/5/2001" or "Ram v. WP/5/2001" is not read as the code "ORDERINWP".
_LEADING_CASE_REFERENCE = re.compile(
    r"^\s*(?P<code>[A-Z][A-Z.&\-]*(?:\s+[A-Z][A-Z.&\-]*){0,2}(?:\s*\([A-Za-z]{1,6}\))?)"
    + _NUMBER_AND_YEAR
)

# Otherwise the reference may appear anywhere in the title; the code is then one token.
_CASE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<code>[A-Za-z][A-Za-z.&\-]*(?:\s*\([A-Za-z]{1,6}\))?)" + _NUMBER_AND_YEAR
)


def parse_case_reference(title: Optional[str]) -> Optional[Dict[str, str]]:
    """Reads a reference such as ``WP/1519/2000`` from a title.

    A reference at the start of the title keeps a multi-word code whole ("CR. MISC./57199/2022"
    gives ``CRMISC``). A reference elsewhere in the title uses the single token before the first
    slash. Returns ``code`` (uppercase letters and parentheses only, for example ``WP(L)``),
    ``number`` and ``year``, or None when the title holds no such reference.
    """
    if not title or not isinstance(title, str):
        return None
    match = _LEADING_CASE_REFERENCE.match(title) or _CASE_REFERENCE.search(title)
    if match is None:
        return None
    code = re.sub(r"[^A-Z()]", "", match.group("code").upper())
    if not code:
        return None
    return {"code": code, "number": match.group("number"), "year": match.group("year")}


def case_type_name(code: Optional[str], is_supreme_court: bool = False) -> Optional[str]:
    """Maps a parsed case type code to a readable name, or None if it is not in the table."""
    if not code or is_supreme_court:
        return None
    return BOMBAY_CASE_TYPE_NAMES.get(re.sub(r"\(.*\)$", "", code))


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

DATE_RULES = ("prefer_decision", "prefer_cnr", "agree_only")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def parse_iso_date(value: Any) -> Optional[dt.date]:
    """Parses ``YYYY-MM-DD`` (a longer timestamp is cut to its date). Returns None if invalid."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    match = _ISO_DATE.match(str(value).strip())
    if match is None:
        return None
    try:
        return dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def cnr_date_text(cnr: Optional[str]) -> Optional[str]:
    """Returns the third underscore-separated part of a CNR (the date), or None."""
    if not cnr or not isinstance(cnr, str):
        return None
    parts = cnr.split("_")
    return parts[2] if len(parts) >= 3 else None


def _date_status(value: Any, today: dt.date) -> Tuple[Optional[dt.date], str]:
    """Returns the date and a status: ok, missing, unparseable or implausible."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing"
    parsed = parse_iso_date(value)
    if parsed is None:
        return None, "unparseable"
    if parsed.year < MIN_PLAUSIBLE_YEAR or parsed > today:
        return None, "implausible"
    return parsed, "ok"


@dataclass(frozen=True)
class DateResolution:
    """Outcome of applying a date rule to one record."""

    judgment_date: Optional[str]
    date_source: Optional[str]  # "decision_date", "cnr" or None
    decision_date_status: str
    cnr_date_status: str
    conflict: bool  # both dates usable and different
    year_conflict: bool  # both dates usable and in different years


def resolve_date(
    decision_date: Any,
    cnr: Optional[str],
    rule: str = "prefer_decision",
    today: Optional[dt.date] = None,
) -> DateResolution:
    """Chooses the judgment date from ``decision_date`` and the date inside the CNR.

    Dates earlier than 1862 or later than ``today`` are rejected, not repaired.

    Rules:
        prefer_decision: use ``decision_date``; fall back to the CNR date if it is unusable.
        prefer_cnr: use the CNR date; fall back to ``decision_date``.
        agree_only: use a date only when both agree exactly; otherwise leave it empty.
    """
    if rule not in DATE_RULES:
        raise ValueError(f"Unknown date rule {rule!r}; expected one of {DATE_RULES}.")
    today = today or dt.date.today()
    decided, decision_status = _date_status(decision_date, today)
    from_cnr, cnr_status = _date_status(cnr_date_text(cnr), today)

    both = decided is not None and from_cnr is not None
    conflict = both and decided != from_cnr
    year_conflict = both and decided.year != from_cnr.year

    chosen: Optional[dt.date]
    source: Optional[str]
    if rule == "agree_only":
        chosen, source = (decided, "decision_date") if both and not conflict else (None, None)
    else:
        order = [(decided, "decision_date"), (from_cnr, "cnr")]
        if rule == "prefer_cnr":
            order.reverse()
        chosen, source = next(((d, s) for d, s in order if d is not None), (None, None))

    return DateResolution(
        judgment_date=chosen.isoformat() if chosen else None,
        date_source=source,
        decision_date_status=decision_status,
        cnr_date_status=cnr_status,
        conflict=conflict,
        year_conflict=year_conflict,
    )


# ---------------------------------------------------------------------------
# Repeated text
# ---------------------------------------------------------------------------

_WORD = re.compile(r"\S+")
_NON_WORD = re.compile(r"\W+")


def collapse_repeated_phrases(
    text: str,
    head_chars: int = 3000,
    min_words: int = 3,
    max_words: int = 60,
) -> Tuple[str, int]:
    """Collapses a phrase that is repeated back to back, such as ``CORAM ... CORAM ... CORAM``.

    Only the first ``head_chars`` characters are examined, because the repeats seen in the
    data profile are extraction repeats in the header. Only exact consecutive repeats are
    removed (ignoring case and punctuation), at least ``min_words`` words long, so ordinary
    legal phrases that merely recur are left alone. One copy is kept.

    Returns the cleaned text and the number of characters removed.
    """
    if not text or head_chars <= 0:
        return text, 0

    cut = min(head_chars, len(text))
    while cut < len(text) and not text[cut].isspace():
        cut += 1
    head, tail = text[:cut], text[cut:]

    spans = [(m.start(), m.end()) for m in _WORD.finditer(head)]
    norm = [_NON_WORD.sub("", head[s:e]).lower() for s, e in spans]
    total = len(norm)
    if total < 2 * min_words:
        return text, 0

    positions: Dict[str, List[int]] = {}
    for index, token in enumerate(norm):
        positions.setdefault(token, []).append(index)

    removals: List[Tuple[int, int]] = []
    i = 0
    while i < total:
        match_len = 0
        # Shortest period first, so a phrase repeated 20 times collapses to one copy, not ten.
        # A later position holding the same word is a candidate period.
        candidates = [p - i for p in positions[norm[i]] if min_words <= p - i <= max_words]
        for n in sorted(candidates):
            if i + 2 * n > total:
                continue
            unit = norm[i : i + n]
            if unit == norm[i + n : i + 2 * n] and any(unit):
                match_len = n
                break
        if not match_len:
            i += 1
            continue
        end = i + match_len
        while end + match_len <= total and norm[end : end + match_len] == norm[i : i + match_len]:
            end += match_len
        removals.append((spans[i + match_len - 1][1], spans[end - 1][1]))
        i = end

    if not removals:
        return text, 0

    pieces: List[str] = []
    last = 0
    removed = 0
    for start, stop in removals:
        pieces.append(head[last:start])
        removed += stop - start
        last = stop
    pieces.append(head[last:])
    return "".join(pieces) + tail, removed


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

DEDUP_MODES = ("prefix", "exact", "off")


def text_fingerprint(text: str, mode: str = "prefix", prefix_chars: int = 2000) -> str:
    """Hashes the text with case, spaces and punctuation removed.

    ``prefix`` uses the first ``prefix_chars`` normalised characters, which is the near
    duplicate definition used in the data profile. ``exact`` uses the whole text.
    """
    if mode == "prefix":
        # Normalise only as much text as is needed to get prefix_chars characters; same result
        # as normalising everything, without scanning a whole judgment to hash its first 2,000.
        window = max(prefix_chars * 2, 1)
        while True:
            normalised = _NON_WORD.sub("", text[:window]).lower()
            if len(normalised) >= prefix_chars or window >= len(text):
                break
            window *= 2
        normalised = normalised[:prefix_chars]
    elif mode == "exact":
        normalised = _NON_WORD.sub("", text).lower()
    else:
        raise ValueError(f"Unknown fingerprint mode {mode!r}.")
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


class ReservoirSampler:
    """Keeps a uniform random sample of ``size`` items from a stream, reproducibly."""

    def __init__(self, size: int, seed: int = 42):
        self.size = size
        self._rng = random.Random(seed)
        self._seen = 0
        self.items: List[Any] = []

    def add(self, item: Any) -> None:
        self._seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        slot = self._rng.randrange(self._seen)
        if slot < self.size:
            self.items[slot] = item

    @property
    def seen(self) -> int:
        return self._seen


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

DROP_REASONS = (
    "court_not_selected",
    "missing_cnr",
    "duplicate_cnr",
    "empty_text",
    "corrupted_text",
    "no_valid_date",
    "near_duplicate",
)

# (column, type) pairs for the output tables; the script maps type names to Arrow types.
OUTPUT_FIELDS: List[Tuple[str, str]] = [
    ("cnr", "string"),
    ("source_config", "string"),
    ("court_code", "string"),
    ("court_name", "string"),
    ("bench", "string"),
    ("bench_name", "string"),
    ("bench_name_inferred", "bool"),
    ("title", "string"),
    ("petitioner", "string"),
    ("respondent", "string"),
    ("judge", "string"),
    ("judgment_date", "string"),
    ("judgment_year", "int32"),
    ("date_source", "string"),
    ("decision_date_raw", "string"),
    ("cnr_date_raw", "string"),
    ("decision_date_status", "string"),
    ("cnr_date_status", "string"),
    ("date_conflict", "bool"),
    ("year_conflict", "bool"),
    ("case_type", "string"),
    ("case_type_source", "string"),
    ("case_type_code", "string"),
    ("case_type_name", "string"),
    ("case_number", "string"),
    ("case_year", "string"),
    ("outcome", "string"),
    ("raw_chars", "int64"),
    ("clean_chars", "int64"),
    ("repeat_chars_removed", "int64"),
    ("in_search_index", "bool"),
    ("text", "string"),
]


@dataclass(frozen=True)
class SourceSpec:
    """One input config of the dataset and which of its records to select."""

    config: str
    court_code: Optional[str]  # None keeps every record
    court_name: str
    bench_name_if_missing: Optional[str] = None
    is_supreme_court: bool = False


def default_sources(bombay_court_code: str = BOMBAY_COURT_CODE) -> Dict[str, SourceSpec]:
    return {
        "high_courts": SourceSpec("high_courts", bombay_court_code, BOMBAY_COURT_NAME),
        "supreme_court": SourceSpec(
            "supreme_court",
            None,
            SUPREME_COURT_NAME,
            bench_name_if_missing=SUPREME_COURT_NAME,
            is_supreme_court=True,
        ),
    }


@dataclass
class BuildConfig:
    min_chars: int = 5000
    date_rule: str = "prefer_decision"
    require_date: bool = False
    dedup_mode: str = "prefix"
    dedup_prefix_chars: int = 2000
    repeat_head_chars: int = 3000
    repeat_min_words: int = 3
    repeat_max_words: int = 60
    clean_repeats: bool = True
    today: Optional[dt.date] = None

    def __post_init__(self) -> None:
        if self.date_rule not in DATE_RULES:
            raise ValueError(f"date_rule must be one of {DATE_RULES}.")
        if self.dedup_mode not in DEDUP_MODES:
            raise ValueError(f"dedup_mode must be one of {DEDUP_MODES}.")
        if self.today is None:
            self.today = dt.date.today()


@dataclass
class BuildStats:
    """Counts for the build report. Every scanned record ends in exactly one outcome."""

    scanned: int = 0
    dropped: Counter = field(default_factory=Counter)
    kept_search: int = 0
    kept_lookup_only: int = 0
    by_source: Dict[str, Counter] = field(default_factory=dict)
    decision_date_status: Counter = field(default_factory=Counter)
    cnr_date_status: Counter = field(default_factory=Counter)
    date_source: Counter = field(default_factory=Counter)
    date_conflicts: int = 0
    year_conflicts: int = 0
    unknown_bench: Counter = field(default_factory=Counter)
    bench_cnr_prefix_mismatch: int = 0
    case_type_source: Counter = field(default_factory=Counter)
    unmapped_case_codes: Counter = field(default_factory=Counter)
    repeat_cleaned_records: int = 0
    repeat_chars_removed: int = 0
    near_duplicate_links: int = 0

    def source_counter(self, source: str) -> Counter:
        return self.by_source.setdefault(source, Counter())

    @property
    def accounted(self) -> int:
        return sum(self.dropped.values()) + self.kept_search + self.kept_lookup_only

    def identity_holds(self) -> bool:
        """True when scanned == dropped + kept, i.e. no record vanished without a reason."""
        return self.scanned == self.accounted

    def to_dict(self, top_unmapped: int = 25) -> Dict[str, Any]:
        return {
            "scanned": self.scanned,
            "dropped_by_reason": {reason: self.dropped.get(reason, 0) for reason in DROP_REASONS},
            "dropped_total": sum(self.dropped.values()),
            "kept_search_index": self.kept_search,
            "kept_lookup_only": self.kept_lookup_only,
            "identity_holds": self.identity_holds(),
            "by_source": {name: dict(counter) for name, counter in sorted(self.by_source.items())},
            "decision_date_status": dict(self.decision_date_status),
            "cnr_date_status": dict(self.cnr_date_status),
            "date_source": dict(self.date_source),
            "date_conflicts": self.date_conflicts,
            "year_conflicts": self.year_conflicts,
            "unknown_bench": dict(self.unknown_bench),
            "bench_cnr_prefix_mismatch": self.bench_cnr_prefix_mismatch,
            "case_type_source": dict(self.case_type_source),
            "top_unmapped_case_codes": dict(self.unmapped_case_codes.most_common(top_unmapped)),
            "repeat_cleaned_records": self.repeat_cleaned_records,
            "repeat_chars_removed": self.repeat_chars_removed,
            "near_duplicate_links": self.near_duplicate_links,
        }


@dataclass
class BuildResult:
    """What happened to one record. ``row`` is set unless the record was dropped."""

    row: Optional[Dict[str, Any]]
    route: str  # "search", "lookup_only" or "dropped"
    drop_reason: Optional[str] = None
    duplicate_of: Optional[str] = None  # winner CNR when dropped as a near duplicate
    cnr: Optional[str] = None


def _blank_to_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CorpusBuilder:
    """Applies the v2 corpus rules to records one at a time and counts every outcome."""

    def __init__(self, config: Optional[BuildConfig] = None):
        self.config = config or BuildConfig()
        self.stats = BuildStats()
        self._seen_cnrs: set = set()
        self._fingerprints: Dict[str, str] = {}

    def note_prefiltered(self, source: str, count: int) -> None:
        """Counts records the reader skipped with a court filter, so totals still reconcile."""
        if count <= 0:
            return
        self.stats.scanned += count
        self.stats.dropped["court_not_selected"] += count
        self.stats.source_counter(source)["court_not_selected"] += count

    def process(self, record: Mapping[str, Any], source: SourceSpec) -> BuildResult:
        cfg = self.config
        stats = self.stats
        per_source = stats.source_counter(source.config)
        stats.scanned += 1

        def drop(
            reason: str, duplicate_of: Optional[str] = None, cnr: Optional[str] = None
        ) -> BuildResult:
            stats.dropped[reason] += 1
            per_source[reason] += 1
            return BuildResult(None, "dropped", reason, duplicate_of, cnr)

        if source.court_code is not None and _blank_to_none(record.get("court_code")) != source.court_code:
            return drop("court_not_selected")

        cnr = _blank_to_none(record.get("cnr"))
        if cnr is None:
            return drop("missing_cnr")
        if cnr in self._seen_cnrs:
            return drop("duplicate_cnr", cnr=cnr)
        self._seen_cnrs.add(cnr)

        raw_text = record.get("full_text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            return drop("empty_text", cnr=cnr)
        if is_corrupted_document(raw_text):
            return drop("corrupted_text", cnr=cnr)

        text = clean_judgment_text(raw_text)
        removed = 0
        if cfg.clean_repeats:
            text, removed = collapse_repeated_phrases(
                text, cfg.repeat_head_chars, cfg.repeat_min_words, cfg.repeat_max_words
            )

        date = resolve_date(record.get("decision_date"), cnr, cfg.date_rule, cfg.today)

        reasoned = len(text) >= cfg.min_chars
        if reasoned and cfg.require_date and date.judgment_date is None:
            return drop("no_valid_date", cnr=cnr)

        duplicate_of: Optional[str] = None
        if reasoned and cfg.dedup_mode != "off":
            fingerprint = text_fingerprint(text, cfg.dedup_mode, cfg.dedup_prefix_chars)
            duplicate_of = self._fingerprints.get(fingerprint)
            if duplicate_of is None:
                self._fingerprints[fingerprint] = cnr
            else:
                stats.near_duplicate_links += 1
                return drop("near_duplicate", duplicate_of=duplicate_of, cnr=cnr)

        # Date and cleaning counts describe the records that are kept, not the dropped ones.
        stats.decision_date_status[date.decision_date_status] += 1
        stats.cnr_date_status[date.cnr_date_status] += 1
        stats.date_conflicts += int(date.conflict)
        stats.year_conflicts += int(date.year_conflict)
        stats.date_source[date.date_source or "none"] += 1
        if removed:
            stats.repeat_cleaned_records += 1
            stats.repeat_chars_removed += removed

        row = self._build_row(record, source, cnr, text, raw_text, removed, date, reasoned)
        if reasoned:
            stats.kept_search += 1
            per_source["kept_search"] += 1
            return BuildResult(row, "search", cnr=cnr)
        stats.kept_lookup_only += 1
        per_source["kept_lookup_only"] += 1
        return BuildResult(row, "lookup_only", cnr=cnr)

    def _build_row(
        self,
        record: Mapping[str, Any],
        source: SourceSpec,
        cnr: str,
        text: str,
        raw_text: str,
        removed: int,
        date: DateResolution,
        reasoned: bool,
    ) -> Dict[str, Any]:
        stats = self.stats
        bench = _blank_to_none(record.get("bench"))
        bench_name = source.bench_name_if_missing
        inferred = False
        if bench in BENCH_NAMES:
            bench_name, expected_prefix, inferred = BENCH_NAMES[bench]
            if not cnr.startswith(expected_prefix):
                stats.bench_cnr_prefix_mismatch += 1
        elif not source.is_supreme_court:
            stats.unknown_bench[bench or "(blank)"] += 1
            bench_name = f"Unknown ({bench})" if bench else "Unknown"
            by_prefix = _PREFIX_TO_BENCH.get(cnr[:6])
            if by_prefix:
                bench_name = BENCH_NAMES[by_prefix][0]
                inferred = True

        title = _blank_to_none(record.get("title"))
        reference = parse_case_reference(title)
        dataset_case_type = _blank_to_none(record.get("case_type"))
        code = reference["code"] if reference else None
        if dataset_case_type:
            case_type, case_type_source = dataset_case_type, "dataset"
        elif code:
            case_type, case_type_source = code, "title"
        else:
            case_type, case_type_source = None, "none"
        stats.case_type_source[case_type_source] += 1
        name = case_type_name(code, source.is_supreme_court)
        if code and name is None and not source.is_supreme_court:
            stats.unmapped_case_codes[re.sub(r"\(.*\)$", "", code)] += 1

        judgment_year = int(date.judgment_date[:4]) if date.judgment_date else _as_int(record.get("year"))

        return {
            "cnr": cnr,
            "source_config": source.config,
            "court_code": _blank_to_none(record.get("court_code")) or ("SC" if source.is_supreme_court else None),
            "court_name": source.court_name,
            "bench": bench,
            "bench_name": bench_name,
            "bench_name_inferred": inferred,
            "title": title,
            "petitioner": _blank_to_none(record.get("petitioner")),
            "respondent": _blank_to_none(record.get("respondent")),
            "judge": _blank_to_none(record.get("judge")),
            "judgment_date": date.judgment_date,
            "judgment_year": judgment_year,
            "date_source": date.date_source,
            "decision_date_raw": _blank_to_none(record.get("decision_date")),
            "cnr_date_raw": cnr_date_text(cnr),
            "decision_date_status": date.decision_date_status,
            "cnr_date_status": date.cnr_date_status,
            "date_conflict": date.conflict,
            "year_conflict": date.year_conflict,
            "case_type": case_type,
            "case_type_source": case_type_source,
            "case_type_code": code,
            "case_type_name": name,
            "case_number": reference["number"] if reference else None,
            "case_year": reference["year"] if reference else None,
            "outcome": _blank_to_none(record.get("disposal_nature_normalized")),
            "raw_chars": len(raw_text),
            "clean_chars": len(text),
            "repeat_chars_removed": removed,
            "in_search_index": reasoned,
            "text": text,
        }

"""
Old-law to new-law section mapping for query expansion (plan task C6).

India's criminal codes were replaced in July 2024: the Indian Penal Code (IPC) by the Bharatiya Nyaya
Sanhita (BNS), the Code of Criminal Procedure (CrPC) by the Bharatiya Nagarik Suraksha Sanhita (BNSS),
and the Indian Evidence Act (IEA) by the Bharatiya Sakshya Adhiniyam (BSA). Judgments before that date
cite the old numbers and later ones cite the new numbers, so a lexical search for "Section 302 IPC" misses
a judgment that says "Section 103 BNS" and the reverse. This module finds section references in a query and
adds the other law's number so BM25 can match either form.

It contains NO section data. The mapping comes from a data file built from the official correspondence
table by ``scripts/build_statute_map.py`` and reviewed by a person; ``StatuteMap.load`` refuses a file that
nobody has marked as reviewed. A reference is only expanded when the query names the law ("Section 302
IPC", "s. 438 of the Cr.P.C.", "BNSS 482"); a bare "Section 302" is never guessed.

Limits to keep in mind: the table is many-to-many (one IPC section can map to several BNS sub-sections,
some IPC sections were omitted and some BNS sections are new), so a reference can map to several
counterparts or to none; the result says which. This is a search aid, not a statement of what the law is.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

FORMAT_VERSION = 1

#: pair id -> (old act key, new act key)
PAIRS: Dict[str, Tuple[str, str]] = {
    "IPC-BNS": ("IPC", "BNS"),
    "CRPC-BNSS": ("CRPC", "BNSS"),
    "IEA-BSA": ("IEA", "BSA"),
}

#: act key -> (display abbreviation, full name)
ACTS: Dict[str, Tuple[str, str]] = {
    "IPC": ("IPC", "Indian Penal Code, 1860"),
    "BNS": ("BNS", "Bharatiya Nyaya Sanhita, 2023"),
    "CRPC": ("CrPC", "Code of Criminal Procedure, 1973"),
    "BNSS": ("BNSS", "Bharatiya Nagarik Suraksha Sanhita, 2023"),
    "IEA": ("Evidence Act", "Indian Evidence Act, 1872"),
    "BSA": ("BSA", "Bharatiya Sakshya Adhiniyam, 2023"),
}

RELATIONSHIPS = ("same", "changed", "split", "merged", "omitted", "new")

# --------------------------------------------------------------------------- recognising a law in text

_DOTS = r"\.?\s*"


def _letters(word: str) -> str:
    """'CrPC' -> 'C\\.?\\s*r\\.?\\s*P...': tolerate dots and spaces between the letters ('Cr.P.C.')."""
    return _DOTS.join(re.escape(ch) for ch in word) + r"\.?"


_ACT_PATTERNS: List[Tuple[str, str]] = [
    # Longest names first so "BNSS" is not read as "BNS" + "S".
    ("BNSS", r"Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita|" + _letters("BNSS")),
    ("BNS", r"Bharatiya\s+Nyaya\s+Sanhita|" + _letters("BNS")),
    ("BSA", r"Bharatiya\s+Sakshya\s+Adhiniyam|" + _letters("BSA")),
    ("CRPC", r"Code\s+of\s+Criminal\s+Procedure|" + _letters("CrPC")),
    ("IPC", r"Indian\s+Penal\s+Code|" + _letters("IPC")),
    ("IEA", r"(?:Indian\s+)?Evidence\s+Act|" + _letters("IEA")),
]
_ACT_RE = "(?P<act>" + "|".join(f"(?P<act_{key}>{pattern})" for key, pattern in _ACT_PATTERNS) + r")(?![A-Za-z])"

_NUM = r"\d{1,3}[A-Za-z]?(?:\([0-9A-Za-z]{1,4}\))*"
_NUM_LIST = rf"{_NUM}(?:\s*(?:,|and|&|/|or)\s*{_NUM})*"
_YEAR_AFTER_ACT = r"(?:\s*,?\s*(?:of\s+)?(?:1860|1872|1973|2023))?"

# "Section 302 IPC", "ss. 302 and 34 of the Indian Penal Code, 1860", "u/s 438(2) Cr.P.C."
_SECTION_FIRST = re.compile(
    r"(?<![A-Za-z])(?:sections?|secs?\.?|ss\.?|s\.|u/s)\s*(?:no\.?\s*)?(?P<nums>" + _NUM_LIST + r")\s*,?\s*"
    r"(?:(?:of|under)\s+(?:the\s+)?|read\s+with\s+(?:the\s+)?)?" + _ACT_RE,
    re.IGNORECASE,
)
# "IPC 302", "BNS section 103(1)", "Cr.P.C. s. 482"
_ACT_FIRST = re.compile(
    r"(?<![A-Za-z])" + _ACT_RE + _YEAR_AFTER_ACT + r"\s*,?\s*(?:sections?|secs?\.?|ss\.?|s\.)?\s*(?P<nums>" + _NUM_LIST + r")(?![0-9A-Za-z])",
    re.IGNORECASE,
)
_NUM_SPLIT = re.compile(_NUM)


def normalize_section(text: str) -> str:
    """'498a' -> '498A', '438( 2 )' -> '438(2)', '85(A)' -> '85(a)': suffix letter upper, sub-clauses lower."""
    compact = re.sub(r"\s+", "", text)
    match = re.match(r"(\d+)([A-Za-z]?)((?:\([0-9A-Za-z]+\))*)$", compact)
    if not match:
        raise ValueError(f"Not a section reference: {text!r}")
    number, letter, rest = match.groups()
    return f"{int(number)}{letter.upper()}{rest.lower()}"


def base_section(section: str) -> str:
    """'103(1)(a)' -> '103'; '498A' -> '498A'."""
    return section.split("(", 1)[0]


@dataclass(frozen=True)
class SectionRef:
    """A section reference found in text."""

    act: str  # key in ACTS
    section: str  # normalised, for example '438(2)'
    text: str  # the matched text
    start: int
    end: int

    @property
    def base(self) -> str:
        return base_section(self.section)


@dataclass(frozen=True)
class Counterpart:
    act: str
    section: str
    relationship: str
    note: str = ""


@dataclass
class MatchedRef:
    ref: SectionRef
    status: str  # "mapped", "no_counterpart" (omitted or new) or "not_in_table"
    counterparts: List[Counterpart] = field(default_factory=list)


@dataclass
class Expansion:
    original: str
    matches: List[MatchedRef]
    additions: List[str]

    @property
    def retrieval_query(self) -> str:
        """The query with the other law's references appended (unchanged when there is nothing to add)."""
        return self.original if not self.additions else f"{self.original} {' '.join(self.additions)}"

    @property
    def expanded(self) -> bool:
        return bool(self.additions)


def find_references(text: str) -> List[SectionRef]:
    """Every 'Section N <law>' or '<law> N' reference in ``text``, in order, without overlaps."""
    found: List[SectionRef] = []
    taken: List[Tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < b and a < end for a, b in taken)

    for pattern in (_SECTION_FIRST, _ACT_FIRST):
        for match in pattern.finditer(text or ""):
            if overlaps(match.start(), match.end()):
                continue
            act = next(key for key, _ in _ACT_PATTERNS if match.group(f"act_{key}"))
            taken.append((match.start(), match.end()))
            for number in _NUM_SPLIT.finditer(match.group("nums")):
                found.append(SectionRef(act, normalize_section(number.group(0)), match.group(0), match.start(), match.end()))
    found.sort(key=lambda r: (r.start, r.section))
    return found


# --------------------------------------------------------------------------- the data file


class StatuteMapError(ValueError):
    """The mapping data file is missing, malformed or not reviewed."""


@dataclass(frozen=True)
class MapRow:
    pair: str
    old: Optional[str]  # None: a provision with no counterpart in the old law (new)
    new: Optional[str]  # None: a provision with no counterpart in the new law (omitted)
    relationship: str = "same"
    note: str = ""


def validate_row(raw: Mapping[str, Any], index: int = 0) -> MapRow:
    """Checks one row and returns it normalised. Raises StatuteMapError naming the row."""
    where = f"row {index}"
    pair = raw.get("pair")
    if pair not in PAIRS:
        raise StatuteMapError(f"{where}: pair must be one of {sorted(PAIRS)}, got {pair!r}")
    parts: List[Optional[str]] = []
    for key in ("old", "new"):
        value = raw.get(key)
        if value is None or str(value).strip() == "":
            parts.append(None)
            continue
        try:
            parts.append(normalize_section(str(value)))
        except ValueError as exc:
            raise StatuteMapError(f"{where}: {key} {value!r} is not a section number") from exc
    old, new = parts
    if old is None and new is None:
        raise StatuteMapError(f"{where}: a row needs an old or a new section")
    if old is None:
        expected = ("new",)  # a provision with no counterpart in the old law
    elif new is None:
        expected = ("omitted",)  # a provision with no counterpart in the new law
    else:
        expected = ("same", "changed", "split", "merged")
    relationship = str(raw.get("relationship") or expected[0])
    if relationship not in expected:
        raise StatuteMapError(
            f"{where}: relationship {relationship!r} does not fit old={old!r}, new={new!r}; expected one of {expected}"
        )
    return MapRow(pair, old, new, relationship, str(raw.get("note") or ""))


class StatuteMap:
    """Bidirectional section lookup and query expansion over a reviewed data file."""

    def __init__(self, rows: Iterable[MapRow], metadata: Optional[Dict[str, Any]] = None):
        self.rows = list(rows)
        self.metadata = metadata or {}
        seen = set()
        self._by_side: Dict[Tuple[str, str], Dict[str, List[MapRow]]] = {}
        for row in self.rows:
            key = (row.pair, row.old, row.new)
            if key in seen:
                raise StatuteMapError(f"duplicate row {key}")
            seen.add(key)
            old_act, new_act = PAIRS[row.pair]
            for act, section in ((old_act, row.old), (new_act, row.new)):
                if section is None:
                    continue
                index = self._by_side.setdefault((row.pair, act), {})
                index.setdefault(section, []).append(row)
                if base_section(section) != section:  # 103(1) is also found by a search for 103
                    index.setdefault(base_section(section), []).append(row)

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], require_reviewed: bool = True) -> "StatuteMap":
        if data.get("format") != FORMAT_VERSION:
            raise StatuteMapError(f"unsupported statute map format {data.get('format')!r}; expected {FORMAT_VERSION}")
        rows = data.get("rows")
        if not isinstance(rows, list) or not rows:
            raise StatuteMapError("the statute map has no rows")
        if require_reviewed and not str(data.get("reviewed_by") or "").strip():
            raise StatuteMapError(
                "the statute map has not been reviewed (no reviewed_by). A person must check it against the official "
                "table and rebuild it with --reviewed-by, or load it with require_reviewed=False for testing only."
            )
        parsed = [validate_row(r, i) for i, r in enumerate(rows)]
        metadata = {k: v for k, v in data.items() if k != "rows"}
        return cls(parsed, metadata)

    @classmethod
    def load(cls, path: Union[str, Path], require_reviewed: bool = True) -> "StatuteMap":
        path = Path(path)
        if not path.is_file():
            raise StatuteMapError(
                f"statute map not found at {path}. Build it with scripts/build_statute_map.py from the official table."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StatuteMapError(f"{path} is not valid JSON") from exc
        return cls.from_dict(data, require_reviewed)

    @property
    def reviewed_by(self) -> Optional[str]:
        return self.metadata.get("reviewed_by") or None

    # ------------------------------------------------------------------ lookup

    def counterparts(self, act: str, section: str) -> Tuple[str, List[Counterpart]]:
        """The other law's sections for ``section`` of ``act``.

        Returns ``(status, counterparts)``. ``status`` is "mapped", "no_counterpart" (the table marks the
        section omitted or new) or "not_in_table". A reference with a sub-section such as 438(2) uses the
        table's entry for that sub-section if there is one, otherwise the whole section.
        """
        found: List[MapRow] = []
        for pair, (old_act, new_act) in PAIRS.items():
            if act not in (old_act, new_act):
                continue
            index = self._by_side.get((pair, act), {})
            rows = index.get(section) or index.get(base_section(section)) or []
            found.extend(rows)
        if not found:
            return "not_in_table", []
        out: List[Counterpart] = []
        for row in found:
            old_act, new_act = PAIRS[row.pair]
            other_act, other = (new_act, row.new) if act == old_act else (old_act, row.old)
            if other is not None:
                out.append(Counterpart(other_act, other, row.relationship, row.note))
        unique = list(dict.fromkeys(out))
        return ("mapped", unique) if unique else ("no_counterpart", [])

    def expand(self, query: str, include_act_names: bool = False, max_additions: int = 12) -> Expansion:
        """Appends the other law's number for every named reference in ``query``.

        ``include_act_names`` also appends the other law's full name once. ``max_additions`` bounds how
        many references are appended, so a very broad section cannot swamp the query.
        """
        matches: List[MatchedRef] = []
        additions: List[str] = []
        names: List[str] = []
        for ref in find_references(query):
            status, counterparts = self.counterparts(ref.act, ref.section)
            matches.append(MatchedRef(ref, status, counterparts))
            for item in counterparts:
                phrase = f"Section {item.section} {ACTS[item.act][0]}"
                if phrase not in additions and len(additions) < max_additions:
                    additions.append(phrase)
                if include_act_names and ACTS[item.act][1] not in names:
                    names.append(ACTS[item.act][1])
        if include_act_names and additions:
            additions.extend(names)
        return Expansion(query, matches, additions)

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for row in self.rows:
            counts[row.pair] = counts.get(row.pair, 0) + 1
        return {"rows": len(self.rows), "rows_by_pair": counts, "reviewed_by": self.reviewed_by}


def sha256_of(path: Union[str, Path]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

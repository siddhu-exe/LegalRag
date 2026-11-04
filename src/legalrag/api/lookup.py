"""
GET /lookup: find a case among ALL records in the corpus (plan task C7).

Unlike /query this does not use the retrieval models or the LLM. It searches metadata only (CNR, case
number, title and party names, judge, date, court, bench) and returns the matching records. A search with
no match returns HTTP 200 with status "not_in_our_data": the data has gaps (nothing for Bombay after May
2026 or the Supreme Court after 2024), so no result is ever worded as a statement about whether a case
exists. Off unless LOOKUP_ENABLED=true.

HTTP semantics: 200 found or not in our data; 422 invalid request (no criteria, bad date, text too long);
503 lookup not enabled or its index not available; 500 unexpected failure. Error bodies are sanitized.
"""

from __future__ import annotations

import atexit
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from legalrag.api.config import Settings, get_settings
from legalrag.lookup.index import MAX_LIMIT, MAX_OFFSET, MAX_TEXT_CHARS, CaseLookupIndex, LookupError_, LookupQuery

logger = logging.getLogger(__name__)

router = APIRouter()

LOOKUP_UNAVAILABLE_DETAIL = "Case lookup is not available on this server."
LOOKUP_FAILED_DETAIL = "Case lookup failed."
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class CaseRecord(BaseModel):
    cnr: str
    title: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    judge: Optional[str] = Field(default=None, description="As recorded in the data; null when not recorded.")
    judge_display: str = Field(default="not recorded")
    judgment_date: Optional[str] = Field(default=None, description="Chosen by the corpus date rule; see date_conflict.")
    judgment_year: Optional[int] = None
    court_name: Optional[str] = None
    bench_name: Optional[str] = None
    case_reference: Optional[str] = Field(default=None, description="For example WP/1519/2000, when it could be read from the title.")
    case_type_code: Optional[str] = None
    case_type_name: Optional[str] = None
    outcome: Optional[str] = None
    in_search_index: bool = Field(default=False, description="True if the judgment text is searchable in /query.")
    date_conflict: bool = Field(default=False, description="The dataset date and the date in the CNR disagree.")
    date_source: Optional[str] = None
    source_config: Optional[str] = None
    matched_via_duplicate_cnr: Optional[str] = Field(
        default=None, description="Set when the CNR searched for was merged into this near-identical record."
    )


class CoverageItem(BaseModel):
    court: str
    records: int
    first_date: Optional[str] = None
    last_date: Optional[str] = None


class LookupResponse(BaseModel):
    status: Literal["found", "not_in_our_data"]
    message: str
    total: int
    offset: int
    limit: int
    results: List[CaseRecord] = Field(default_factory=list)
    criteria: Dict[str, Any] = Field(default_factory=dict, description="The search as understood, after cleaning.")
    coverage: List[CoverageItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    disclaimer: str


_cache_lock = threading.Lock()
_cached_index: Optional[CaseLookupIndex] = None
_stub_directory: Optional[tempfile.TemporaryDirectory] = None


def reset_lookup_cache() -> None:
    """Forget the opened index (tests, and after rebuilding the file)."""
    global _cached_index
    with _cache_lock:
        if _cached_index is not None:
            _cached_index.close()
        _cached_index = None


def _stub_index() -> CaseLookupIndex:
    """A tiny in-process index for local_stub mode, so the endpoint can be exercised without artifacts."""
    global _stub_directory
    _stub_directory = tempfile.TemporaryDirectory()
    atexit.register(_stub_directory.cleanup)
    rows = [
        {"cnr": "STUB010000012020_1", "title": "WP/100/2020", "petitioner": "ABC Enterprises", "respondent": "State of Maharashtra",
         "judge": "Stub Judge", "judgment_date": "2020-01-15", "judgment_year": 2020, "court_name": "Bombay High Court",
         "bench_name": "Nagpur", "case_type_code": "WP", "case_type_name": "Writ Petition", "case_number": "100", "case_year": "2020",
         "in_search_index": True, "source_config": "high_courts"},
        {"cnr": "STUB010000022021_1", "title": "CRA/7/2021", "petitioner": "State", "respondent": "XYZ", "judge": None,
         "judgment_date": "2021-03-22", "judgment_year": 2021, "court_name": "Bombay High Court", "bench_name": "Aurangabad",
         "case_type_code": "CRA", "case_type_name": "Criminal Appeal", "case_number": "7", "case_year": "2021",
         "in_search_index": False, "source_config": "high_courts"},
    ]
    return CaseLookupIndex.build(Path(_stub_directory.name) / "lookup.sqlite", rows)


def get_lookup_index(settings: Settings = Depends(get_settings)) -> CaseLookupIndex:
    """The opened index, or HTTP 503 when lookup is off or its file is missing (details are logged, not returned)."""
    global _cached_index
    if not settings.lookup_enabled:
        raise HTTPException(status_code=503, detail=LOOKUP_UNAVAILABLE_DETAIL)
    with _cache_lock:
        if _cached_index is None:
            try:
                _cached_index = _stub_index() if settings.is_local_stub else CaseLookupIndex(settings.lookup_db_path)
            except Exception as exc:  # noqa: BLE001 - never return paths or internals
                logger.error("Case lookup index could not be opened (%s: %s).", type(exc).__name__, exc)
                raise HTTPException(status_code=503, detail=LOOKUP_UNAVAILABLE_DETAIL) from exc
        return _cached_index


@router.get("/lookup", response_model=LookupResponse, tags=["Lookup"])
def lookup(
    cnr: Optional[str] = Query(default=None, max_length=MAX_TEXT_CHARS, description="Case Record Number."),
    case_ref: Optional[str] = Query(default=None, max_length=MAX_TEXT_CHARS, description="For example WP/1519/2000 or 'Writ Petition No. 1519 of 2000'."),
    title: Optional[str] = Query(default=None, max_length=MAX_TEXT_CHARS, description="Words from the title or the party names."),
    party: Optional[str] = Query(default=None, max_length=MAX_TEXT_CHARS, description="Words from a petitioner or respondent name."),
    judge: Optional[str] = Query(default=None, max_length=MAX_TEXT_CHARS, description="Words from the judge's name, as recorded."),
    date: Optional[str] = Query(default=None, pattern=DATE_PATTERN, description="Judgment date, YYYY-MM-DD."),
    date_from: Optional[str] = Query(default=None, pattern=DATE_PATTERN),
    date_to: Optional[str] = Query(default=None, pattern=DATE_PATTERN),
    year: Optional[int] = Query(default=None, ge=1800, le=2200),
    court: Optional[str] = Query(default=None, max_length=MAX_TEXT_CHARS),
    bench: Optional[str] = Query(default=None, max_length=MAX_TEXT_CHARS),
    limit: int = Query(default=20, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0, le=MAX_OFFSET),
    index: CaseLookupIndex = Depends(get_lookup_index),
) -> LookupResponse:
    """Find a case by CNR, case number, names, judge, date, court or bench. At least one criterion is required."""
    query = LookupQuery(cnr=cnr, case_ref=case_ref, title=title, party=party, judge=judge, date=date,
                        date_from=date_from, date_to=date_to, year=year, court=court, bench=bench, limit=limit, offset=offset)
    try:
        result = index.search(query)
    except LookupError_ as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - sanitize unexpected failures
        logger.exception("Case lookup failure (%s).", type(exc).__name__)
        raise HTTPException(status_code=500, detail=LOOKUP_FAILED_DETAIL) from exc
    return LookupResponse(
        status=result.status, message=result.message, total=result.total, offset=result.offset, limit=result.limit,
        results=[CaseRecord(**{k: v for k, v in r.items() if k in CaseRecord.model_fields}) for r in result.results],
        criteria=result.criteria, coverage=[CoverageItem(**c) for c in result.coverage], warnings=result.warnings,
        disclaimer=result.disclaimer,
    )

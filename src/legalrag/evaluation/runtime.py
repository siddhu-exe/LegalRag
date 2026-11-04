"""
Helpers for evaluation scripts that call a paid LLM (ground rule 5 in docs/IMPROVEMENT_PLAN.md):

- print an estimated cost before any call and ask for confirmation,
- save every item as soon as it is done,
- resume after a failure without repeating finished items,
- refuse to resume with different settings (for example a different judge model), because
  that would silently mix two experiments into one result file.

No network access and no LLM client lives here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

CHARS_PER_TOKEN = 4  # rough rule of thumb for English; the estimate says so on screen


def estimate_tokens(text: str) -> int:
    """Rough token count (characters / 4). An estimate, not a tokenizer result."""
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


@dataclass
class CostLine:
    """Estimated usage for one kind of call (for example generation or judging)."""

    label: str
    calls: int
    input_tokens: int
    output_tokens: int
    price_in_per_mtok: Optional[float] = None  # USD per million tokens, supplied by the user
    price_out_per_mtok: Optional[float] = None

    @property
    def usd(self) -> Optional[float]:
        if self.price_in_per_mtok is None or self.price_out_per_mtok is None:
            return None
        return (
            self.input_tokens * self.price_in_per_mtok + self.output_tokens * self.price_out_per_mtok
        ) / 1_000_000.0


def build_cost_line(
    label: str,
    prompts: Iterable[str],
    expected_output_tokens: int,
    price_in_per_mtok: Optional[float] = None,
    price_out_per_mtok: Optional[float] = None,
) -> CostLine:
    prompt_list = list(prompts)
    return CostLine(
        label=label,
        calls=len(prompt_list),
        input_tokens=sum(estimate_tokens(p) for p in prompt_list),
        output_tokens=len(prompt_list) * expected_output_tokens,
        price_in_per_mtok=price_in_per_mtok,
        price_out_per_mtok=price_out_per_mtok,
    )


def format_cost_estimate(lines: Sequence[CostLine]) -> str:
    """A plain-text estimate. Prices are never assumed: without them only tokens are shown."""
    rows = ["Estimated usage (tokens are characters / 4, an estimate):"]
    total_usd: Optional[float] = 0.0
    for line in lines:
        usd = line.usd
        cost = f"${usd:,.2f}" if usd is not None else "price not given"
        rows.append(
            f"  {line.label:<12} {line.calls:>6,} calls  ~{line.input_tokens:>10,} in  "
            f"~{line.output_tokens:>9,} out  {cost}"
        )
        total_usd = None if (usd is None or total_usd is None) else total_usd + usd
    if total_usd is not None and len(lines) > 1:
        rows.append(f"  {'total':<12} ${total_usd:,.2f}")
    if any(line.usd is None for line in lines):
        rows.append(
            "  Pass --*-price-in and --*-price-out (USD per million tokens, from your provider's "
            "price page) to see a dollar estimate."
        )
    return "\n".join(rows)


def confirm_or_abort(
    message: str,
    assume_yes: bool = False,
    input_fn: Callable[[str], str] = input,
    interactive: Optional[bool] = None,
) -> None:
    """Exits unless the user agrees. Non-interactive runs must pass ``--yes`` explicitly."""
    if assume_yes:
        return
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise SystemExit(f"{message}\nNot running interactively: pass --yes to proceed, or --dry-run.")
    if input_fn(f"{message}\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SystemExit("Aborted. Nothing was sent.")


class JsonlCheckpoint:
    """Append-only JSONL results file. Each item is written and flushed as soon as it is done.

    Crash safety:
      * The file is read as bytes, so a last line cut in the middle of a UTF-8 character cannot
        raise. Any line that is not valid UTF-8 and valid JSON is skipped and counted in
        ``skipped_lines``.
      * A last line with no newline is kept only if it is a complete record; a cut-off one is
        dropped. Reading never changes the file. The first ``append`` repairs the tail (removes
        the cut-off bytes, or adds the missing newline) so a new record is never glued onto an
        old partial line.

    If an id appears more than once (for example after ``--retry-errors``), the last record wins.
    """

    def __init__(self, path: Path, id_field: str = "item_id"):
        self.path = Path(path)
        self.id_field = id_field
        self.records: Dict[str, Dict[str, Any]] = {}
        self.skipped_lines = 0
        self._truncate_to: Optional[int] = None  # byte offset where a cut-off last line starts
        self._add_newline = False  # last line is a complete record that lacks its newline
        if self.path.exists():
            self._load(self.path.read_bytes())

    def _parse(self, raw: bytes) -> Optional[Dict[str, Any]]:
        try:
            record = json.loads(raw.decode("utf-8"))
            if not isinstance(record, dict):
                return None
            record[self.id_field]  # noqa: B018 - a record without an id is unusable
            return record
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
            return None

    def _load(self, data: bytes) -> None:
        offset = 0
        pieces = data.split(b"\n")
        tail = pieces.pop()  # whatever follows the last newline (b"" when the file ends cleanly)
        for piece in pieces:
            offset += len(piece) + 1
            if not piece.strip():
                continue
            record = self._parse(piece)
            if record is None:
                self.skipped_lines += 1
            else:
                self.records[str(record[self.id_field])] = record
        if tail.strip():
            record = self._parse(tail)
            if record is None:
                self.skipped_lines += 1
                self._truncate_to = offset
            else:
                self.records[str(record[self.id_field])] = record
                self._add_newline = True
        elif tail:
            self._truncate_to = offset  # stray whitespace after the last newline

    def is_done(self, item_id: str, retry_statuses: Iterable[str] = ()) -> bool:
        record = self.records.get(str(item_id))
        if record is None:
            return False
        return record.get("status") not in set(retry_statuses)

    def _repair_tail(self) -> None:
        if self._truncate_to is None and not self._add_newline:
            return
        with self.path.open("r+b") as handle:
            if self._truncate_to is not None:
                handle.truncate(self._truncate_to)
            else:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._truncate_to = None
        self._add_newline = False

    def append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if self.path.exists():
            self._repair_tail()
        with self.path.open("ab") as handle:
            handle.write(line)  # one write call per record
            handle.flush()
            os.fsync(handle.fileno())
        self.records[str(record[self.id_field])] = dict(record)

    def values(self) -> List[Dict[str, Any]]:
        return list(self.records.values())


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """SHA-256 of a file, streamed so multi-gigabyte index files do not fill memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_run_config(
    path: Path,
    config: Mapping[str, Any],
    keys: Optional[Sequence[str]] = None,
    force: bool = False,
    require_existing: bool = False,
) -> bool:
    """Read-only check of the settings that produced the saved results. Writes nothing.

    Returns True if a saved config existed and matched, False if there is none yet. Stops with
    SystemExit when a setting changed (a different judge, benchmark, retrieval file, artifact or
    prompt would silently mix two experiments), or, with ``require_existing``, when there is no
    saved config at all (results of unknown origin). ``force`` skips the check and returns False.
    """
    path = Path(path)
    if force:
        return False
    if not path.exists():
        if require_existing:
            raise SystemExit(
                f"No saved settings at {path}, so these results cannot be matched to a run. "
                "Re-run without --metrics-only, or use --force-resume if you know where they came from."
            )
        return False
    current = {k: config[k] for k in (keys or config)}
    saved = json.loads(path.read_text(encoding="utf-8"))
    changed = {k: (saved.get(k), current[k]) for k in current if saved.get(k) != current[k]}
    if changed:
        detail = "; ".join(f"{k}: was {old!r}, now {new!r}" for k, (old, new) in changed.items())
        raise SystemExit(
            f"Refusing to use the results in {path.parent}: settings changed ({detail}). "
            "Use a new --out-dir, or --force-resume if you accept mixed results."
        )
    return True


def ensure_run_config(
    path: Path,
    config: Mapping[str, Any],
    keys: Optional[Sequence[str]] = None,
    force: bool = False,
) -> None:
    """Checks the saved settings (see ``check_run_config``), or writes them on the first run.

    The file is written to a temporary name and renamed, so a crash cannot leave half a config.
    """
    path = Path(path)
    if check_run_config(path, config, keys, force):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(config), indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)

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

    If an id appears more than once (for example after ``--retry-errors``), the last record
    wins. A last line cut off by a crash is ignored and counted in ``skipped_lines``.
    """

    def __init__(self, path: Path, id_field: str = "item_id"):
        self.path = Path(path)
        self.id_field = id_field
        self.records: Dict[str, Dict[str, Any]] = {}
        self.skipped_lines = 0
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self.records[str(record[self.id_field])] = record
                    except (json.JSONDecodeError, KeyError, TypeError):
                        self.skipped_lines += 1

    def is_done(self, item_id: str, retry_statuses: Iterable[str] = ()) -> bool:
        record = self.records.get(str(item_id))
        if record is None:
            return False
        return record.get("status") not in set(retry_statuses)

    def append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[str(record[self.id_field])] = dict(record)

    def values(self) -> List[Dict[str, Any]]:
        return list(self.records.values())


def ensure_run_config(
    path: Path,
    config: Mapping[str, Any],
    keys: Optional[Sequence[str]] = None,
    force: bool = False,
) -> None:
    """Writes the run settings next to the results, or checks them when resuming.

    ``keys`` limits the check to settings that change results (models, benchmark, system).
    A mismatch stops the run, so a resume cannot mix judges or generators in one file.
    """
    path = Path(path)
    current = {k: config[k] for k in (keys or config)}
    if path.exists() and not force:
        saved = json.loads(path.read_text(encoding="utf-8"))
        changed = {
            k: (saved.get(k), current[k]) for k in current if saved.get(k) != current[k]
        }
        if changed:
            detail = "; ".join(f"{k}: was {old!r}, now {new!r}" for k, (old, new) in changed.items())
            raise SystemExit(
                f"Refusing to resume {path.parent}: settings changed ({detail}). "
                "Use a new --out-dir, or --force-resume if you accept mixed results."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(config), indent=2, default=str), encoding="utf-8")

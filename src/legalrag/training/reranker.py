"""
Fine-tune a cross-encoder reranker on (question, passage) groups (plan task C10).

Why: the v1 reranker (ms-marco-MiniLM-L-6-v2) was trained on web search, and in the v1 benchmark it raised
Recall@1 but lowered Recall@5 and Recall@10 against fusion alone (docs/EXPERIMENT_LOG.md). A model fine-tuned on
legal question-passage pairs may fix that. Whether it does is for plan task P8, with the paired test in
``scripts/run_retrieval_eval.py``; nothing here claims it works.

Training data (JSONL, one example per line; the questions and the hard negatives are made in plan tasks P6/P7):

    {"query_id": "q1", "query": "...", "cnr": "<judgment the positives come from>",
     "positives": [{"chunk_id": "...", "text": "..."}],
     "negatives": [{"chunk_id": "...", "cnr": "<judgment of this passage>", "text": "..."}]}

Every passage needs a CNR so the data can be split BY JUDGMENT (``splits``). The judgments behind the
evaluation benchmark are forced into the test split, so the benchmark is only ever test data.

Method: for each positive, the model scores the positive and ``negatives_per_positive`` negatives; the loss is
the softmax cross-entropy that the positive ranks first among them (a listwise loss). The validation split picks
the best epoch; the test split is scored once at the end, for the starting model and the tuned model on the
same groups, with a paired bootstrap whose groups are judgments. Plain PyTorch and transformers are used, so
there is no dependence on a sentence-transformers training API.

Limits: the ranking metrics here score a positive among the stored negatives, which is easier than ranking
among everything retrieved; they show direction, not the effect on the cascade. The starting-model comparison is
only meaningful when the base model is already a cross-encoder (the default is).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from legalrag.evaluation.stats import paired_bootstrap
from legalrag.training.splits import SplitData


class ExampleError(ValueError):
    """The training file is malformed. The message lists every problem with its line number."""


@dataclass
class TrainingConfig:
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    epochs: int = 2
    batch_groups: int = 4  # groups per optimizer micro-step; each has 1 + negatives_per_positive passages
    grad_accum: int = 4
    negatives_per_positive: int = 7
    eval_negatives: int = 30
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    max_grad_norm: float = 1.0
    fp16: bool = True  # used only on a CUDA device
    seed: int = 42
    eval_batch_size: int = 32
    max_train_groups: Optional[int] = None  # a quick trial on part of the training data

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_groups", "grad_accum", "negatives_per_positive", "eval_negatives", "max_length", "eval_batch_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.learning_rate <= 0 or not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("learning_rate must be positive and warmup_ratio in [0, 1)")


# --------------------------------------------------------------------------- data


def load_examples(path: Path) -> List[Dict[str, Any]]:
    """Reads and checks the training file. Every problem is reported at once, with its line number."""
    errors: List[str] = []
    examples: List[Dict[str, Any]] = []
    seen: set = set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {number}: not valid JSON")
            continue
        problem = _check_example(raw)
        if problem:
            errors.append(f"line {number}: {problem}")
            continue
        if raw["query_id"] in seen:
            errors.append(f"line {number}: duplicate query_id {raw['query_id']!r}")
            continue
        seen.add(raw["query_id"])
        examples.append({
            "query_id": str(raw["query_id"]), "query": raw["query"].strip(), "cnr": raw["cnr"].strip(),
            "positives": [{"chunk_id": str(p["chunk_id"]), "text": p["text"]} for p in raw["positives"]],
            "negatives": [{"chunk_id": str(n["chunk_id"]), "cnr": n["cnr"].strip(), "text": n["text"]} for n in raw["negatives"]],
        })
    if not examples and not errors:
        errors.append("the file contains no examples")
    if errors:
        shown = errors[:25]
        more = f" (and {len(errors) - 25} more)" if len(errors) > 25 else ""
        raise ExampleError(f"{path}: " + "; ".join(shown) + more)
    return examples


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_example(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return "not an object"
    for key in ("query_id", "query", "cnr"):
        if key == "query_id":
            if raw.get(key) in (None, ""):
                return "missing query_id"
        elif not _nonempty(raw.get(key)):
            return f"missing {key}"
    positives, negatives = raw.get("positives"), raw.get("negatives")
    if not isinstance(positives, list) or not positives:
        return "needs at least one positive"
    if not isinstance(negatives, list) or not negatives:
        return "needs at least one negative"
    for kind, items in (("positive", positives), ("negative", negatives)):
        for item in items:
            if not isinstance(item, dict) or item.get("chunk_id") in (None, "") or not _nonempty(item.get("text")):
                return f"every {kind} needs a chunk_id and non-empty text"
    for item in negatives:
        if not _nonempty(item.get("cnr")):
            return "every negative needs its judgment's cnr, so the split can keep judgments apart"
    for item in positives:
        if item.get("cnr") not in (None, "") and str(item["cnr"]).strip() != raw["cnr"].strip():
            return "positives must come from the example's judgment (their cnr differs)"
    positive_ids = {str(p["chunk_id"]) for p in positives}
    if positive_ids & {str(n["chunk_id"]) for n in negatives}:
        return "a passage is listed as both positive and negative"
    return None


def benchmark_judgments(path: Path) -> List[str]:
    """CNRs behind an evaluation benchmark (gold judgments and source judgments), to be kept as test data only."""
    from legalrag.evaluation.benchmark import load_benchmark

    cnrs = set()
    for item in load_benchmark(path):
        cnrs.update(item["gold_cnrs"])
        if item.get("source_cnr"):
            cnrs.add(str(item["source_cnr"]))
    return sorted(cnrs)


Group = Dict[str, Any]  # {"query": str, "texts": [positive, negatives...], "cnr": str}


def make_groups(examples: Sequence[Mapping[str, Any]], negatives: int, seed: int, shuffle_negatives: bool) -> List[Group]:
    """One group per positive: the positive first, then up to ``negatives`` negatives.

    Training draws a fresh random subset each epoch; evaluation takes the first ones in file order.
    """
    rng = random.Random(seed)
    groups: List[Group] = []
    for example in examples:
        pool = list(example["negatives"])
        for positive in example["positives"]:
            chosen = rng.sample(pool, min(negatives, len(pool))) if shuffle_negatives else pool[:negatives]
            groups.append({"query": example["query"], "texts": [positive["text"]] + [n["text"] for n in chosen], "cnr": example["cnr"]})
    return groups


def ranking_metrics(score_groups: Sequence[Sequence[float]]) -> Dict[str, float]:
    """MRR, Recall@1 and Recall@3 of the positive (index 0) among each group. Ties count against the positive."""
    ranks = positive_ranks(score_groups)
    n = len(ranks)
    if n == 0:
        return {"groups": 0, "mrr": float("nan"), "recall_at_1": float("nan"), "recall_at_3": float("nan")}
    return {
        "groups": n,
        "mrr": sum(1.0 / r for r in ranks) / n,
        "recall_at_1": sum(r <= 1 for r in ranks) / n,
        "recall_at_3": sum(r <= 3 for r in ranks) / n,
    }


def positive_ranks(score_groups: Sequence[Sequence[float]]) -> List[int]:
    return [1 + sum(1 for s in scores[1:] if s >= scores[0]) for scores in score_groups]


# --------------------------------------------------------------------------- torch


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Training needs PyTorch and transformers (pip install torch transformers).") from exc
    return torch


def resolve_device(requested: str = "auto") -> str:
    torch = _torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but no CUDA device is available.")
    return requested


def score_groups(model: Any, tokenizer: Any, groups: Sequence[Group], device: str, max_length: int, batch_size: int) -> List[List[float]]:
    """Scores every (query, passage) pair in the groups. Returns one list of scores per group."""
    torch = _torch()
    pairs = [(g["query"], text) for g in groups for text in g["texts"]]
    scores: List[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer([q for q, _ in batch], [t for _, t in batch], padding=True, truncation="only_second",
                                max_length=max_length, return_tensors="pt", return_token_type_ids=True)
            encoded = {k: v.to(device) for k, v in encoded.items() if k in _model_inputs(model)}
            scores.extend(model(**encoded).logits.reshape(-1).float().cpu().tolist())
    out, cursor = [], 0
    for group in groups:
        out.append(scores[cursor : cursor + len(group["texts"])])
        cursor += len(group["texts"])
    return out


def _model_inputs(model: Any) -> set:
    import inspect

    return set(inspect.signature(model.forward).parameters)


def evaluate(model: Any, tokenizer: Any, groups: Sequence[Group], config: TrainingConfig, device: str) -> Tuple[Dict[str, float], List[List[float]]]:
    scores = score_groups(model, tokenizer, groups, device, config.max_length, config.eval_batch_size)
    return ranking_metrics(scores), scores


def _seed_everything(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load(model_path: str, device: str) -> Tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    return model.to(device), tokenizer


def train(
    data: SplitData,
    out_dir: Path,
    config: TrainingConfig,
    device: str = "auto",
    log: Any = print,
) -> Dict[str, Any]:
    """Fine-tunes and evaluates. Writes ``best/`` (model and tokenizer) and returns the report.

    Splits must already be clean (``assert_no_leakage``); this function does not look at the test split until
    training has finished and the best epoch has been chosen on the validation split.
    """
    torch = _torch()
    from transformers import get_linear_schedule_with_warmup

    device = resolve_device(device)
    _seed_everything(config.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load(config.base_model, device)

    train_examples = list(data.train)
    if config.max_train_groups is not None:
        train_examples = train_examples[: config.max_train_groups]
    if not train_examples:
        raise ValueError("There are no training examples after the split.")
    val_groups = make_groups(data.val, config.eval_negatives, config.seed, shuffle_negatives=False)
    test_groups = make_groups(data.test, config.eval_negatives, config.seed, shuffle_negatives=False)

    started = time.perf_counter()
    baseline = {"val": evaluate(model, tokenizer, val_groups, config, device)[0] if val_groups else None}
    baseline_test_metrics, baseline_test_scores = evaluate(model, tokenizer, test_groups, config, device) if test_groups else (None, [])
    baseline["test"] = baseline_test_metrics

    steps_per_epoch = max(1, -(-len(make_groups(train_examples, config.negatives_per_positive, config.seed, True)) // (config.batch_groups * config.grad_accum)))
    total_steps = steps_per_epoch * config.epochs
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.warmup_ratio), total_steps)
    use_amp = device.startswith("cuda") and config.fp16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    history: List[Dict[str, Any]] = []
    best_mrr, best_epoch = float("-inf"), 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        groups = make_groups(train_examples, config.negatives_per_positive, config.seed + epoch, shuffle_negatives=True)
        random.Random(config.seed * 1000 + epoch).shuffle(groups)
        losses: List[float] = []
        optimizer.zero_grad()
        for step, start in enumerate(range(0, len(groups), config.batch_groups), start=1):
            batch = groups[start : start + config.batch_groups]
            pairs = [(g["query"], t) for g in batch for t in g["texts"]]
            encoded = tokenizer([q for q, _ in pairs], [t for _, t in pairs], padding=True, truncation="only_second",
                                max_length=config.max_length, return_tensors="pt", return_token_type_ids=True)
            encoded = {k: v.to(device) for k, v in encoded.items() if k in _model_inputs(model)}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**encoded).logits.reshape(-1).float()
            loss = torch.stack([
                torch.nn.functional.cross_entropy(chunk.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
                for chunk in torch.split(logits, [len(g["texts"]) for g in batch])
            ]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Training loss became {loss.item()} at epoch {epoch}, step {step}; lower the learning rate.")
            losses.append(float(loss.item()))
            (scaler.scale(loss / config.grad_accum).backward() if scaler else (loss / config.grad_accum).backward())
            if step % config.grad_accum == 0 or start + config.batch_groups >= len(groups):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                (scaler.step(optimizer), scaler.update()) if scaler else optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        entry: Dict[str, Any] = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "groups": len(groups)}
        if val_groups:
            entry["val"] = evaluate(model, tokenizer, val_groups, config, device)[0]
            selection = entry["val"]["mrr"]
        else:
            selection = float(epoch)  # no validation split: keep the last epoch
        history.append(entry)
        log(f"epoch {epoch}: train loss {entry['train_loss']:.4f}" + (f", val MRR {entry['val']['mrr']:.4f}" if val_groups else ""))
        if selection > best_mrr:
            best_mrr, best_epoch = selection, epoch
            model.save_pretrained(str(out_dir / "best"))
            tokenizer.save_pretrained(str(out_dir / "best"))

    tuned, tokenizer_best = _load(str(out_dir / "best"), device)
    tuned_test_metrics, tuned_test_scores = evaluate(tuned, tokenizer_best, test_groups, config, device) if test_groups else (None, [])
    comparison = None
    if test_groups:
        base_rr = [1.0 / r for r in positive_ranks(baseline_test_scores)]
        tuned_rr = [1.0 / r for r in positive_ranks(tuned_test_scores)]
        result = paired_bootstrap(tuned_rr, base_rr, groups=[g["cnr"] for g in test_groups], seed=config.seed)
        comparison = {"metric": "reciprocal rank on held-out judgments, tuned minus starting model", **result.to_dict(),
                      "judgments": len({g["cnr"] for g in test_groups})}
    return {
        "config": asdict(config), "device": device, "best_epoch": best_epoch, "history": history,
        "baseline": baseline, "tuned": {"test": tuned_test_metrics},
        "test_comparison": comparison, "seconds": round(time.perf_counter() - started, 1),
        "best_model_dir": str(out_dir / "best"),
        "notes": ["Ranking metrics score the positive among the stored negatives only; they show direction, "
                  "not the effect on the retrieval cascade (plan task P8 measures that).",
                  "The test split was scored once, after the best epoch was chosen on the validation split."],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

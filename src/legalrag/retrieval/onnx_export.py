"""
Export a Hugging Face encoder or cross-encoder to the ONNX directory ``onnx_models`` reads
(plan tasks C4 and C10).

This is the only place that needs PyTorch and ``transformers``, and only when you run it (on your PC,
or wherever the model lives); serving does not. Every export is verified before it is reported as
done: the same texts go through the original model and through ``onnxruntime`` (using the saved
``tokenizer.json``), and the outputs must agree within ``tolerance`` or ``ExportVerificationError``
is raised and the directory should not be used.

Two traps this module avoids on purpose: the model is put in ``eval()`` mode and run under
``no_grad()`` for both the export and the reference (a model left in training mode applies dropout
and looks like a large mismatch), and the TorchScript exporter is used when the installed PyTorch
offers a choice, because it was the path checked here.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from legalrag.retrieval.onnx_models import CONFIG_FILE, MODEL_FILE, TOKENIZER_FILE, OnnxEmbedder, OnnxReranker

DEFAULT_OPSET = 17
DEFAULT_TOLERANCE = 1e-4
SAMPLE_TEXTS = [
    "Anticipatory bail under Section 438 of the Code of Criminal Procedure.",
    "The writ petition is dismissed.",
    "Whether the arbitral award suffers from patent illegality appearing on the face of the award, "
    "and if so whether this Court may interfere under Section 34 of the Arbitration and Conciliation Act.",
]


def _worst_difference(got: np.ndarray, expected: np.ndarray) -> float:
    """Largest absolute difference, or relative difference for values above 1 (raw logits can be large)."""
    return float((np.abs(got - expected) / np.maximum(1.0, np.abs(expected))).max())


class ExportVerificationError(RuntimeError):
    """The ONNX model does not reproduce the original model's outputs."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Exporting to ONNX needs PyTorch and transformers (pip install torch transformers).") from exc
    return torch


def _export_onnx(module: Any, example: Tuple[Any, ...], path: Path, input_names: List[str], output_name: str, opset: int) -> None:
    torch = _torch()
    dynamic = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic[output_name] = {0: "batch", 1: "sequence"} if output_name == "last_hidden_state" else {0: "batch"}
    kwargs: Dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # the TorchScript exporter: the path whose output was checked against PyTorch
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module, example, str(path), input_names=input_names, output_names=[output_name],
            dynamic_axes=dynamic, opset_version=opset, **kwargs,
        )


def _load_hf(model_path: str, task: str, tokenizer_path: Optional[str]) -> Tuple[Any, Any]:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    if getattr(tokenizer, "backend_tokenizer", None) is None:
        raise ValueError("A fast tokenizer is required (it is saved as tokenizer.json for the runtime).")
    loader = AutoModel if task == "embedder" else AutoModelForSequenceClassification
    return loader.from_pretrained(model_path).eval(), tokenizer


def _model_inputs(model: Any) -> List[str]:
    accepted = inspect.signature(model.forward).parameters
    return [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in accepted]


def _wrapper(model: Any, names: List[str], attribute: str) -> Any:
    torch = _torch()

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, *tensors: Any) -> Any:
            return getattr(self.model(**dict(zip(names, tensors))), attribute)

    return Wrapper().eval()


def _example(tokenizer: Any, texts: Sequence[Any], names: List[str], max_length: int) -> Tuple[Any, ...]:
    torch = _torch()
    # Ask for token_type_ids explicitly: some tokenizers leave them out of their default outputs, and the
    # runtime always supplies the real segment ids, so the reference must use the same ones.
    options = dict(padding=True, truncation=True, max_length=max_length, return_tensors="pt", return_token_type_ids=True)
    if texts and isinstance(texts[0], tuple):
        encoded = tokenizer([t[0] for t in texts], [t[1] for t in texts], **options)
    else:
        encoded = tokenizer(list(texts), **options)
    return tuple(encoded[n] if n in encoded else torch.zeros_like(encoded["input_ids"]) for n in names)


def _write_common(out_dir: Path, tokenizer: Any, config: Dict[str, Any]) -> None:
    tokenizer.backend_tokenizer.save(str(out_dir / TOKENIZER_FILE))
    config["pad_token"] = tokenizer.pad_token or "[PAD]"
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")


def export_embedder(
    model_path: str,
    out_dir: Union[str, Path],
    pooling: str = "cls",
    normalize: bool = True,
    max_length: int = 512,
    tokenizer_path: Optional[str] = None,
    opset: int = DEFAULT_OPSET,
    verify_texts: Optional[Sequence[str]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, float]:
    """Exports an encoder (BGE uses ``pooling="cls"``) and verifies it. Returns the verification numbers."""
    if pooling not in ("cls", "mean"):
        raise ValueError("pooling must be 'cls' or 'mean'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_hf(model_path, "embedder", tokenizer_path)
    names = _model_inputs(model)
    example = _example(tokenizer, list(verify_texts or SAMPLE_TEXTS), names, max_length)
    _export_onnx(_wrapper(model, names, "last_hidden_state"), example, out_dir / MODEL_FILE, names, "last_hidden_state", opset)
    _write_common(out_dir, tokenizer, {
        "kind": "embedder", "source_model": str(model_path), "max_length": max_length,
        "dim": int(model.config.hidden_size), "pooling": pooling, "normalize": normalize,
        "output": "last_hidden_state", "opset": opset,
    })
    return verify_embedder(out_dir, model, tokenizer, list(verify_texts or SAMPLE_TEXTS), tolerance)


def verify_embedder(out_dir: Union[str, Path], model: Any, tokenizer: Any, texts: Sequence[str], tolerance: float = DEFAULT_TOLERANCE) -> Dict[str, float]:
    torch = _torch()
    onnx_model = OnnxEmbedder.load(out_dir)
    config = onnx_model.config
    names = _model_inputs(model)
    with torch.no_grad():
        batch = _example(tokenizer, texts, names, int(config["max_length"]))
        hidden = model(**dict(zip(names, batch))).last_hidden_state
        mask = batch[names.index("attention_mask")] if "attention_mask" in names else torch.ones(hidden.shape[:2])
        pooled = hidden[:, 0] if config["pooling"] == "cls" else (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if config["normalize"]:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
    got = onnx_model.encode(list(texts), batch_size=2, normalize_embeddings=bool(config["normalize"]))
    worst = _worst_difference(got, pooled.numpy())
    if not worst <= tolerance:
        raise ExportVerificationError(f"ONNX embedder differs from the original by {worst:.2e} (tolerance {tolerance:.0e}).")
    return {"max_abs_difference": worst, "texts_checked": len(texts), "tolerance": tolerance}


def export_reranker(
    model_path: str,
    out_dir: Union[str, Path],
    max_length: int = 512,
    activation: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    opset: int = DEFAULT_OPSET,
    verify_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, float]:
    """Exports a one-score cross-encoder and verifies it.

    ``activation`` is "sigmoid" or "none"; by default "sigmoid", which is what the v1
    ``sentence-transformers`` CrossEncoder applies to a one-label model. Ranking does not depend on it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_hf(model_path, "reranker", tokenizer_path)
    if int(model.config.num_labels) != 1:
        raise ValueError(f"A reranker needs one output score, but the model has {model.config.num_labels} labels.")
    activation = activation or "sigmoid"
    if activation not in ("sigmoid", "none"):
        raise ValueError("activation must be 'sigmoid' or 'none'")
    pairs = list(verify_pairs or [(SAMPLE_TEXTS[0], SAMPLE_TEXTS[2]), (SAMPLE_TEXTS[1], SAMPLE_TEXTS[0]), (SAMPLE_TEXTS[2], SAMPLE_TEXTS[1])])
    names = _model_inputs(model)
    example = _example(tokenizer, pairs, names, max_length)
    _export_onnx(_wrapper(model, names, "logits"), example, out_dir / MODEL_FILE, names, "logits", opset)
    _write_common(out_dir, tokenizer, {
        "kind": "reranker", "source_model": str(model_path), "max_length": max_length,
        "activation": activation, "output": "logits", "opset": opset,
    })
    return verify_reranker(out_dir, model, tokenizer, pairs, tolerance)


def verify_reranker(out_dir: Union[str, Path], model: Any, tokenizer: Any, pairs: Sequence[Tuple[str, str]], tolerance: float = DEFAULT_TOLERANCE) -> Dict[str, float]:
    torch = _torch()
    onnx_model = OnnxReranker.load(out_dir)
    names = _model_inputs(model)
    with torch.no_grad():
        logits = model(**dict(zip(names, _example(tokenizer, pairs, names, onnx_model.max_length)))).logits[:, 0]
        expected = torch.sigmoid(logits) if onnx_model.config["activation"] == "sigmoid" else logits
    got = onnx_model.predict(list(pairs), batch_size=2)
    worst = _worst_difference(got, expected.numpy())
    if not worst <= tolerance:
        raise ExportVerificationError(f"ONNX reranker differs from the original by {worst:.2e} (tolerance {tolerance:.0e}).")
    return {"max_abs_difference": worst, "pairs_checked": len(pairs), "tolerance": tolerance}

"""
Exports an embedding model or a cross-encoder to the ONNX directory the serving code reads (plan task C4).

Needs PyTorch and transformers, so run it where the model lives (your PC). Serving needs neither.
The export is verified against the original model before it reports success; a mismatch raises an error.

    python scripts/export_onnx.py embedder --model BAAI/bge-base-en-v1.5 --out-dir artifacts/v2/onnx/embedder --pooling cls
    python scripts/export_onnx.py reranker --model cross-encoder/ms-marco-MiniLM-L-6-v2 --out-dir artifacts/v2/onnx/reranker
    python scripts/export_onnx.py reranker --model artifacts/v2/reranker/best --out-dir artifacts/v2/onnx/reranker   # a fine-tuned model

Then set EMBEDDER_BACKEND=onnx and/or RERANKER_BACKEND=onnx with ARTIFACT_DIR=artifacts/v2.
The verification covers a few sample texts; plan task P5 must also compare retrieval quality and speed
on the real models. A downloaded model name needs network access when you run this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    from legalrag.retrieval.onnx_export import DEFAULT_OPSET, DEFAULT_TOLERANCE, export_embedder, export_reranker
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.retrieval.onnx_export import DEFAULT_OPSET, DEFAULT_TOLERANCE, export_embedder, export_reranker  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kinds = parser.add_subparsers(dest="kind", required=True)
    for name in ("embedder", "reranker"):
        sub = kinds.add_parser(name)
        sub.add_argument("--model", required=True, help="a Hugging Face model name or a local folder")
        sub.add_argument("--out-dir", type=Path, required=True)
        sub.add_argument("--tokenizer", default=None, help="tokenizer name or folder if different from --model")
        sub.add_argument("--max-length", type=int, default=512)
        sub.add_argument("--opset", type=int, default=DEFAULT_OPSET)
        sub.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="largest allowed difference from the original model")
        if name == "embedder":
            sub.add_argument("--pooling", choices=("cls", "mean"), default="cls", help="BGE uses cls")
            sub.add_argument("--no-normalize", action="store_true")
        else:
            sub.add_argument("--activation", choices=("sigmoid", "none"), default=None, help="default sigmoid, as sentence-transformers applies")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.kind == "embedder":
        report = export_embedder(args.model, args.out_dir, args.pooling, not args.no_normalize, args.max_length,
                                 args.tokenizer, args.opset, tolerance=args.tolerance)
    else:
        report = export_reranker(args.model, args.out_dir, args.max_length, args.activation, args.tokenizer,
                                 args.opset, tolerance=args.tolerance)
    print(json.dumps({"exported_to": str(args.out_dir), "verification": report}, indent=2))


if __name__ == "__main__":
    main()

"""
Measures RAM and latency of the v1 retrieval pipeline, component by component.

Loads each part of the production stack in serving order, recording process memory after each
step, then runs sample queries through the cascade. Generation is skipped because the LLM runs
remotely. Writes results/ram_baseline.json and counts chunks per court in legal_chunks.parquet.

Each component's figure is the memory added at its step, so it depends on load order. Run in a
fresh process and close other heavy applications first.

Requires psutil, which is not a project dependency:
    pip install psutil
    PYTHONPATH=src python scripts/measure_ram.py --artifact-dir artifacts
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil

SAMPLE_QUERIES = [
    "When can a court quash proceedings under Section 482 CrPC?",
    "What are the ingredients of an offence under Section 138 of the Negotiable Instruments Act?",
    "Can bail be granted under Section 439 CrPC when the charge sheet has been filed?",
    "What is the limitation period for filing a suit for specific performance?",
    "When can a writ petition be entertained despite an alternative remedy?",
    "How is compensation calculated in a motor accident claim for a deceased earning member?",
]
RETRIEVAL_TOP_K = 50
RERANK_TOP_K = 5
GIB = 1024**3


class MemoryLog:
    """Records process memory and elapsed time for each named step."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.steps: List[Dict[str, Any]] = []
        self.previous = self.rss()

    def rss(self) -> int:
        return self.process.memory_info().rss

    def peak(self) -> int:
        info = self.process.memory_info()
        # Windows reports the peak directly. Elsewhere, use the highest RSS we recorded.
        recorded = max((step["rss_bytes"] for step in self.steps), default=info.rss)
        return getattr(info, "peak_wset", recorded)

    def record(self, name: str, started: float) -> None:
        current = self.rss()
        self.steps.append(
            {
                "step": name,
                "seconds": round(time.perf_counter() - started, 2),
                "added_gib": round((current - self.previous) / GIB, 3),
                "rss_gib": round(current / GIB, 3),
                "rss_bytes": current,
            }
        )
        print(f"{name:<34} +{(current - self.previous) / GIB:7.3f} GiB  total {current / GIB:7.3f} GiB")
        self.previous = current


def percentile(values: List[float], fraction: float) -> float:
    """Returns a nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure v1 pipeline RAM and latency.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results/ram_baseline.json"))
    args = parser.parse_args()

    log = MemoryLog()

    started = time.perf_counter()
    import pandas as pd

    log.record("import pandas", started)

    started = time.perf_counter()
    from legalrag.retrieval.bm25 import BM25Retriever
    from legalrag.retrieval.dense import DenseRetriever
    from legalrag.retrieval.fusion import reciprocal_rank_fusion
    from legalrag.retrieval.reranker import CrossEncoderReranker

    log.record("import torch, faiss, retrievers", started)

    started = time.perf_counter()
    chunks = pd.read_parquet(args.artifact_dir / "legal_chunks.parquet")
    log.record("chunk text (pandas)", started)

    started = time.perf_counter()
    chunk_ids = chunks["chunk_id"].tolist()
    log.record("chunk ID list", started)

    started = time.perf_counter()
    bm25 = BM25Retriever.load(args.artifact_dir / "bm25.pkl")
    log.record("BM25 index", started)

    started = time.perf_counter()
    dense = DenseRetriever.load_index(args.artifact_dir / "dense.index", chunk_ids=chunk_ids)
    log.record("FAISS flat index", started)

    started = time.perf_counter()
    dense.warmup()
    log.record("embedding model", started)

    started = time.perf_counter()
    reranker = CrossEncoderReranker()
    reranker.warmup()
    log.record("reranker model", started)

    text_by_id = dict(zip(chunks["chunk_id"], chunks["text"]))
    latencies: Dict[str, List[float]] = {"bm25": [], "dense": [], "rrf": [], "rerank": [], "total": []}

    started = time.perf_counter()
    for query in SAMPLE_QUERIES:
        t0 = time.perf_counter()
        bm25_hits = bm25.search(query, top_k=RETRIEVAL_TOP_K)
        t1 = time.perf_counter()
        dense_hits = dense.search(query, top_k=RETRIEVAL_TOP_K)
        t2 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in bm25_hits], [cid for cid, _ in dense_hits]], top_n=RETRIEVAL_TOP_K
        )
        t3 = time.perf_counter()
        reranker.rerank(query, [(cid, text_by_id[cid]) for cid, _ in fused], top_k=RERANK_TOP_K)
        t4 = time.perf_counter()
        for name, value in zip(latencies, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
            latencies[name].append(value * 1000)
    log.record(f"{len(SAMPLE_QUERIES)} queries", started)

    prefixes = chunks["cnr"].str[:4].value_counts()
    judgments = chunks.groupby(chunks["cnr"].str[:4])["cnr"].nunique()

    result = {
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "total_ram_gib": round(psutil.virtual_memory().total / GIB, 1),
            "logical_cpus": psutil.cpu_count(),
        },
        "note": "added_gib depends on load order; generation runs remotely and is excluded.",
        "peak_gib": round(log.peak() / GIB, 3),
        "steps": [{k: v for k, v in step.items() if k != "rss_bytes"} for step in log.steps],
        "latency_ms": {
            name: {
                "mean": round(statistics.mean(values), 1),
                "median": round(statistics.median(values), 1),
                "worst": round(percentile(values, 0.95), 1),
            }
            for name, values in latencies.items()
        },
        "corpus": {
            "chunks": int(len(chunks)),
            "judgments": int(chunks["cnr"].nunique()),
            "bombay_chunks": int(prefixes.get("HCBM", 0)),
            "bombay_judgments": int(judgments.get("HCBM", 0)),
            "judgments_by_cnr_prefix": {key: int(value) for key, value in judgments.items()},
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Peak {result['peak_gib']} GiB. Wrote {args.output}")


if __name__ == "__main__":
    main()

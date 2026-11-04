"""
Builds the low-memory v2 retrieval parts (plan task C4). Run on your PC.

Everything is written under a directory with a ``v2`` part (default ``artifacts/v2``); v1 artifacts are
never touched. Point the API or the evaluation at the result with ARTIFACT_DIR=artifacts/v2 and the flags
BM25_BACKEND=bm25s, CHUNK_STORE=sqlite, EMBEDDER_BACKEND=onnx, RERANKER_BACKEND=onnx, DENSE_NPROBE=<n>.

Input is a chunk parquet with a ``chunk_id`` column and a ``text`` column (v1's legal_chunks.parquet has
this shape; the v2 chunk table must keep the same rows in the same order as the embeddings).

Steps, BM25 first because it is the largest part of the v1 RAM baseline (results/ram_baseline.json):

    bm25s       <out>/bm25s/               memory-mappable BM25 index (bm25s)
    chunks      <out>/chunks/chunks.sqlite chunk text on disk, plus chunk_ids.npy
    faiss       <out>/dense.index          IVF-SQ8 or OPQ+IVF+PQ index from embedding shards (.npy)
    faiss-recall                           recall of an approximate index against exact search on a subset

Examples:
    python scripts/build_retrieval_indexes_v2.py bm25s --chunks artifacts/v2/chunks.parquet --max-chunks 50000
    python scripts/build_retrieval_indexes_v2.py bm25s --chunks artifacts/v2/chunks.parquet
    python scripts/build_retrieval_indexes_v2.py chunks --chunks artifacts/v2/chunks.parquet
    python scripts/build_retrieval_indexes_v2.py faiss --embeddings "artifacts/v2/emb/*.npy" --kind ivf_sq8
    python scripts/build_retrieval_indexes_v2.py faiss --embeddings "artifacts/v2/emb/*.npy" --kind opq_pq --pq-m 96
    python scripts/build_retrieval_indexes_v2.py faiss-recall --embeddings "artifacts/v2/emb/*.npy" \\
        --kind ivf_sq8 --subset-rows 200000 --nprobe 1 4 16 64

Build memory for bm25s grows with the number of tokens in the corpus. Try --max-chunks first and
measure peak memory on your machine before the full corpus; no figure is claimed here.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

try:
    from legalrag.retrieval.bm25s_backend import Bm25sRetriever
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.retrieval.bm25s_backend import Bm25sRetriever  # noqa: E402

from legalrag.retrieval.chunk_store import SqliteChunkStore  # noqa: E402
from legalrag.retrieval.faiss_builders import (  # noqa: E402
    EmbeddingShards,
    IndexSpec,
    build_index,
    recall_against_exact,
    suggest_nlist,
    write_index,
)

DEFAULT_OUT = Path("artifacts/v2")


def assert_v2_dir(path: Path) -> None:
    """Outputs must live under a ``v2`` directory so v1 artifacts cannot be overwritten."""
    if "v2" not in Path(path).resolve().parts and "v2" not in Path(path).parts:
        raise SystemExit(f"Refusing to write to {path}: the output directory must contain a 'v2' part (for example artifacts/v2).")


def iter_chunk_rows(parquet_path: Path, columns: Sequence[str], batch_rows: int = 20_000, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(parquet_path))
    names = set(parquet.schema_arrow.names)
    missing = [c for c in columns if c not in names]
    if missing:
        raise SystemExit(f"{parquet_path} lacks column(s) {', '.join(missing)}. Columns found: {', '.join(parquet.schema_arrow.names)}.")
    seen = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=list(columns)):
        for row in batch.to_pylist():
            if limit is not None and seen >= limit:
                return
            seen += 1
            yield row


def build_bm25s(chunks: Path, out_dir: Path, k1: float, b: float, max_chunks: Optional[int], overwrite: bool) -> Dict[str, Any]:
    assert_v2_dir(out_dir)
    target = out_dir / "bm25s"
    if target.exists() and not overwrite:
        raise SystemExit(f"{target} already exists. Use --overwrite or a new --out-dir.")
    started = time.perf_counter()
    ids: List[str] = []
    texts: List[str] = []
    for row in iter_chunk_rows(chunks, ["chunk_id", "text"], limit=max_chunks):
        ids.append(str(row["chunk_id"]))
        texts.append(row["text"] or "")
    retriever = Bm25sRetriever.build(ids, texts, k1=k1, b=b)
    del texts
    staging = out_dir / ".bm25s_staging"
    import shutil

    shutil.rmtree(staging, ignore_errors=True)
    retriever.save(staging)
    if target.exists():
        shutil.rmtree(target)
    staging.replace(target)
    report = {"chunks": len(ids), "partial": max_chunks is not None, "seconds": round(time.perf_counter() - started, 1), "path": str(target)}
    print(json.dumps(report))
    return report


def build_chunk_store(chunks: Path, out_dir: Path, max_chunks: Optional[int], overwrite: bool) -> Dict[str, Any]:
    assert_v2_dir(out_dir)
    store_path = out_dir / "chunks" / "chunks.sqlite"
    if store_path.exists() and not overwrite:
        raise SystemExit(f"{store_path} already exists. Use --overwrite or a new --out-dir.")
    started = time.perf_counter()
    columns = ["chunk_id", "cnr", "text", "court_code", "decision_date", "title"]
    import pyarrow.parquet as pq

    present = set(pq.ParquetFile(str(chunks)).schema_arrow.names)
    store = SqliteChunkStore.build(store_path, iter_chunk_rows(chunks, [c for c in columns if c in present], limit=max_chunks))
    report = {"chunks": len(store), "partial": max_chunks is not None, "seconds": round(time.perf_counter() - started, 1), "path": str(store_path)}
    store.close()
    print(json.dumps(report))
    return report


def expand_shards(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        paths.extend(sorted(p.glob("*.npy")) if p.is_dir() else [Path(x) for x in sorted(glob.glob(pattern))])
    if not paths:
        raise SystemExit(f"No embedding files matched {list(patterns)}.")
    return paths


def make_spec(args: argparse.Namespace, shards: EmbeddingShards, rows: int) -> IndexSpec:
    nlist = args.nlist or suggest_nlist(rows)
    return IndexSpec(args.kind, shards.dim, nlist, args.pq_m, args.pq_bits)


def build_faiss(args: argparse.Namespace) -> Dict[str, Any]:
    assert_v2_dir(args.out_dir)
    target = args.out_dir / "dense.index"
    if target.exists() and not args.overwrite:
        raise SystemExit(f"{target} already exists. Use --overwrite or a new --out-dir.")
    shards = EmbeddingShards(expand_shards(args.embeddings))
    if args.chunk_store:
        expected = len(SqliteChunkStore(args.chunk_store))
        if expected != shards.total:
            raise SystemExit(f"The embeddings have {shards.total} rows but the chunk store has {expected}; they must match one to one, in order.")
    spec = make_spec(args, shards, shards.total)
    started = time.perf_counter()
    index, metadata = build_index(shards, spec, train_size=args.train_size, seed=args.seed, nprobe=args.nprobe[0],
                                  extra_metadata={"embedding_model": args.embedding_model, "query_instruction": args.query_instruction})
    write_index(index, target, metadata)
    report = {**metadata, "seconds": round(time.perf_counter() - started, 1), "path": str(target)}
    print(json.dumps(report, indent=2))
    return report


def run_faiss_recall(args: argparse.Namespace) -> List[Dict[str, float]]:
    shards = EmbeddingShards(expand_shards(args.embeddings))
    rows = min(args.subset_rows, shards.total)
    vectors = shards.sample(rows, seed=args.seed)  # a random subset, in corpus order
    spec = IndexSpec(args.kind, shards.dim, args.nlist or suggest_nlist(rows), args.pq_m, args.pq_bits)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        subset = Path(tmp) / "subset.npy"
        np.save(subset, vectors)
        index, _ = build_index(EmbeddingShards([subset]), spec, train_size=args.train_size, seed=args.seed, nprobe=args.nprobe[0])
    query_rows = np.random.default_rng(args.seed + 1).choice(rows, size=min(args.queries, rows), replace=False)
    results = recall_against_exact(index, vectors, vectors[query_rows], k=args.k, nprobe_values=args.nprobe)
    print(f"{spec.factory} on {rows} vectors, {len(query_rows)} queries, recall of the exact top-{args.k}:")
    for r in results:
        print(f"  nprobe {int(r['nprobe']):>4}: {r['recall_at_k']:.3f}")
    print("This measures the index against exact search, not retrieval quality on the benchmark.")
    return results


def add_faiss_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embeddings", nargs="+", required=True, help="float32 .npy shards (files, folders or globs), in chunk order")
    parser.add_argument("--kind", choices=("ivf_sq8", "opq_pq"), required=True)
    parser.add_argument("--nlist", type=int, default=None, help="inverted lists (default: about 4*sqrt(rows); not tuned)")
    parser.add_argument("--pq-m", type=int, default=None, help="PQ sub-vectors for opq_pq; must divide the embedding dimension")
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=None, help="training sample (default: 39 per list)")
    parser.add_argument("--nprobe", type=int, nargs="+", default=[16], help="lists visited per search (first value is stored)")
    parser.add_argument("--seed", type=int, default=42)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    bm25 = commands.add_parser("bm25s", help="build the memory-mappable BM25 index")
    bm25.add_argument("--chunks", type=Path, required=True)
    bm25.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    bm25.add_argument("--k1", type=float, default=1.5)
    bm25.add_argument("--b", type=float, default=0.75)
    bm25.add_argument("--max-chunks", type=int, default=None, help="index only the first N chunks (a trial; not a usable index)")
    bm25.add_argument("--overwrite", action="store_true")

    store = commands.add_parser("chunks", help="build the SQLite chunk store")
    store.add_argument("--chunks", type=Path, required=True)
    store.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    store.add_argument("--max-chunks", type=int, default=None)
    store.add_argument("--overwrite", action="store_true")

    index = commands.add_parser("faiss", help="build an approximate FAISS index from embedding shards")
    add_faiss_options(index)
    index.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    index.add_argument("--chunk-store", type=Path, default=None, help="check the row count against this store")
    index.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    index.add_argument("--query-instruction", default="Represent this sentence for searching relevant passages: ")
    index.add_argument("--overwrite", action="store_true")

    recall = commands.add_parser("faiss-recall", help="compare an approximate index with exact search on a subset")
    add_faiss_options(recall)
    recall.add_argument("--subset-rows", type=int, default=200_000)
    recall.add_argument("--queries", type=int, default=1000)
    recall.add_argument("--k", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "bm25s":
        build_bm25s(args.chunks, args.out_dir, args.k1, args.b, args.max_chunks, args.overwrite)
    elif args.command == "chunks":
        build_chunk_store(args.chunks, args.out_dir, args.max_chunks, args.overwrite)
    elif args.command == "faiss":
        build_faiss(args)
    else:
        run_faiss_recall(args)


if __name__ == "__main__":
    main()

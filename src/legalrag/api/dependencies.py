"""
Dependency injection and pipeline component loading layer for the LegalRAG FastAPI service.

Manages singleton lifecycle of heavy retrieval indexes, rerankers, and LLM clients
across dual runtime environments:
- 'local_stub': Lightweight deterministic stub components for 6GB dev laptops and test harnesses.
- 'production': Full production components loaded from verified disk artifacts and external APIs.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd

from legalrag.api.config import Settings, get_settings
from legalrag.retrieval.bm25 import BM25Retriever
from legalrag.retrieval.dense import DenseRetriever
from legalrag.retrieval.reranker import CrossEncoderReranker
from legalrag.retrieval.bm25s_backend import Bm25sRetriever
from legalrag.retrieval.chunk_store import DataFrameChunkSource
from legalrag.retrieval.onnx_models import OnnxReranker
from legalrag.generation.client import GenerationResult, LegalGenerationClient

logger = logging.getLogger(__name__)


class PipelineInitializationError(RuntimeError):
    """
    Raised when the pipeline required for inference cannot be initialized.

    In production this includes missing/unavailable artifacts or invalid generation
    configuration. Callers must treat this as a non-ready state (HTTP 503) and must
    never fall back to stub components.
    """


# ---------------------------------------------------------------------------
# Stub Component Implementations for 'local_stub' Mode
# ---------------------------------------------------------------------------

class StubBM25Retriever:
    """Lightweight deterministic BM25 retriever stub for local development."""

    def __init__(self, chunk_ids: Optional[List[str]] = None):
        self.chunk_ids = chunk_ids or [
            "stub_chunk_0",
            "stub_chunk_1",
            "stub_chunk_2",
            "stub_chunk_3",
            "stub_chunk_4",
        ]

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Returns deterministic scored chunk IDs."""
        return [
            (cid, round(1.0 / (idx + 1), 4))
            for idx, cid in enumerate(self.chunk_ids[:top_k])
        ]


class StubDenseRetriever:
    """Lightweight deterministic Dense vector retriever stub for local development."""

    def __init__(self, chunk_ids: Optional[List[str]] = None):
        # Permuted order to simulate realistic rank fusion across lexical & semantic stages
        if chunk_ids:
            # Shift order slightly: chunk_1, chunk_0, chunk_2, ...
            self.chunk_ids = [chunk_ids[i % len(chunk_ids)] for i in [1, 0, 2, 4, 3] if i < len(chunk_ids)]
        else:
            self.chunk_ids = [
                "stub_chunk_1",
                "stub_chunk_0",
                "stub_chunk_2",
                "stub_chunk_4",
                "stub_chunk_3",
            ]

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Returns deterministic scored chunk IDs."""
        return [
            (cid, round(0.95 - (idx * 0.05), 4))
            for idx, cid in enumerate(self.chunk_ids[:top_k])
        ]


class StubCrossEncoderReranker:
    """Lightweight deterministic CrossEncoder reranker stub for local development."""

    def __init__(self, model_name: str = "stub-cross-encoder"):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """Returns deterministic reranked subset of candidate tuples."""
        if not candidates:
            return []
        return [
            (cid, round(0.99 - (idx * 0.08), 4))
            for idx, (cid, _text) in enumerate(candidates[:top_k])
        ]


class StubGenerator:
    """Lightweight deterministic LLM generation client stub for local development."""

    def __init__(self, model_name: str = "stub-groq-llama-3.3-70b"):
        self.model_name = model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """Returns deterministic generation result with valid chunk citation."""
        text = (
            "Based on the retrieved judicial precedents [Chunk ID: stub_chunk_0], "
            "the statutory requirements and legal principles are established under the relevant enactments."
        )
        return GenerationResult(
            text=text,
            status="success",
            model_name=self.model_name,
            prompt_tokens=150,
            completion_tokens=45,
            total_tokens=195,
            finish_reason="stop",
        )

    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Returns deterministic answer text."""
        result = self.generate(system_prompt, user_prompt, temperature, max_tokens)
        return result.text or ""


def create_stub_chunks() -> pd.DataFrame:
    """Creates a small in-memory DataFrame of deterministic stub chunk records."""
    data = [
        {
            "chunk_id": "stub_chunk_0",
            "cnr": "STUB010000012020_1",
            "chunk_index": 0,
            "text": (
                "The Supreme Court and High Courts have consistently held that anticipatory bail "
                "under Section 438 CrPC can be granted when there is reasonable apprehension of arrest "
                "for a non-bailable offence, subject to statutory conditions under Section 438(2)."
            ),
            "court_code": "01",
            "decision_date": "2020-01-15",
            "title": "State of Maharashtra v. ABC",
        },
        {
            "chunk_id": "stub_chunk_1",
            "cnr": "STUB010000022021_1",
            "chunk_index": 0,
            "text": (
                "Under Section 138 of the Negotiable Instruments Act, 1881, the statutory notice "
                "must be served within 30 days of receiving information regarding the dishonour of the cheque."
            ),
            "court_code": "02",
            "decision_date": "2021-03-22",
            "title": "XYZ Enterprises v. Union of India",
        },
        {
            "chunk_id": "stub_chunk_2",
            "cnr": "STUB010000032022_1",
            "chunk_index": 0,
            "text": (
                "An application under Section 482 CrPC for quashing of an FIR or charge-sheet is maintainable "
                "where allegations in the first information report do not disclose the commission of any cognizable offence."
            ),
            "court_code": "03",
            "decision_date": "2022-07-10",
            "title": "PQR Corporation v. State",
        },
        {
            "chunk_id": "stub_chunk_3",
            "cnr": "STUB010000042023_1",
            "chunk_index": 0,
            "text": (
                "Article 226 of the Constitution confers discretionary jurisdiction upon the High Courts "
                "to issue prerogative writs for the enforcement of fundamental rights and for any other purpose."
            ),
            "court_code": "04",
            "decision_date": "2023-05-18",
            "title": "RST Ltd v. Assistant Commissioner",
        },
        {
            "chunk_id": "stub_chunk_4",
            "cnr": "STUB010000052024_1",
            "chunk_index": 0,
            "text": (
                "The principles governing interference with an award under Section 34 of the Arbitration "
                "and Conciliation Act, 1996 require patent illegality appearing on the face of the award."
            ),
            "court_code": "05",
            "decision_date": "2024-02-09",
            "title": "LMN Builders v. Metro Rail",
        },
    ]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Component Container
# ---------------------------------------------------------------------------

@dataclass
class PipelineComponents:
    """
    Container holding loaded RAG pipeline components for dependency injection.
    """

    bm25: Union[BM25Retriever, StubBM25Retriever]
    dense: Union[DenseRetriever, StubDenseRetriever]
    reranker: Union[CrossEncoderReranker, StubCrossEncoderReranker]
    chunks: pd.DataFrame
    generator: Optional[Any] = None
    environment: Literal["local_stub", "production"] = "local_stub"
    #: Optional on-disk chunk store (``SqliteChunkStore``). When set, chunk rows are fetched from it
    #: by id and ``chunks`` is an empty frame; when None (the v1 default) ``chunks`` is used.
    chunk_store: Optional[Any] = None

    def get_chunks(self, chunk_ids: Any) -> Dict[str, Dict[str, Any]]:
        """Rows for the given chunk ids (text, cnr, court_code, ...), from whichever store is configured."""
        source = self.chunk_store if self.chunk_store is not None else DataFrameChunkSource(self.chunks)
        return source.get_many(list(chunk_ids))


# ---------------------------------------------------------------------------
# Pipeline Loaders
# ---------------------------------------------------------------------------

def load_stub_pipeline() -> PipelineComponents:
    """Initializes and returns lightweight stub components."""
    stub_chunks = create_stub_chunks()
    stub_chunk_ids = stub_chunks["chunk_id"].tolist()

    return PipelineComponents(
        bm25=StubBM25Retriever(chunk_ids=stub_chunk_ids),
        dense=StubDenseRetriever(chunk_ids=stub_chunk_ids),
        reranker=StubCrossEncoderReranker(),
        chunks=stub_chunks,
        generator=StubGenerator(),
        environment="local_stub",
    )


def _validate_store_chunk_id_resolution(pipeline: PipelineComponents, sample: int = 1000) -> None:
    """For the on-disk chunk store: ids line up by count, and a spread of them resolve to rows.

    Checking every id would read the whole database at startup, so this checks the sizes, the first
    and last ids, and a seeded random sample.
    """
    import random

    store = pipeline.chunk_store
    for name, retriever in (("BM25", pipeline.bm25), ("Dense", pipeline.dense)):
        ids = getattr(retriever, "chunk_ids", None)
        if ids is None or not len(ids):
            raise ValueError(f"{name} retriever exposes no chunk_ids.")
        if len(ids) != len(store):
            raise ValueError(
                f"{name} retriever maps {len(ids)} chunk_ids but the chunk store holds {len(store)} rows; "
                "the index and the store were built from different corpora."
            )
        rng = random.Random(0)
        picks = {0, len(ids) - 1, *(rng.randrange(len(ids)) for _ in range(min(sample, len(ids))))}
        wanted = [ids[i] for i in sorted(picks)]
        missing = [cid for cid in wanted if cid not in store.get_many(wanted)]
        if missing:
            raise ValueError(
                f"{name} retriever references chunk_id(s) absent from the chunk store (first: {missing[0]!r})."
            )


def _validate_chunk_id_resolution(pipeline: PipelineComponents) -> None:
    """Ensures every retrieval chunk ID actually resolves against legal_chunks.parquet."""
    if pipeline.chunk_store is not None:
        _validate_store_chunk_id_resolution(pipeline)
        return
    chunk_ids = pipeline.chunks["chunk_id"]
    if chunk_ids.isna().any():
        raise ValueError("legal_chunks.parquet contains null chunk_id values.")
    if chunk_ids.duplicated().any():
        raise ValueError("legal_chunks.parquet contains duplicate chunk_id values.")

    known_ids = set(chunk_ids.tolist())
    for name, ids in (
        ("BM25", list(getattr(pipeline.bm25, "chunk_ids", None) or [])),
        ("Dense", list(getattr(pipeline.dense, "chunk_ids", None) or [])),
    ):
        if not ids:
            raise ValueError(f"{name} retriever exposes no chunk_ids.")
        missing = [cid for cid in ids if cid not in known_ids]
        if missing:
            raise ValueError(
                f"{name} retriever references {len(missing)} chunk_id(s) absent from "
                f"legal_chunks.parquet (first: {missing[0]!r})."
            )


def validate_pipeline_components(pipeline: PipelineComponents, require_generator: bool = True) -> None:
    """
    Verifies that every retrieval component in a pipeline is actually usable.

    Structural checks run against real retrievers only; stub components are intentionally
    lightweight. For production pipelines, generation must be configured (unless
    ``require_generator`` is False, as for retrieval-only evaluation, which needs no Groq key)
    and every chunk ID returned by the retrievers must resolve against legal_chunks.parquet.
    The API always uses the default, so serving still fails closed without a generator.

    Raises:
        ValueError: when a retrieval component cannot serve queries.
    """
    if isinstance(pipeline.bm25, (BM25Retriever, Bm25sRetriever)):
        pipeline.bm25.validate()
    if isinstance(pipeline.dense, DenseRetriever):
        pipeline.dense.validate()
    if isinstance(pipeline.reranker, (CrossEncoderReranker, OnnxReranker)):
        pipeline.reranker.validate()

    if pipeline.environment == "production":
        if require_generator and pipeline.generator is None:
            raise ValueError("Production pipeline has no generation client configured.")
        _validate_chunk_id_resolution(pipeline)


def _missing_artifacts(settings: Settings) -> List[str]:
    """Plain-language list of required artifacts that are not on disk, for the configured backends."""
    from legalrag.retrieval.bm25s_backend import METADATA_FILE

    missing: List[str] = []
    if settings.bm25_backend == "bm25s":
        if not (settings.bm25s_dir / METADATA_FILE).is_file():
            missing.append(f"bm25s index missing at: {settings.bm25s_dir}")
    elif not settings.bm25_path.is_file():
        missing.append(f"BM25 index artifact missing at: {settings.bm25_path}")
    if not settings.dense_index_path.is_file():
        missing.append(f"Dense FAISS index artifact missing at: {settings.dense_index_path}")
    if settings.chunk_store == "sqlite":
        if not settings.chunk_store_path.is_file():
            missing.append(f"SQLite chunk store missing at: {settings.chunk_store_path}")
        if not settings.chunk_store_path.with_name("chunk_ids.npy").is_file():
            missing.append(f"Chunk id array missing at: {settings.chunk_store_path.with_name('chunk_ids.npy')}")
    elif not settings.chunks_path.is_file():
        missing.append(f"Chunks parquet artifact missing at: {settings.chunks_path}")
    if settings.embedder_backend == "onnx" and not settings.onnx_embedder_dir.is_dir():
        missing.append(f"ONNX embedder directory missing at: {settings.onnx_embedder_dir}")
    if settings.reranker_backend == "onnx" and not settings.onnx_reranker_dir.is_dir():
        missing.append(f"ONNX reranker directory missing at: {settings.onnx_reranker_dir}")
    return missing


def _check_ids_agree(first: Any, second: Any, first_name: str, second_name: str, sample: int = 1000) -> None:
    """Two id sequences must line up: same length, same first and last id, and a seeded random sample equal."""
    import random

    if len(first) != len(second):
        raise ValueError(
            f"{first_name} has {len(first)} chunk ids but {second_name} has {len(second)}; they were built from different corpora."
        )
    if not len(first):
        return
    rng = random.Random(0)
    picks = sorted({0, len(first) - 1, *(rng.randrange(len(first)) for _ in range(min(sample, len(first))))})
    for position in picks:
        if first[position] != second[position]:
            raise ValueError(
                f"{first_name} and {second_name} disagree at position {position} "
                f"({first[position]!r} vs {second[position]!r}); the indexes do not match."
            )


def load_retrieval_components(settings: Settings) -> PipelineComponents:
    """
    Loads the retrieval parts (BM25, dense index, reranker, chunk text) with no generation client.

    The backends follow the settings flags; the defaults load exactly the v1 artifacts. Used by
    ``load_production_pipeline`` and by the retrieval evaluation, which needs no Groq key.
    Raises FileNotFoundError if any required artifact is missing.
    """
    missing_artifacts = _missing_artifacts(settings)
    if missing_artifacts:
        raise FileNotFoundError(
            "Production pipeline initialization failed due to missing required artifacts:\n"
            + "\n".join(f"  - {err}" for err in missing_artifacts)
            + "\nPlease run 'scripts/download_artifacts.py' or verify configured artifact paths."
        )

    chunk_store = None
    if settings.chunk_store == "sqlite":
        from legalrag.retrieval.chunk_store import SqliteChunkStore

        logger.info("Opening SQLite chunk store at %s", settings.chunk_store_path)
        chunk_store = SqliteChunkStore(settings.chunk_store_path)
        chunk_ids: Any = chunk_store.load_chunk_ids(mmap=True)
        chunks_df = pd.DataFrame()
    else:
        logger.info("Loading chunk metadata parquet from %s", settings.chunks_path)
        chunks_df = pd.read_parquet(settings.chunks_path)
        if "chunk_id" not in chunks_df.columns:
            raise ValueError(
                f"Chunks parquet at '{settings.chunks_path}' is missing required 'chunk_id' column."
            )
        chunk_ids = chunks_df["chunk_id"].tolist()

    if settings.bm25_backend == "bm25s":
        logger.info("Loading bm25s index from %s (mmap=%s)", settings.bm25s_dir, settings.bm25s_mmap)
        bm25: Any = Bm25sRetriever.load(settings.bm25s_dir, mmap=settings.bm25s_mmap)
        _check_ids_agree(bm25.chunk_ids, chunk_ids, "the bm25s index", "the chunk table")
    else:
        logger.info("Loading BM25 index from %s", settings.bm25_path)
        bm25 = BM25Retriever.load(settings.bm25_path)
        if not bm25.chunk_ids and len(chunk_ids):
            logger.warning(
                "BM25 artifact carries no chunk_ids; falling back to legal_chunks.parquet ordering."
            )
            bm25.chunk_ids = list(chunk_ids)

    embedder = None
    if settings.embedder_backend == "onnx":
        from legalrag.retrieval.onnx_models import OnnxEmbedder

        logger.info("Loading ONNX embedder from %s", settings.onnx_embedder_dir)
        embedder = OnnxEmbedder.load(settings.onnx_embedder_dir, settings.onnx_threads)
    logger.info("Loading Dense FAISS index from %s", settings.dense_index_path)
    dense = DenseRetriever.load_index(
        index_path=settings.dense_index_path,
        chunk_ids=chunk_ids,
        model_name=settings.embedding_model_name,
        nprobe=settings.dense_nprobe,
        mmap=settings.dense_mmap,
        embedder=embedder,
    )

    if settings.reranker_backend == "onnx":
        logger.info("Loading ONNX reranker from %s", settings.onnx_reranker_dir)
        reranker: Any = OnnxReranker.load(settings.onnx_reranker_dir, settings.onnx_threads)
    else:
        logger.info("Initializing Cross-Encoder reranker (%s)", settings.reranker_model_name)
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model_name,
        )

    return PipelineComponents(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        chunks=chunks_df,
        generator=None,
        environment="production",
        chunk_store=chunk_store,
    )


def load_production_pipeline(settings: Settings) -> PipelineComponents:
    """
    Loads full production pipeline components from verified disk artifacts.
    Raises explicit FileNotFoundError if any artifact is missing.
    """
    pipeline = load_retrieval_components(settings)

    logger.info("Initializing Groq generation client (%s)", settings.groq_model_name)
    pipeline.generator = LegalGenerationClient(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model_name,
        request_timeout=settings.groq_request_timeout,
        max_retries=settings.groq_max_retries,
    )

    # Fail closed at initialization if any retrieval component is not actually usable.
    validate_pipeline_components(pipeline)

    if settings.model_warmup_enabled:
        logger.info("Warming up embedding model (%s)", settings.embedding_model_name)
        pipeline.dense.warmup()
        logger.info("Warming up cross-encoder reranker (%s)", settings.reranker_model_name)
        pipeline.reranker.warmup()

    return pipeline


# ---------------------------------------------------------------------------
# Singleton Loader & Dependency Injection
# ---------------------------------------------------------------------------

_cached_pipeline: Optional[PipelineComponents] = None
_initialization_error: Optional[PipelineInitializationError] = None
_pipeline_lock = threading.Lock()


def get_pipeline(settings: Optional[Settings] = None) -> PipelineComponents:
    """
    Returns the process-level singleton PipelineComponents instance.

    Loading is protected by a lock so concurrent first requests cannot initialize
    SentenceTransformer, CrossEncoder, FAISS, BM25, or the generator more than once.
    A failed initialization is remembered and re-raised (fail closed) rather than
    retried on every request or replaced with stub components.
    """
    global _cached_pipeline, _initialization_error

    if _cached_pipeline is not None:
        return _cached_pipeline
    if _initialization_error is not None:
        raise _initialization_error

    if settings is None:
        settings = get_settings()

    with _pipeline_lock:
        # Re-check inside the lock: a concurrent request may have initialized it already.
        if _cached_pipeline is not None:
            return _cached_pipeline
        if _initialization_error is not None:
            raise _initialization_error

        try:
            if settings.is_local_stub:
                logger.info("Initializing LegalRAG pipeline in 'local_stub' mode.")
                pipeline = load_stub_pipeline()
            elif settings.is_production:
                logger.info("Initializing LegalRAG pipeline in 'production' mode.")
                pipeline = load_production_pipeline(settings)
            else:
                raise ValueError(f"Unrecognized environment mode: '{settings.environment}'")

            if settings.is_production and pipeline.environment != "production":
                raise PipelineInitializationError(
                    "Production mode refused to serve stub pipeline components."
                )

            _cached_pipeline = pipeline
            _initialization_error = None
        except PipelineInitializationError as exc:
            _initialization_error = exc
            raise
        except Exception as exc:  # noqa: BLE001 - normalize all init failures
            wrapped = PipelineInitializationError(str(exc))
            _initialization_error = wrapped
            raise wrapped from exc

    return _cached_pipeline


def check_readiness(settings: Optional[Settings] = None) -> Tuple[bool, str]:
    """
    Lightweight readiness probe.

    Verifies that inference dependencies are initialized and available. Returns
    ``(True, reason)`` when ready and ``(False, reason)`` otherwise. Never initializes
    anything for the liveness endpoint.
    """
    if settings is None:
        settings = get_settings()

    if settings.environment == "production":
        config_issues = settings.production_configuration_issues
        if config_issues:
            logger.error("Readiness failed due to configuration: %s", "; ".join(config_issues))
            return False, "Service dependencies are not initialized."

    try:
        pipeline = get_pipeline(settings)
        validate_pipeline_components(pipeline)
    except Exception as exc:  # noqa: BLE001 - readiness must never raise
        logger.error("Readiness failed during pipeline validation: %s", exc)
        return False, "Service dependencies are not initialized."

    if settings.is_production and pipeline.environment != "production":
        logger.error("Readiness failed: production requested but stub components loaded.")
        return False, "Service dependencies are not initialized."

    return True, "ready"


def reset_pipeline_cache() -> None:
    """
    Clears the cached pipeline singleton and any recorded initialization failure.
    Used for test isolation and environment reconfiguration.
    """
    global _cached_pipeline, _initialization_error
    _cached_pipeline = None
    _initialization_error = None

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
from typing import Any, List, Literal, Optional, Tuple, Union

import pandas as pd

from legalrag.api.config import Settings, get_settings
from legalrag.retrieval.bm25 import BM25Retriever
from legalrag.retrieval.dense import DenseRetriever
from legalrag.retrieval.reranker import CrossEncoderReranker
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


def _validate_chunk_id_resolution(pipeline: PipelineComponents) -> None:
    """Ensures every retrieval chunk ID actually resolves against legal_chunks.parquet."""
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


def validate_pipeline_components(pipeline: PipelineComponents) -> None:
    """
    Verifies that every retrieval component in a pipeline is actually usable.

    Structural checks run against real retrievers only; stub components are intentionally
    lightweight. For production pipelines, generation must be configured and every chunk
    ID returned by the retrievers must resolve against legal_chunks.parquet.

    Raises:
        ValueError: when a retrieval component cannot serve queries.
    """
    if isinstance(pipeline.bm25, BM25Retriever):
        pipeline.bm25.validate()
    if isinstance(pipeline.dense, DenseRetriever):
        pipeline.dense.validate()
    if isinstance(pipeline.reranker, CrossEncoderReranker):
        pipeline.reranker.validate()

    if pipeline.environment == "production":
        if pipeline.generator is None:
            raise ValueError("Production pipeline has no generation client configured.")
        _validate_chunk_id_resolution(pipeline)


def load_production_pipeline(settings: Settings) -> PipelineComponents:
    """
    Loads full production pipeline components from verified disk artifacts.
    Raises explicit FileNotFoundError if any artifact is missing.
    """
    missing_artifacts = []
    if not settings.bm25_path.is_file():
        missing_artifacts.append(f"BM25 index artifact missing at: {settings.bm25_path}")
    if not settings.dense_index_path.is_file():
        missing_artifacts.append(f"Dense FAISS index artifact missing at: {settings.dense_index_path}")
    if not settings.chunks_path.is_file():
        missing_artifacts.append(f"Chunks parquet artifact missing at: {settings.chunks_path}")

    if missing_artifacts:
        raise FileNotFoundError(
            "Production pipeline initialization failed due to missing required artifacts:\n"
            + "\n".join(f"  - {err}" for err in missing_artifacts)
            + "\nPlease run 'scripts/download_artifacts.py' or verify configured artifact paths."
        )

    logger.info("Loading chunk metadata parquet from %s", settings.chunks_path)
    chunks_df = pd.read_parquet(settings.chunks_path)
    if "chunk_id" not in chunks_df.columns:
        raise ValueError(
            f"Chunks parquet at '{settings.chunks_path}' is missing required 'chunk_id' column."
        )
    chunk_ids = chunks_df["chunk_id"].tolist()

    logger.info("Loading BM25 index from %s", settings.bm25_path)
    bm25 = BM25Retriever.load(settings.bm25_path)
    if not bm25.chunk_ids and chunk_ids:
        logger.warning(
            "BM25 artifact carries no chunk_ids; falling back to legal_chunks.parquet ordering."
        )
        bm25.chunk_ids = list(chunk_ids)

    logger.info("Loading Dense FAISS index from %s", settings.dense_index_path)
    dense = DenseRetriever.load_index(
        index_path=settings.dense_index_path,
        chunk_ids=chunk_ids,
        model_name=settings.embedding_model_name,
    )

    logger.info("Initializing Cross-Encoder reranker (%s)", settings.reranker_model_name)
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
    )

    logger.info("Initializing Groq generation client (%s)", settings.groq_model_name)
    generator = LegalGenerationClient(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model_name,
        request_timeout=settings.groq_request_timeout,
        max_retries=settings.groq_max_retries,
    )

    pipeline = PipelineComponents(
        bm25=bm25,
        dense=dense,
        reranker=reranker,
        chunks=chunks_df,
        generator=generator,
        environment="production",
    )

    # Fail closed at initialization if any retrieval component is not actually usable.
    validate_pipeline_components(pipeline)

    if settings.model_warmup_enabled:
        logger.info("Warming up embedding model (%s)", settings.embedding_model_name)
        dense.warmup()
        logger.info("Warming up cross-encoder reranker (%s)", settings.reranker_model_name)
        reranker.warmup()

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

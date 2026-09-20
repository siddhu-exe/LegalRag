"""
FastAPI route definitions for LegalRAG service.

Exposes:
- GET  /health: Lightweight liveness check. Never loads models/artifacts.
- GET  /ready:  Readiness check. Verifies the inference pipeline is initialized.
- POST /query:  End-to-end multi-stage retrieval and grounded QA query pipeline.

HTTP error semantics:
- 422 invalid request body (FastAPI/Pydantic validation)
- 500 unexpected retrieval/reranking failure
- 502 LLM/generation provider failure
- 503 pipeline not initialized / artifacts unavailable (readiness failure)

All error bodies are sanitized: no API keys, tokens, stack traces, internal paths, or
provider credentials are ever returned to clients.
"""

import time
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from legalrag.api.config import Settings, get_settings
from legalrag.api.dependencies import PipelineComponents, check_readiness, get_pipeline
from legalrag.api.schemas import (
    Citation,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)
from legalrag.generation.client import GenerationError
from legalrag.generation.prompts import SYSTEM_PROMPT, build_rag_prompt, extract_citations
from legalrag.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

router = APIRouter()

RETRIEVAL_ERROR_DETAIL = "Document retrieval failed."
RERANK_ERROR_DETAIL = "Candidate reranking failed."
GENERATION_ERROR_DETAIL = "Answer generation failed."
NOT_READY_DETAIL = "Service is not ready."


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Lightweight liveness probe.

    Reports only that the process is up. It deliberately does not initialize or load
    the retrieval pipeline, models, or artifacts.
    """
    return HealthResponse(
        status="ok",
        environment=settings.environment,
    )


@router.get("/ready", response_model=ReadinessResponse, tags=["Health"])
def readiness_check(
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """
    Readiness probe.

    Returns HTTP 200 only when the pipeline required for inference is initialized and
    its required artifacts/models are available. Returns HTTP 503 otherwise, so
    orchestrators never route traffic to a production instance that would fail closed.
    """
    ready, _reason = check_readiness(settings)
    if not ready:
        raise HTTPException(status_code=503, detail=NOT_READY_DETAIL)
    return ReadinessResponse(status="ready", environment=settings.environment)


@router.post("/query", response_model=QueryResponse, tags=["Query"])
def query_endpoint(
    request: QueryRequest,
    pipeline: PipelineComponents = Depends(get_pipeline),
) -> QueryResponse:
    """
    Executes the multi-stage grounded legal question answering cascade:
    1. BM25 Lexical Retrieval (top-50)
    2. Dense Semantic Vector Retrieval (top-50)
    3. Reciprocal Rank Fusion (RRF, top-50)
    4. Neural Cross-Encoder Reranking (top-5)
    5. Prompt construction and grounded LLM generation
    6. Citation extraction and latency attribution
    """
    t_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Stage 1: Hybrid Retrieval & Fusion
    # -------------------------------------------------------------------------
    t_ret_start = time.perf_counter()
    try:
        bm25_scored = pipeline.bm25.search(request.question, top_k=50)
        dense_scored = pipeline.dense.search(request.question, top_k=50)

        bm25_ids = [cid for cid, _ in bm25_scored]
        dense_ids = [cid for cid, _ in dense_scored]

        fused_scored = reciprocal_rank_fusion(
            ranked_lists=[bm25_ids, dense_ids],
            k=60,
            top_n=50,
        )
        fused_ids = [cid for cid, _ in fused_scored]
        retrieve_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)
    except Exception as exc:  # noqa: BLE001 - sanitize backend failures
        logger.exception("Retrieval stage failure (%s).", type(exc).__name__)
        raise HTTPException(status_code=500, detail=RETRIEVAL_ERROR_DETAIL) from exc

    # -------------------------------------------------------------------------
    # Stage 2: Cross-Encoder Reranking
    # -------------------------------------------------------------------------
    t_rerank_start = time.perf_counter()
    top_chunk_ids: List[str] = []
    chunk_dict: Dict[str, Dict[str, Any]] = {}

    try:
        if not pipeline.chunks.empty and "chunk_id" in pipeline.chunks.columns:
            subset_df = pipeline.chunks[pipeline.chunks["chunk_id"].isin(fused_ids)]
            chunk_dict = subset_df.set_index("chunk_id").to_dict(orient="index")

        candidates = [
            (cid, chunk_dict[cid].get("text", ""))
            for cid in fused_ids
            if cid in chunk_dict
        ]

        if candidates:
            reranked_scored = pipeline.reranker.rerank(
                query=request.question,
                candidates=candidates,
                top_k=5,
            )
            top_chunk_ids = [cid for cid, _ in reranked_scored]
        else:
            top_chunk_ids = fused_ids[:5]

        rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)
    except Exception as exc:  # noqa: BLE001 - sanitize backend failures
        logger.exception("Reranking stage failure (%s).", type(exc).__name__)
        raise HTTPException(status_code=500, detail=RERANK_ERROR_DETAIL) from exc

    # -------------------------------------------------------------------------
    # Stage 3: Grounded Answer Generation
    # -------------------------------------------------------------------------
    t_gen_start = time.perf_counter()
    try:
        context_blocks = []
        for cid in top_chunk_ids:
            meta = chunk_dict.get(cid, {})
            context_blocks.append(
                {
                    "chunk_id": cid,
                    "text": meta.get("text", ""),
                    "court_code": meta.get("court_code"),
                    "decision_date": meta.get("decision_date"),
                }
            )

        user_prompt = build_rag_prompt(
            question=request.question,
            context_blocks=context_blocks,
        )

        if pipeline.generator is None:
            raise GenerationError(
                message="Generation client is not configured on pipeline.",
                status_code=502,
            )

        gen_result = pipeline.generator.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        if gen_result is None or not getattr(gen_result, "is_success", False) or gen_result.text is None:
            raise GenerationError(message=GENERATION_ERROR_DETAIL, status_code=502)
    except GenerationError as exc:
        logger.warning("Generation failed (%s).", type(exc).__name__)
        raise HTTPException(status_code=502, detail=GENERATION_ERROR_DETAIL) from exc
    except Exception as exc:  # noqa: BLE001 - convert provider failures to 502
        logger.exception("Generation stage failure (%s).", type(exc).__name__)
        raise HTTPException(status_code=502, detail=GENERATION_ERROR_DETAIL) from exc

    generate_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 2)
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    answer_text = gen_result.text
    cited_cids = extract_citations(answer_text)

    # Build citation metadata objects, strictly grounded to context chunks
    citations: List[Citation] = []
    for cid in cited_cids:
        if cid not in top_chunk_ids:
            # Shield against hallucinated citation IDs not present in context
            logger.warning("Generation cited chunk_id '%s' not present in retrieved context.", cid)
            continue

        meta = chunk_dict.get(cid)
        if meta is None and not pipeline.chunks.empty and "chunk_id" in pipeline.chunks.columns:
            match = pipeline.chunks[pipeline.chunks["chunk_id"] == cid]
            meta = match.iloc[0].to_dict() if not match.empty else {}
        elif meta is None:
            meta = {}

        citations.append(
            Citation(
                chunk_id=cid,
                cnr=meta.get("cnr"),
                court_code=str(meta["court_code"]) if meta.get("court_code") is not None else None,
                decision_date=str(meta["decision_date"]) if meta.get("decision_date") is not None else None,
                title=meta.get("title"),
            )
        )

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        retrieved_chunk_ids=top_chunk_ids,
        retrieve_ms=retrieve_ms,
        rerank_ms=rerank_ms,
        generate_ms=generate_ms,
        total_ms=total_ms,
        status="ok",
    )

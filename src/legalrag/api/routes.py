"""
FastAPI route definitions for LegalRAG service.

Exposes:
- GET  /health: Lightweight health check returning environment and service status.
- POST /query:  End-to-end multi-stage retrieval and grounded QA query pipeline.
"""

import time
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends

from legalrag.api.config import Settings, get_settings
from legalrag.api.dependencies import PipelineComponents, get_pipeline
from legalrag.api.schemas import Citation, HealthResponse, QueryRequest, QueryResponse
from legalrag.generation.prompts import SYSTEM_PROMPT, build_rag_prompt, extract_citations
from legalrag.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Lightweight health probe endpoint.
    Returns operational status and active runtime environment without loading heavy models.
    """
    return HealthResponse(
        status="ok",
        environment=settings.environment,
    )


@router.post("/query", response_model=QueryResponse)
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
    retrieve_ms = 0.0
    rerank_ms = 0.0
    generate_ms = 0.0

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
    except Exception as exc:
        retrieve_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        logger.exception("Retrieval stage failure: %s", str(exc))
        return QueryResponse(
            answer=f"Retrieval failed: {str(exc)}",
            citations=[],
            retrieved_chunk_ids=[],
            retrieve_ms=retrieve_ms,
            rerank_ms=0.0,
            generate_ms=0.0,
            total_ms=total_ms,
            status="retrieval_error",
        )

    # -------------------------------------------------------------------------
    # Stage 2: Cross-Encoder Reranking
    # -------------------------------------------------------------------------
    t_rerank_start = time.perf_counter()
    top_chunk_ids: List[str] = []
    chunk_dict: Dict[str, Dict[str, Any]] = {}

    try:
        # Extract chunk texts and metadata for fused candidate set
        if not pipeline.chunks.empty and "chunk_id" in pipeline.chunks.columns:
            subset_df = pipeline.chunks[pipeline.chunks["chunk_id"].isin(fused_ids)]
            chunk_dict = subset_df.set_index("chunk_id").to_dict(orient="index")

        candidates = [
            (cid, chunk_dict[cid].get("text", ""))
            for cid in fused_ids
            if cid in chunk_dict
        ]

        # If candidates are empty (e.g. empty corpus), fallback to empty rerank
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
    except Exception as exc:
        rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        logger.exception("Reranking stage failure: %s", str(exc))
        return QueryResponse(
            answer=f"Reranking failed: {str(exc)}",
            citations=[],
            retrieved_chunk_ids=[],
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            generate_ms=0.0,
            total_ms=total_ms,
            status="retrieval_error",
        )

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
                    "court_name": meta.get("court_name"),
                    "decision_date": meta.get("decision_date"),
                }
            )

        user_prompt = build_rag_prompt(
            question=request.question,
            context_blocks=context_blocks,
        )

        if pipeline.generator is None:
            raise RuntimeError("Generation client is not configured on pipeline.")

        gen_result = pipeline.generator.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        generate_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        if not getattr(gen_result, "is_success", False) or gen_result.text is None:
            error_msg = getattr(gen_result, "error_message", None) or "LLM generation failed."
            logger.error("Generation failed: %s", error_msg)
            return QueryResponse(
                answer=f"Generation failed: {error_msg}",
                citations=[],
                retrieved_chunk_ids=top_chunk_ids,
                retrieve_ms=retrieve_ms,
                rerank_ms=rerank_ms,
                generate_ms=generate_ms,
                total_ms=total_ms,
                status="generation_error",
            )

        answer_text = gen_result.text
        cited_cids = extract_citations(answer_text)

        # Build citation metadata objects
        citations: List[Citation] = []
        for cid in cited_cids:
            meta = chunk_dict.get(cid)
            if meta is None and not pipeline.chunks.empty and "chunk_id" in pipeline.chunks.columns:
                match = pipeline.chunks[pipeline.chunks["chunk_id"] == cid]
                if not match.empty:
                    meta = match.iloc[0].to_dict()
                else:
                    meta = {}
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

    except Exception as exc:
        generate_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        logger.exception("Generation stage failure: %s", str(exc))
        return QueryResponse(
            answer=f"Generation failed: {str(exc)}",
            citations=[],
            retrieved_chunk_ids=top_chunk_ids,
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
            status="generation_error",
        )

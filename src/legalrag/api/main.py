"""
Main FastAPI application entrypoint for the LegalRAG service.

Initializes the FastAPI application, mounts API routes, sets up CORS middleware, and
manages application lifespan events (production artifact provisioning + pipeline warmup).

Production fails closed: if artifact provisioning or pipeline initialization fails, the
service still starts but reports NOT READY (HTTP 503) and never serves stub responses.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from legalrag.api.config import get_settings
from legalrag.api.dependencies import PipelineInitializationError, get_pipeline
from legalrag.api.provisioning import ArtifactProvisioningError, provision_production_artifacts
from legalrag.api.routes import router as api_router
from legalrag.generation.client import GenerationError

# Configure root logger format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager for startup and shutdown routines.

    In production, missing artifacts are provisioned from Hugging Face Hub before the
    pipeline is initialized. Initialization failures are logged and left uncached as a
    not-ready state; they must never silently degrade to stub components.
    """
    settings = get_settings()
    logger.info(
        "Starting LegalRAG API service (Environment: '%s', Port: %d)",
        settings.environment,
        settings.api_port,
    )

    try:
        if settings.is_production:
            provision_production_artifacts(settings)
        # Pre-warm/initialize pipeline components
        pipeline = get_pipeline(settings)
        logger.info(
            "LegalRAG pipeline initialized successfully in '%s' mode.",
            pipeline.environment,
        )
    except Exception as exc:  # noqa: BLE001 - keep serving so /ready can report 503
        logger.error(
            "Pipeline initialization failed on startup (%s). Service will report not ready.",
            type(exc).__name__,
        )

    yield

    logger.info("Shutting down LegalRAG API service.")


def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    """
    app = FastAPI(
        title="LegalRAG API",
        description=(
            "Hybrid Retrieval & Evidence-Grounded Question Answering "
            "over Indian High Court Judgments"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS for local UI, Streamlit, and external clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Sanitized error handlers: never leak credentials, paths, or stack traces.
    @app.exception_handler(PipelineInitializationError)
    async def _pipeline_not_ready_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "Service is not ready."})

    @app.exception_handler(ArtifactProvisioningError)
    async def _provisioning_not_ready_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "Service is not ready."})

    @app.exception_handler(GenerationError)
    async def _generation_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": "Answer generation failed."})

    # Mount API routes
    app.include_router(api_router)

    @app.get("/", tags=["Root"])
    def root():
        """Root endpoint returning service identity and docs link."""
        settings = get_settings()
        return {
            "name": "LegalRAG API",
            "version": "0.1.0",
            "environment": settings.environment,
            "docs_url": "/docs",
            "health_url": "/health",
            "readiness_url": "/ready",
            "query_url": "/query",
        }

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "legalrag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )

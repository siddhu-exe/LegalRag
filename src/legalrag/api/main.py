"""
Main FastAPI application entrypoint for the LegalRAG service.

Initializes the FastAPI application, mounts API routes, sets up CORS middleware,
and manages application lifespan events (logging environment configuration and pipeline warmup).
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from legalrag.api.config import get_settings
from legalrag.api.dependencies import get_pipeline
from legalrag.api.routes import router as api_router

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
    Initializes pipeline singleton and logs environment configuration.
    """
    settings = get_settings()
    logger.info(
        "Starting LegalRAG API service (Environment: '%s', Port: %d)",
        settings.environment,
        settings.api_port,
    )

    try:
        # Pre-warm/initialize pipeline components
        pipeline = get_pipeline(settings)
        logger.info(
            "LegalRAG pipeline initialized successfully in '%s' mode.",
            pipeline.environment,
        )
    except Exception as exc:
        logger.error("Error during pipeline initialization on startup: %s", exc)
        # We let the server start so health probes or error messages can be served
        if settings.is_production:
            logger.warning(
                "Production startup encountered missing artifacts or invalid keys. "
                "Verify environment variables and run artifact download script."
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

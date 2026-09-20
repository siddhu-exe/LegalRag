"""
Configuration management for the LegalRAG FastAPI backend service.

Supports two runtime environments:
- 'production' (fail-closed default): Full pipeline execution with real retrieval indexes
  and Groq LLM generation. Production never falls back to stub components.
- 'local_stub': Lightweight local development mode without loading heavy models or FAISS
  index. Must be selected explicitly (via environment variable or .env) for local dev/tests.

All configuration is server-side only. Clients cannot supply any of these values through
the API request body.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables and/or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Runtime Environment: fail-closed default is 'production'. Local development and tests
    # must explicitly opt in to 'local_stub'.
    environment: Literal["local_stub", "production"] = "production"

    # Artifact directory and Hub storage location
    artifact_dir: Path = Field(
        default=Path("artifacts"),
        validation_alias=AliasChoices("artifact_dir", "ARTIFACT_DIR"),
        description="Local directory containing runtime retrieval artifacts.",
    )
    hf_repo_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("hf_repo_id", "HF_REPO_ID", "HF_HUB_REPO"),
        description="Hugging Face Hub repository ID hosting production artifacts.",
    )
    hf_token: Optional[str] = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices(
            "hf_token", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"
        ),
        description="Hugging Face Hub access token for private repositories or rate limits.",
    )

    # Model identifiers (locked architecture defaults + Groq generation)
    embedding_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        validation_alias=AliasChoices("embedding_model_name", "EMBEDDING_MODEL_NAME"),
        description="SentenceTransformer embedding model name.",
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices("reranker_model_name", "RERANKER_MODEL_NAME"),
        description="CrossEncoder reranker model name.",
    )
    groq_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("groq_model_name", "GROQ_MODEL_NAME"),
        description="Groq LLM model identifier for grounded generation.",
    )

    # Groq client hardening (explicit timeout + bounded retries for transient failures)
    groq_request_timeout: float = Field(
        default=30.0,
        validation_alias=AliasChoices("groq_request_timeout", "GROQ_REQUEST_TIMEOUT"),
        description="Per-request timeout (seconds) for Groq generation calls.",
    )
    groq_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias=AliasChoices("groq_max_retries", "GROQ_MAX_RETRIES"),
        description="Maximum bounded retries for transient Groq provider failures.",
    )

    # Groq API configuration (externalized secrets)
    groq_api_key: Optional[str] = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("groq_api_key", "GROQ_API_KEY"),
        description="Groq API key for grounded generation.",
    )

    # API Server Network Binding (supports Azure/HF PORT or API_PORT)
    api_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("api_host", "API_HOST"),
        description="Network interface binding host.",
    )
    api_port: int = Field(
        default=7860,
        validation_alias=AliasChoices("api_port", "API_PORT", "PORT"),
        description="Port for the API service (defaults to standard 7860).",
    )

    # Startup provisioning of production artifacts from Hugging Face Hub
    artifact_download_timeout_seconds: int = Field(
        default=3600,
        validation_alias=AliasChoices(
            "artifact_download_timeout_seconds", "ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS"
        ),
        description="Maximum time (seconds) allowed for startup artifact provisioning.",
    )

    @property
    def bm25_path(self) -> Path:
        """Filesystem path to the BM25 index pickle artifact."""
        return self.artifact_dir / "bm25.pkl"

    @property
    def dense_index_path(self) -> Path:
        """Filesystem path to the FAISS dense index artifact."""
        return self.artifact_dir / "dense.index"

    @property
    def chunks_path(self) -> Path:
        """Filesystem path to the chunk metadata parquet artifact."""
        return self.artifact_dir / "legal_chunks.parquet"

    @property
    def is_production(self) -> bool:
        """Returns True if the runtime environment is set to production."""
        return self.environment == "production"

    @property
    def is_local_stub(self) -> bool:
        """Returns True if the runtime environment is set to local stub."""
        return self.environment == "local_stub"

    @property
    def production_configuration_issues(self) -> List[str]:
        """
        Returns production configuration problems that make the service non-ready.

        These are reported by the readiness endpoint (HTTP 503) rather than raising at
        settings construction time, so the process can still expose a non-ready state
        instead of crashing or silently serving stub responses.
        """
        issues: List[str] = []
        if self.environment == "production":
            if not self.groq_api_key or not self.groq_api_key.strip():
                issues.append("GROQ_API_KEY is not configured.")
        return issues


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of Settings.
    Can be overridden in dependency injection during testing.
    """
    return Settings()

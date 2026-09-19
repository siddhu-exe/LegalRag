"""
Configuration management for the LegalRAG FastAPI backend service.

Supports two runtime environments:
- 'local_stub': Lightweight local development mode without loading heavy models or FAISS index.
- 'production': Full pipeline execution with real retrieval indexes and Groq LLM generation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from pydantic import AliasChoices, Field, model_validator
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

    # Runtime Environment: 'local_stub' (default for 6GB dev laptop) or 'production'
    environment: Literal["local_stub", "production"] = "local_stub"

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
        validation_alias=AliasChoices(
            "groq_model_name", "GROQ_MODEL_NAME", "generation_model_name", "GENERATION_MODEL_NAME"
        ),
        description="Groq LLM model identifier for grounded generation.",
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

    @model_validator(mode="after")
    def validate_environment_credentials(self) -> "Settings":
        """
        Ensures production-specific credentials are provided when running in production mode.
        Local stub mode does not require external credentials.
        """
        if self.environment == "production":
            if not self.groq_api_key or not self.groq_api_key.strip():
                raise ValueError(
                    "Production environment requires 'GROQ_API_KEY' to be set."
                )
        return self


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of Settings.
    Can be overridden in dependency injection during testing.
    """
    return Settings()

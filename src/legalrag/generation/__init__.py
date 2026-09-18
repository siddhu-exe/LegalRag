"""Context assembly, prompt formatting, and LLM generation interface."""

from legalrag.generation.prompts import (
    SYSTEM_PROMPT,
    format_context_block,
    build_rag_prompt,
)
from legalrag.generation.client import LegalGenerationClient

__all__ = [
    "SYSTEM_PROMPT",
    "format_context_block",
    "build_rag_prompt",
    "LegalGenerationClient",
]

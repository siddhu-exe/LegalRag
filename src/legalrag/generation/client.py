"""
OpenAI-compatible client interface for LegalRAG generation layer.
Routes requests via OmniRoute / configured OpenAI-compatible gateway with greedy decoding (temperature=0.0).
"""

import os
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class LegalGenerationClient:
    """
    Generation client for executing grounded legal QA against an OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4-turbo-preview",
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for LegalGenerationClient. Install it via pip install openai.")

        self.api_key = api_key or os.getenv("OMNIROUTE_API_KEY") or os.getenv("OPENAI_API_KEY", "dummy-key")
        self.base_url = base_url or os.getenv("OMNIROUTE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Executes greedy generation for grounded legal question answering.

        Args:
            system_prompt: System prompt with strict citation and anti-hallucination rules.
            user_prompt: Formatted user prompt with retrieved context blocks.
            temperature: Sampling temperature (default 0.0 for deterministic reproducibility).
            max_tokens: Maximum token limit for the response.

        Returns:
            The generated response string.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""

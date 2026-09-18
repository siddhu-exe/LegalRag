"""
Google Gemini generation client interface for the LegalRAG generation layer.

Implements grounded legal QA generation via the official google-genai SDK
with greedy decoding (temperature=0.0) and explicit error shielding.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    genai_errors = None  # type: ignore


class GenerationError(Exception):
    """Custom exception raised when upstream LLM API generation fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_type: str = "generation_error",
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


@dataclass
class GenerationResult:
    """
    Structured outcome of an LLM generation call.
    Explicitly separates valid model output from upstream system errors.
    """

    text: Optional[str] = None
    status: str = "success"  # "success", "api_error", "rate_limited", "auth_error", "bad_request", "empty_response"
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    model_name: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == "success" and self.text is not None


class LegalGenerationClient:
    """
    Generation client for executing grounded legal QA against Google Gemini models.
    Provides strict error isolation to prevent upstream error strings from leaking into answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
    ):
        if genai is None:
            raise ImportError(
                "google-genai package is required for LegalGenerationClient. "
                "Install it via: pip install google-genai"
            )

        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "Google Gemini API key is required. Set 'GEMINI_API_KEY' environment variable "
                "or pass 'api_key' to LegalGenerationClient."
            )

        self._api_key = resolved_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self._api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """
        Executes generation and returns a typed GenerationResult.
        Traps all API and connection errors gracefully to prevent error leaks into answers.
        """
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )

            # Extract generated text
            content = response.text
            if content is None or not content.strip():
                return GenerationResult(
                    text=None,
                    status="empty_response",
                    error_message="Gemini model returned empty content payload.",
                    error_type="EmptyResponseError",
                    model_name=self.model_name,
                )

            # Extract token usage and finish reason if available
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            finish_reason = None

            if getattr(response, "usage_metadata", None) is not None:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                completion_tokens = getattr(usage, "candidates_token_count", None)
                total_tokens = getattr(usage, "total_token_count", None)

            if getattr(response, "candidates", None) and len(response.candidates) > 0:
                finish_reason = str(getattr(response.candidates[0], "finish_reason", "STOP"))

            return GenerationResult(
                text=content,
                status="success",
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
            )

        except genai_errors.ClientError as exc:
            # 4xx HTTP client errors (auth, bad request, rate limit)
            status_code = getattr(exc, "code", None)
            err_type = "ClientError"
            status = "api_error"
            if status_code == 401 or status_code == 403:
                status = "auth_error"
            elif status_code == 429:
                status = "rate_limited"
            elif status_code == 400:
                status = "bad_request"

            logger.error("Gemini API ClientError (%s): %s", status, str(exc))
            return GenerationResult(
                text=None,
                status=status,
                error_message=f"Gemini API client error: {str(exc)}",
                error_type=err_type,
                model_name=self.model_name,
            )

        except genai_errors.ServerError as exc:
            # 5xx HTTP server errors
            logger.error("Gemini API ServerError: %s", str(exc))
            return GenerationResult(
                text=None,
                status="api_error",
                error_message=f"Gemini API server error: {str(exc)}",
                error_type="ServerError",
                model_name=self.model_name,
            )

        except genai_errors.APIError as exc:
            logger.error("Gemini API general APIError: %s", str(exc))
            return GenerationResult(
                text=None,
                status="api_error",
                error_message=f"Gemini API error: {str(exc)}",
                error_type="APIError",
                model_name=self.model_name,
            )

        except Exception as exc:
            logger.error("Unexpected error during Gemini generation: %s", str(exc))
            return GenerationResult(
                text=None,
                status="api_error",
                error_message=f"Unexpected generation failure: {str(exc)}",
                error_type=type(exc).__name__,
                model_name=self.model_name,
            )

    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Executes generation and returns the generated text string.
        Raises GenerationError on API failure rather than returning raw error strings.
        """
        result = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not result.is_success or result.text is None:
            raise GenerationError(
                message=result.error_message or "Generation failed without specific error.",
                error_type=result.error_type or result.status,
            )

        return result.text


# Alias for explicit naming
GeminiGenerationClient = LegalGenerationClient

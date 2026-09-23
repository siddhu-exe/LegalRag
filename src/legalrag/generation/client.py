"""
Groq generation client interface for the LegalRAG generation layer.

Implements grounded legal QA generation via the official groq SDK
with greedy decoding (temperature=0.0) and explicit error shielding.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import groq
    from groq import Groq
    from groq import (
        APIError as GroqAPIError,
        APIConnectionError as GroqAPIConnectionError,
        RateLimitError as GroqRateLimitError,
        AuthenticationError as GroqAuthenticationError,
        BadRequestError as GroqBadRequestError,
        InternalServerError as GroqInternalServerError,
    )
except ImportError:
    groq = None  # type: ignore
    Groq = None  # type: ignore
    GroqAPIError = None  # type: ignore
    GroqAPIConnectionError = None  # type: ignore
    GroqRateLimitError = None  # type: ignore
    GroqAuthenticationError = None  # type: ignore
    GroqBadRequestError = None  # type: ignore
    GroqInternalServerError = None  # type: ignore


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
    Generation client for executing grounded legal QA against Groq models.
    Provides strict error isolation to prevent upstream error strings from leaking into answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "qwen/qwen3.8-27b",
        request_timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ):
        if Groq is None:
            raise ImportError(
                "groq package is required for LegalGenerationClient. "
                "Install it via: pip install groq"
            )

        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "Groq API key is required. Set 'GROQ_API_KEY' environment variable "
                "or pass 'api_key' to LegalGenerationClient."
            )

        self._api_key = resolved_key
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_max = retry_backoff_max
        # Bounded, explicit retries are implemented in this client; disable SDK-internal
        # retries so the total number of provider calls stays predictable.
        self.client = Groq(
            api_key=self._api_key,
            timeout=request_timeout,
            max_retries=0,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """
        Executes generation with an explicit request timeout and bounded retries for
        transient provider failures.

        Returns a successful GenerationResult on success. Provider failures are converted
        into a controlled GenerationError (surfaced as HTTP 502 at the API boundary) so
        raw provider errors and credentials never leak into responses.
        """
        attempts = max(1, self.max_retries + 1)
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.request_timeout,
                )
                return self._parse_response(response)
            except GenerationError:
                raise
            except Exception as exc:  # noqa: BLE001 - classify provider failures
                if self._is_retryable(exc) and attempt < attempts:
                    delay = min(
                        self.retry_backoff_base * (2 ** (attempt - 1)),
                        self.retry_backoff_max,
                    )
                    logger.warning(
                        "Transient Groq failure (attempt %d/%d, %s); retrying in %.2fs.",
                        attempt,
                        attempts,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise self._to_generation_error(exc) from exc

        raise self._to_generation_error(last_error)

    def _parse_response(self, response: Any) -> GenerationResult:
        """Extracts generated text and token usage, raising on empty payloads."""
        content = None
        finish_reason = None
        if getattr(response, "choices", None):
            choice = response.choices[0]
            if getattr(choice, "message", None) is not None:
                content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)

        if content is None or not str(content).strip():
            raise GenerationError(
                message="Generation provider returned an empty response.",
                status_code=502,
                error_type="EmptyResponseError",
            )

        prompt_tokens = completion_tokens = total_tokens = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        return GenerationResult(
            text=str(content),
            status="success",
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=str(finish_reason) if finish_reason else None,
        )

    @staticmethod
    def _retryable_exception_types() -> tuple:
        """Returns the provider exception classes that are safe to retry."""
        return tuple(
            candidate
            for candidate in (
                GroqRateLimitError,
                GroqAPIConnectionError,
                GroqInternalServerError,
            )
            if candidate is not None
        )

    def _is_retryable(self, exc: Exception) -> bool:
        """Classifies provider failures safe to retry a bounded number of times."""
        retryable = self._retryable_exception_types()
        if retryable and isinstance(exc, retryable):
            return True
        if GroqAPIError is not None and isinstance(exc, GroqAPIError):
            status = getattr(exc, "status_code", None)
            if status in (408, 409, 429):
                return True
            if isinstance(status, int) and status >= 500:
                return True
        return False

    def _to_generation_error(self, exc: Optional[Exception]) -> GenerationError:
        """Maps a provider exception to a sanitized, controlled application exception."""
        logger.error(
            "Groq generation failed (%s).",
            type(exc).__name__ if exc is not None else "unknown",
        )
        error_type = type(exc).__name__ if exc is not None else "GenerationError"
        message = "Generation provider request failed."

        if exc is not None:
            if GroqAuthenticationError is not None and isinstance(exc, GroqAuthenticationError):
                message = "Generation provider authentication failed."
            elif GroqRateLimitError is not None and isinstance(exc, GroqRateLimitError):
                message = "Generation provider rate limit exceeded."
            elif GroqBadRequestError is not None and isinstance(exc, GroqBadRequestError):
                message = "Generation provider rejected the request."
            elif GroqAPIConnectionError is not None and isinstance(exc, GroqAPIConnectionError):
                message = "Could not reach the generation provider."
            elif GroqInternalServerError is not None and isinstance(exc, GroqInternalServerError):
                message = "Generation provider returned a server error."

        return GenerationError(message=message, status_code=502, error_type=error_type)

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


# Aliases for explicit naming
GroqGenerationClient = LegalGenerationClient

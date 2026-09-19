"""
Groq generation client interface for the LegalRAG generation layer.

Implements grounded legal QA generation via the official groq SDK
with greedy decoding (temperature=0.0) and explicit error shielding.
"""

import os
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
        model_name: str = "llama-3.3-70b-versatile",
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
        self.client = Groq(api_key=self._api_key)

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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract generated content
            content = None
            finish_reason = None
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if getattr(choice, "message", None) is not None:
                    content = choice.message.content
                finish_reason = getattr(choice, "finish_reason", None)

            if content is None or not content.strip():
                return GenerationResult(
                    text=None,
                    status="empty_response",
                    error_message="Groq model returned empty content payload.",
                    error_type="EmptyResponseError",
                    model_name=self.model_name,
                )

            # Extract token usage metadata if available
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            if getattr(response, "usage", None) is not None:
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            return GenerationResult(
                text=content,
                status="success",
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=str(finish_reason) if finish_reason else None,
            )

        except GroqAuthenticationError as exc:
            logger.error("Groq API AuthenticationError: %s", str(exc))
            return GenerationResult(
                text=None,
                status="auth_error",
                error_message="Groq API authentication failed.",
                error_type="AuthenticationError",
                model_name=self.model_name,
            )

        except GroqRateLimitError as exc:
            logger.error("Groq API RateLimitError: %s", str(exc))
            return GenerationResult(
                text=None,
                status="rate_limited",
                error_message="Groq API rate limit exceeded.",
                error_type="RateLimitError",
                model_name=self.model_name,
            )

        except GroqBadRequestError as exc:
            logger.error("Groq API BadRequestError: %s", str(exc))
            return GenerationResult(
                text=None,
                status="bad_request",
                error_message=f"Groq API bad request: {str(exc)}",
                error_type="BadRequestError",
                model_name=self.model_name,
            )

        except GroqInternalServerError as exc:
            logger.error("Groq API ServerError: %s", str(exc))
            return GenerationResult(
                text=None,
                status="api_error",
                error_message=f"Groq API server error: {str(exc)}",
                error_type="InternalServerError",
                model_name=self.model_name,
            )

        except GroqAPIConnectionError as exc:
            logger.error("Groq API ConnectionError: %s", str(exc))
            return GenerationResult(
                text=None,
                status="api_error",
                error_message=f"Groq API connection error: {str(exc)}",
                error_type="APIConnectionError",
                model_name=self.model_name,
            )

        except GroqAPIError as exc:
            logger.error("Groq API general APIError: %s", str(exc))
            return GenerationResult(
                text=None,
                status="api_error",
                error_message=f"Groq API error: {str(exc)}",
                error_type="APIError",
                model_name=self.model_name,
            )

        except Exception as exc:
            logger.error("Unexpected error during Groq generation: %s", str(exc))
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


# Aliases for explicit naming
GroqGenerationClient = LegalGenerationClient

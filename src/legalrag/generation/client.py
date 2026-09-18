"""
OpenAI-compatible client interface for LegalRAG generation layer.
Routes requests via OmniRoute / configured OpenAI-compatible gateway with greedy decoding (temperature=0.0).

Includes graceful error shielding to prevent API error leakage into generation outputs.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Generator, AsyncGenerator

logger = logging.getLogger(__name__)

try:
    from openai import (
        OpenAI,
        AsyncOpenAI,
        OpenAIError,
        APIError,
        APIConnectionError,
        RateLimitError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
    )
except ImportError:
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore
    APIError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    BadRequestError = Exception  # type: ignore
    InternalServerError = Exception  # type: ignore


class GenerationError(Exception):
    """Custom exception raised when upstream LLM API generation fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, error_type: str = "api_error"):
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
    status: str = "success"  # "success", "api_error", "rate_limited", "auth_error", "bad_request"
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
    Generation client for executing grounded legal QA against an OpenAI-compatible endpoint.
    Provides strict error isolation to prevent upstream OmniRoute error strings from leaking into answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4-turbo-preview",
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for LegalGenerationClient. Install it via pip install openai."
            )

        self.api_key = api_key or os.getenv("OMNIROUTE_API_KEY") or os.getenv("OPENAI_API_KEY", "dummy-key")
        self.base_url = base_url or os.getenv("OMNIROUTE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.async_client: Optional[Any] = None

    def _get_async_client(self) -> Any:
        if self.async_client is None:
            if AsyncOpenAI is None:
                raise ImportError("AsyncOpenAI requires openai package.")
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self.async_client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """
        Executes generation and returns a typed GenerationResult.
        Traps all API errors gracefully to prevent error leak into downstream evaluation or UI.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            choice = response.choices[0]
            content = choice.message.content

            # Guard against None or empty response
            if content is None:
                return GenerationResult(
                    text=None,
                    status="empty_response",
                    error_message="Model returned empty content payload.",
                    error_type="empty_response",
                    model_name=self.model_name,
                )

            usage = getattr(response, "usage", None)
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else None
            total_tokens = usage.total_tokens if usage else None

            return GenerationResult(
                text=content,
                status="success",
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=getattr(choice, "finish_reason", None),
            )

        except AuthenticationError as e:
            logger.error("Authentication error during generation: %s", str(e))
            return GenerationResult(
                text=None,
                status="auth_error",
                error_message=f"Authentication failed with LLM gateway: {str(e)}",
                error_type="AuthenticationError",
                model_name=self.model_name,
            )
        except RateLimitError as e:
            logger.error("Rate limit exceeded on LLM gateway: %s", str(e))
            return GenerationResult(
                text=None,
                status="rate_limited",
                error_message=f"Rate limit exceeded: {str(e)}",
                error_type="RateLimitError",
                model_name=self.model_name,
            )
        except BadRequestError as e:
            logger.error("Bad request to LLM gateway: %s", str(e))
            return GenerationResult(
                text=None,
                status="bad_request",
                error_message=f"Bad request payload to model: {str(e)}",
                error_type="BadRequestError",
                model_name=self.model_name,
            )
        except (APIConnectionError, InternalServerError, APIError) as e:
            logger.error("API gateway / connection error during generation: %s", str(e))
            return GenerationResult(
                text=None,
                status="api_error",
                error_message=f"Upstream gateway error: {str(e)}",
                error_type=type(e).__name__,
                model_name=self.model_name,
            )
        except Exception as e:
            logger.error("Unexpected error during LLM generation: %s", str(e))
            return GenerationResult(
                text=None,
                status="api_error",
                error_message=f"Unexpected generation failure: {str(e)}",
                error_type=type(e).__name__,
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
        Legacy interface for backward compatibility.
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

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming generation generator for real-time token delivery over SSE.
        """
        client = self._get_async_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

        except Exception as e:
            logger.error("Error during streaming generation: %s", str(e))
            raise GenerationError(
                message=f"Streaming generation failure: {str(e)}",
                error_type=type(e).__name__,
            )

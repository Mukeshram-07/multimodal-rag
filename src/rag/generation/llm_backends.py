"""
LLM backend implementations for the Multimodal RAG System.

Supports two providers via environment variables:

  LLM_PROVIDER=groq   (default)
    LLM_API_BASE=https://api.groq.com/openai/v1
    LLM_MODEL=llama-3.3-70b-versatile
    LLM_API_KEY=gsk_...

  LLM_PROVIDER=ollama  (local dev)
    LLM_API_BASE=http://localhost:11434
    LLM_MODEL=llama3
    LLM_API_KEY=   (not required)

Both providers use the OpenAI-compatible /chat/completions endpoint.
The API key is NEVER exposed to the frontend or any VITE_* variable.
"""

from __future__ import annotations

import httpx

from rag.exceptions import GenerationError
from rag.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 120.0


class OpenAICompatibleBackend:
    """
    LLM backend that calls any OpenAI-compatible /chat/completions endpoint.

    Works with:
      - GroqCloud  (https://api.groq.com/openai/v1)
      - Ollama     (http://localhost:11434/v1)
      - OpenAI     (https://api.openai.com/v1)
      - Any other OpenAI-compatible provider

    Args:
        api_base:  Base URL of the API (without trailing slash).
        model:     Model name to request.
        api_key:   Bearer token. Required for Groq/OpenAI; use "" for Ollama.
        provider:  Human-readable provider name for logging ("groq", "ollama", etc.)
        timeout:   HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_base: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
        api_key: str = "",
        provider: str = "groq",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.provider = provider.lower()
        self.timeout = timeout

        # Validate at construction time so startup fails fast with a clear message.
        if self.provider != "ollama" and not self.api_key:
            logger.warning(
                "LLM_API_KEY is not set for provider '%s'. "
                "Generation requests will fail with HTTP 401. "
                "Set LLM_API_KEY in your .env file.",
                self.provider,
            )

    def complete(self, prompt: str) -> str:
        """
        Send a prompt to the LLM via the /chat/completions endpoint.

        Uses the OpenAI chat completions format (messages array) which is
        supported by Groq, OpenAI, and Ollama's /v1 compatibility layer.

        Args:
            prompt: The full prompt string.

        Returns:
            The model's response text.

        Raises:
            GenerationError: On HTTP error, timeout, missing API key, or
                             unexpected response shape.
        """
        # Guard: reject early if API key is required but missing.
        if self.provider != "ollama" and not self.api_key:
            raise GenerationError(
                message=(
                    f"LLM_API_KEY is not configured for provider '{self.provider}'. "
                    "Set LLM_API_KEY in your environment variables."
                ),
                detail="Missing API key",
            )

        url = f"{self.api_base}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "Bearer ollama",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        logger.info(
            "LLM request: provider=%s model=%s url=%s prompt_chars=%d",
            self.provider,
            self.model,
            url,
            len(prompt),
        )

        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise GenerationError(
                message=f"LLM request timed out after {self.timeout}s (provider={self.provider})",
                detail=str(exc),
            ) from exc

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:300]
            if status == 401:
                raise GenerationError(
                    message=(
                        f"LLM authentication failed (HTTP 401) for provider '{self.provider}'. "
                        "Check that LLM_API_KEY is correct."
                    ),
                    detail=body,
                ) from exc
            if status == 429:
                raise GenerationError(
                    message=f"LLM rate limit exceeded (HTTP 429) for provider '{self.provider}'.",
                    detail=body,
                ) from exc
            raise GenerationError(
                message=f"LLM request failed with HTTP {status} (provider={self.provider})",
                detail=body,
            ) from exc

        except httpx.RequestError as exc:
            raise GenerationError(
                message=f"LLM connection error (provider={self.provider}): {exc}",
                detail=str(exc),
            ) from exc

        try:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise GenerationError(
                message="Unexpected LLM response shape",
                detail=str(exc),
            ) from exc

        logger.info(
            "LLM response received: provider=%s model=%s response_chars=%d",
            self.provider,
            self.model,
            len(answer),
        )
        return answer

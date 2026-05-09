"""
LLM backend implementations for the Multimodal RAG System.

Provides ``OpenAICompatibleBackend``, which calls any OpenAI-compatible
HTTP endpoint (including local Ollama) using ``httpx``.

Requirements: 5.6
"""

from __future__ import annotations

import httpx

from rag.exceptions import GenerationError
from rag.logging_config import get_logger

logger = get_logger(__name__)

# Default timeout for LLM HTTP requests (seconds).
_DEFAULT_TIMEOUT = 120.0


class OpenAICompatibleBackend:
    """
    LLM backend that calls an OpenAI-compatible chat completions endpoint.

    Works with local Ollama (``http://localhost:11434/v1``) and any other
    provider that implements the OpenAI chat completions API.

    Args:
        api_base:   Base URL of the API, e.g. ``"http://localhost:11434/v1"``.
        model:      Model name to request, e.g. ``"llama3"``.
        api_key:    API key sent in the ``Authorization`` header.
                    Use ``"ollama"`` for local Ollama instances.
        timeout:    HTTP request timeout in seconds. Defaults to 120.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str = "ollama",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the generated text.

        Uses the ``/chat/completions`` endpoint with a single user message.

        Args:
            prompt: The full prompt string.

        Returns:
            The model's response text.

        Raises:
            GenerationError: On HTTP error, timeout, or unexpected response shape.
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        logger.info(
            "LLM request: model=%s url=%s prompt_chars=%d",
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
                message=f"LLM request timed out after {self.timeout}s",
                detail=str(exc),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                message=f"LLM request failed with HTTP {exc.response.status_code}",
                detail=str(exc),
            ) from exc
        except httpx.RequestError as exc:
            raise GenerationError(
                message=f"LLM request error: {exc}",
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
            "LLM response received: model=%s response_chars=%d",
            self.model,
            len(answer),
        )
        return answer

"""
Generation module for the Multimodal RAG System.

Exports:
  - LLMBackend:              Protocol for any LLM backend.
  - OpenAICompatibleBackend: HTTP client for OpenAI-compatible endpoints (incl. Ollama).
  - ResponseGenerator:       Builds grounded prompts and returns citation-aware responses.
"""

from rag.generation.llm_backends import OpenAICompatibleBackend
from rag.generation.protocols import LLMBackend
from rag.generation.response_generator import ResponseGenerator

__all__ = ["LLMBackend", "OpenAICompatibleBackend", "ResponseGenerator"]

"""
Protocol definitions for the generation layer.

Defines the LLMBackend Protocol so that any object implementing
``complete(prompt: str) -> str`` can be used wherever an LLM is required,
enabling easy substitution in tests and future extensions.

Requirements: 5.6
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """
    Protocol for any LLM backend.

    Any class that implements ``complete`` with the correct signature
    satisfies this protocol without explicit inheritance.
    """

    def complete(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the generated text.

        Args:
            prompt: The full prompt string to send to the model.

        Returns:
            The model's response as a plain string.
        """
        ...

"""LLM API wrapper for Gemini models."""

import os
import time
from typing import Optional, Dict, Any
import google.generativeai as genai


class GeminiModel:
    """Wrapper for Google Gemini API."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 512,
    ):
        """
        Initialize Gemini model.

        Args:
            model_name: Name of the Gemini model
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens in response
        """
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Initialize model
        self.model = genai.GenerativeModel(model_name)

        # Track API calls for debugging
        self.total_calls = 0

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_attempts: int = 3,
    ) -> str:
        """
        Generate text from prompt with retry logic.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            retry_attempts: Number of retry attempts on failure

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_output_tokens

        generation_config = genai.GenerationConfig(
            temperature=temp,
            max_output_tokens=max_tok,
        )

        for attempt in range(retry_attempts):
            try:
                self.total_calls += 1
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                return response.text.strip()
            except Exception as e:
                if attempt < retry_attempts - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Failed to generate after {retry_attempts} attempts: {e}"
                    )

        return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get model usage statistics."""
        return {
            "model_name": self.model_name,
            "total_calls": self.total_calls,
        }

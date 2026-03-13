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
                # [VALIDATOR FIX - Attempt 2]
                # [PROBLEM]: All predictions are "A" - model might be blocked/empty
                # [CAUSE]: response.text might fail if content is blocked or empty
                # [FIX]: Check for blocked content and handle gracefully
                #
                # [OLD CODE]:
                # return response.text.strip()
                #
                # [NEW CODE]:
                # [VALIDATOR FIX - Attempt 4]
                # [PROBLEM]: All predictions are "A" - extraction fallback always triggers
                # [CAUSE]: Response handling needs better logging and alternative text extraction
                # [FIX]: Add comprehensive logging and try multiple ways to extract text
                #
                # [OLD CODE]:
                # (basic blocked content checking without sufficient logging)
                #
                # [NEW CODE]:
                # Check if response was blocked by safety filters
                if not response.candidates:
                    print(
                        f"[WARNING] No candidates in response (likely blocked by safety filters) for prompt: {prompt[:100]}..."
                    )
                    if attempt < retry_attempts - 1:
                        wait_time = 2**attempt
                        time.sleep(wait_time)
                        continue
                    # Instead of returning empty, throw error to make blocking obvious
                    raise RuntimeError(
                        "Response blocked by safety filters after all attempts"
                    )

                candidate = response.candidates[0]

                # Check finish reason with detailed logging
                finish_reason_name = getattr(candidate, "finish_reason", None)
                if finish_reason_name not in [None, 1, "STOP"]:
                    print(
                        f"[WARNING] Unexpected finish_reason: {finish_reason_name} for prompt: {prompt[:100]}..."
                    )

                # Try multiple ways to extract text
                text = None
                try:
                    text = response.text.strip()
                except ValueError as ve:
                    # response.text throws ValueError if there's no text part
                    print(
                        f"[WARNING] response.text failed: {ve}. Trying alternative extraction..."
                    )
                    # Try to extract from parts directly
                    if hasattr(candidate, "content") and hasattr(
                        candidate.content, "parts"
                    ):
                        parts = candidate.content.parts
                        if parts and hasattr(parts[0], "text"):
                            text = parts[0].text.strip()
                            print(f"[INFO] Extracted text from parts: {text[:100]}...")
                except AttributeError as ae:
                    print(f"[WARNING] AttributeError accessing response.text: {ae}")

                # Validate we got text
                if not text:
                    print(
                        f"[WARNING] Empty response text after all extraction attempts. Prompt: {prompt[:100]}..."
                    )
                    if attempt < retry_attempts - 1:
                        wait_time = 2**attempt
                        time.sleep(wait_time)
                        continue
                    # Raise error instead of returning empty string
                    raise RuntimeError(
                        "Empty response after all attempts and extraction methods"
                    )

                return text

            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1}/{retry_attempts} failed: {e}")
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

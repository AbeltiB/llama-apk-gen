"""
app/llm/llama3_provider.py
Production-ready Llama3 LLM provider using OpenAI-compatible SDK
"""
import logging
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, RateLimitError, AuthenticationError

from .base import BaseLLMProvider, LLMResponse, LLMMessage, LLMProvider

logger = logging.getLogger(__name__)


class Llama3Provider(BaseLLMProvider):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider_name = LLMProvider.LLAMA3

        self.api_url = config.get("llama3_api_url", "https://fastchat.ideeza.com/v1")
        self.model = config.get("llama3_model", "llama-3")
        self.api_key = config.get("llama3_api_key")

        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.request_timeout = config.get("request_timeout", 45.0)

        # Circuit breaker
        self.failure_count = 0
        self.max_failures = config.get("max_failures", 5)
        self.circuit_reset_time = config.get("circuit_reset_seconds", 300)
        self.circuit_tripped_time = None
        self.circuit_open = False

        # Stats
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        if not self.api_key:
            raise ValueError("Llama3 API key (FASTCHAT_API_KEY) is required")

        # Build AsyncOpenAI client — base_url strips trailing /chat/completions if present
        base_url = self.api_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.request_timeout,
            max_retries=0,  # We handle retries ourselves
        )

        logger.info(
            f"Llama3 provider initialized (OpenAI SDK): model={self.model}, "
            f"base_url={base_url}, timeout={self.request_timeout}s"
        )

    # ------------------------------------------------------------------ #
    # Circuit breaker helpers (unchanged from original)
    # ------------------------------------------------------------------ #

    def _check_circuit_breaker(self) -> bool:
        if not self.circuit_open:
            return True
        if self.circuit_tripped_time:
            elapsed = (datetime.now() - self.circuit_tripped_time).total_seconds()
            if elapsed > self.circuit_reset_time:
                logger.info("Circuit breaker resetting")
                self.circuit_open = False
                self.failure_count = 0
                self.circuit_tripped_time = None
                return True
        return False

    def _trip_circuit_breaker(self):
        self.circuit_open = True
        self.circuit_tripped_time = datetime.now()
        logger.error(f"Circuit breaker tripped after {self.failure_count} failures")

    # ------------------------------------------------------------------ #
    # Main generate
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        self.total_requests += 1

        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open — Llama3 unavailable")

        if not self.validate_messages(messages):
            raise ValueError("Invalid messages format")

        temperature = max(0.0, min(2.0, temperature))
        formatted = self.format_messages(messages)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._make_request(formatted, temperature, max_tokens, attempt, **kwargs)
                self.successful_requests += 1
                self.failure_count = 0
                logger.info(
                    f"Llama3 success: tokens={response.tokens_used}, "
                    f"valid_json={response.is_valid_json}"
                )
                return response

            except AuthenticationError as e:
                self.failure_count += 1
                self.failed_requests += 1
                self._trip_circuit_breaker()
                raise Exception("Llama3 authentication failed: check FASTCHAT_API_KEY") from e

            except RateLimitError as e:
                last_error = e
                self.failed_requests += 1
                self.failure_count += 1
                logger.warning(f"Llama3 rate limited (attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * 2)
                    continue

            except APITimeoutError as e:
                last_error = e
                self.failed_requests += 1
                self.failure_count += 1
                logger.warning(f"Llama3 timeout (attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue

            except APIStatusError as e:
                last_error = e
                self.failed_requests += 1
                self.failure_count += 1
                status = e.status_code
                if 400 <= status < 500:
                    logger.error(f"Llama3 client error {status}: {e.message}")
                    break
                logger.warning(f"Llama3 server error {status} (attempt {attempt})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue

            except Exception as e:
                last_error = e
                self.failed_requests += 1
                self.failure_count += 1
                logger.error(f"Llama3 unexpected error (attempt {attempt}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue

        if self.failure_count >= self.max_failures:
            self._trip_circuit_breaker()

        raise Exception(f"Llama3 generation failed after {self.max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------ #
    # Internal request — uses AsyncOpenAI SDK
    # ------------------------------------------------------------------ #

    async def _make_request(
        self,
        formatted_messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        attempt: int,
        **kwargs,
    ) -> LLMResponse:
        start = datetime.now()

        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens or self.max_tokens_default,
            top_p=kwargs.get("top_p", 0.95),
            frequency_penalty=kwargs.get("frequency_penalty", 0),
            presence_penalty=kwargs.get("presence_penalty", 0),
        )

        response_time = (datetime.now() - start).total_seconds()
        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = completion.usage

        llm_response = LLMResponse(
            content=content,
            provider=self.provider_name,
            tokens_used=usage.total_tokens if usage else None,
            finish_reason=choice.finish_reason,
            model=completion.model,
            metadata={
                "attempt": attempt,
                "response_time": response_time,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "id": completion.id,
            },
        )

        logger.info(
            f"Llama3 response (attempt {attempt}): "
            f"tokens={usage.total_tokens if usage else '?'}, "
            f"time={response_time:.2f}s, valid_json={llm_response.is_valid_json}"
        )
        return llm_response

    # ------------------------------------------------------------------ #
    # Health check
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        if self.circuit_open:
            return False
        try:
            completion = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Respond with 'OK'"},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            result = completion.choices[0].message.content or ""
            healthy = "OK" in result.upper()
            logger.info(f"Llama3 health check: {'PASSED' if healthy else 'UNEXPECTED RESPONSE'}")
            return healthy
        except Exception as e:
            logger.warning(f"Llama3 health check FAILED: {e}")
            self.failure_count += 1
            if self.failure_count >= self.max_failures:
                self._trip_circuit_breaker()
            return False

    def get_provider_type(self) -> LLMProvider:
        return self.provider_name

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name.value,
            "model": self.model,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "failure_count": self.failure_count,
            "circuit_open": self.circuit_open,
            "success_rate": (
                self.successful_requests / self.total_requests * 100
                if self.total_requests > 0 else 0
            ),
        }

    def reset_circuit(self):
        logger.info("Manually resetting circuit breaker")
        self.circuit_open = False
        self.failure_count = 0
        self.circuit_tripped_time = None
import httpx
import openai
import structlog
from openai import AsyncOpenAI

log = structlog.get_logger(__name__)


class LLMError(Exception):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMOutOfTokensError(LLMError):
    pass


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 90.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        # No retries: a retried 90-second timeout leaves the user waiting for three minutes.
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("llm request timed out") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"cannot connect to llm: {exc}") from exc
        except openai.APIStatusError as exc:
            log.warning("llm returned error", status=exc.status_code, body=str(exc.body)[:500])
            raise LLMError(f"llm error {exc.status_code}") from exc
        except openai.OpenAIError as exc:
            log.warning("llm call failed", exc_info=True)
            raise LLMError(f"llm call failed: {exc}") from exc

        if not response.choices:
            raise LLMError("llm returned no choices")
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise LLMError("llm returned empty answer")
        return content

    async def close(self) -> None:
        await self._client.close()

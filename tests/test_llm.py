import httpx
import openai
import pytest

from askdocs.rag.llm import LLMClient, LLMError, LLMTimeoutError


def completion(content: str | None, finish_reason: str = "stop") -> dict:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def client_with(handler) -> LLMClient:
    transport = httpx.MockTransport(handler)
    return LLMClient(
        base_url="http://llm.test/v1",
        api_key="key",
        model="test-model",
        http_client=httpx.AsyncClient(transport=transport),
    )


async def test_returns_content_and_sends_model():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=completion("  Ответ [1].  "))

    client = client_with(handler)
    assert await client.complete([{"role": "user", "content": "вопрос"}]) == "Ответ [1]."
    assert '"model":"test-model"' in seen["body"].replace(" ", "")
    assert seen["auth"] == "Bearer key"


async def test_timeout_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeoutError):
        await client_with(handler).complete([])
    assert len(calls) == 1


async def test_connection_refused():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError):
        await client_with(handler).complete([])


@pytest.mark.parametrize("status", [500, 429, 404])
async def test_http_errors(status):
    client = client_with(lambda r: httpx.Response(status, json={"error": {"message": "x"}}))
    with pytest.raises(LLMError) as info:
        await client.complete([])
    assert not isinstance(info.value, LLMTimeoutError)


async def test_other_openai_errors(monkeypatch):
    client = client_with(lambda r: httpx.Response(200, json=completion("x")))

    async def broken(**kwargs):
        raise openai.OpenAIError("unexpected")

    monkeypatch.setattr(client._client.chat.completions, "create", broken)
    with pytest.raises(LLMError):
        await client.complete([])

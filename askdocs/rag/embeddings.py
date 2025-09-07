import asyncio
from collections.abc import Sequence
from typing import Protocol

import structlog

log = structlog.get_logger(__name__)


class Embedder(Protocol):
    dimension: int

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        from sentence_transformers import SentenceTransformer

        log.info("loading embedding model", model=model_name, device=device or "auto")
        try:
            # Skip the round trip to the Hugging Face Hub when the model is already cached.
            self._model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except OSError:
            self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self.dimension: int = self._model.get_sentence_embedding_dimension()

    @classmethod
    async def load(cls, model_name: str, **kwargs) -> "SentenceTransformerEmbedder":
        return await asyncio.to_thread(cls, model_name, **kwargs)

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._encode([self._passage_prefix + t for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._encode([self._query_prefix + text])
        return vectors[0]

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        array = await asyncio.to_thread(
            self._model.encode,
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return array.tolist()

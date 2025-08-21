import pytest
from helpers import FakeEmbedder, FakeLLM
from qdrant_client import QdrantClient

from askdocs.config import Settings
from askdocs.rag.service import RagService
from askdocs.rag.store import ChunkStore


@pytest.fixture
def qdrant():
    client = QdrantClient(location=":memory:")
    yield client
    client.close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def store(qdrant, embedder) -> ChunkStore:
    store = ChunkStore(qdrant, "test_chunks", embedder.dimension)
    await store.setup(create_indexes=False)
    yield store
    store.close()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        bot_token="42:TEST",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
        fetch_k=10,
        score_threshold=0.2,
        # Bag-of-words scores of FakeEmbedder spread widely; the margin has its own test.
        score_margin=1.0,
        max_context_chars=2000,
        max_docs_per_chat=3,
        max_chunks_per_doc=50,
    )


@pytest.fixture
def rag(store, embedder, llm, settings) -> RagService:
    return RagService(store, embedder, llm, settings)

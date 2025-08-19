import asyncio
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import structlog
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from askdocs.rag.parsing import Section

log = structlog.get_logger(__name__)

_READY = qm.FieldCondition(key="ready", match=qm.MatchValue(value=True))


class StoreConfigError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    doc_id: str
    file_name: str
    chunk_count: int
    sha256: str
    uploaded_at: datetime


@dataclass(frozen=True, slots=True)
class Hit:
    text: str
    file_name: str
    page: int | None
    heading: str | None
    score: float
    vector: list[float]


class ChunkStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int):
        self._client = client
        self._collection = collection
        self._dimension = dimension
        # qdrant-client's local mode isn't thread-safe; one worker serializes calls.
        # Server mode needs the order too: a cancelled upload is rolled back only
        # after the upsert it left running in the thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qdrant")

    async def _call(self, method, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, partial(method, self._collection, *args, **kwargs)
        )

    def close(self) -> None:
        self._executor.shutdown()
        self._client.close()

    async def setup(self, *, create_indexes: bool = True) -> None:
        if await self._call(self._client.collection_exists):
            info = await self._call(self._client.get_collection)
            size = info.config.params.vectors.size
            if size != self._dimension:
                raise StoreConfigError(
                    f"collection {self._collection!r} has vector size {size}, "
                    f"embedding model produces {self._dimension}; "
                    "use another QDRANT_COLLECTION or drop the old one"
                )
        else:
            log.info("creating collection", collection=self._collection, dim=self._dimension)
            await self._call(
                self._client.create_collection,
                vectors_config=qm.VectorParams(size=self._dimension, distance=qm.Distance.COSINE),
            )
        # embedded mode ignores payload indexes and warns about them
        if create_indexes:
            for field, schema in (
                ("chat_id", qm.PayloadSchemaType.INTEGER),
                ("doc_id", qm.PayloadSchemaType.KEYWORD),
                ("chunk_index", qm.PayloadSchemaType.INTEGER),
                ("sha256", qm.PayloadSchemaType.KEYWORD),
                ("ready", qm.PayloadSchemaType.BOOL),
            ):
                await self._call(self._client.create_payload_index, field, schema)
        await self._drop_incomplete()

    async def _drop_incomplete(self) -> None:
        # An interrupted upload or delete always leaves unready points behind:
        # mark_ready flips chunk 0 last, and _delete unmarks before deleting.
        records = await self._scroll_all(qm.Filter(must_not=[_READY]), with_payload=["doc_id"])
        for doc_id in {r.payload["doc_id"] for r in records if r.payload.get("doc_id")}:
            log.warning("dropping incomplete document", doc_id=doc_id)
            await self.drop_points(doc_id)

    async def add_document(
        self,
        chat_id: int,
        doc: DocumentInfo,
        chunks: Sequence[Section],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}/{index}")),
                vector=list(vector),
                payload={
                    "chat_id": chat_id,
                    "doc_id": doc.doc_id,
                    "file_name": doc.file_name,
                    "sha256": doc.sha256,
                    "uploaded_at": doc.uploaded_at.isoformat(),
                    "chunk_index": index,
                    "chunk_count": doc.chunk_count,
                    "page": chunk.page,
                    "heading": chunk.heading,
                    "text": chunk.text,
                    "ready": False,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for start in range(0, len(points), 256):
            await self._call(self._client.upsert, points[start : start + 256], wait=True)

    async def mark_ready(self, doc_id: str) -> None:
        # chunk 0 is what lists the document, so it goes last
        rest = qm.Filter(must=_doc_filter(doc_id).must, must_not=[_index_is(0)])
        first = qm.Filter(must=[*_doc_filter(doc_id).must, _index_is(0)])
        for flt in (rest, first):
            await self._call(self._client.set_payload, {"ready": True}, points=flt, wait=True)

    async def drop_points(self, doc_id: str) -> None:
        await self._delete(_doc_filter(doc_id))

    async def search(
        self,
        chat_id: int,
        vector: Sequence[float],
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[Hit]:
        response = await self._call(
            self._client.query_points,
            query=list(vector),
            query_filter=_chat_filter(chat_id),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=True,
        )
        return [
            Hit(
                text=p.payload["text"],
                file_name=p.payload["file_name"],
                page=p.payload.get("page"),
                heading=p.payload.get("heading"),
                score=p.score,
                vector=list(p.vector),
            )
            for p in response.points
        ]

    async def list_documents(self, chat_id: int) -> list[DocumentInfo]:
        records = await self._scroll_all(_chat_filter(chat_id, chunk_index=0))
        docs = [_doc_from_payload(r.payload) for r in records]
        return sorted(docs, key=lambda d: d.uploaded_at)

    async def count_documents(self, chat_id: int) -> int:
        result = await self._call(
            self._client.count, count_filter=_chat_filter(chat_id, chunk_index=0), exact=True
        )
        return result.count

    async def find_by_hash(self, chat_id: int, sha256: str) -> DocumentInfo | None:
        flt = _chat_filter(chat_id, chunk_index=0)
        flt.must.append(qm.FieldCondition(key="sha256", match=qm.MatchValue(value=sha256)))
        records, _ = await self._call(self._client.scroll, scroll_filter=flt, limit=1)
        return _doc_from_payload(records[0].payload) if records else None

    async def delete_document(self, chat_id: int, doc_id: str) -> bool:
        flt = _chat_filter(chat_id)
        flt.must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
        found = await self._call(self._client.count, count_filter=flt, exact=True)
        if not found.count:
            return False
        await self.drop_points(doc_id)
        return True

    async def clear(self, chat_id: int) -> int:
        removed = await self.count_documents(chat_id)
        await self._delete(qm.Filter(must=[_chat_condition(chat_id)]))
        return removed

    async def _delete(self, flt: qm.Filter) -> None:
        # Embedded qdrant deletes point by point. Unmark first, so that a crash
        # halfway leaves points that are hidden and dropped on the next start.
        await self._call(self._client.set_payload, {"ready": False}, points=flt, wait=True)
        await self._call(self._client.delete, points_selector=qm.FilterSelector(filter=flt))

    async def move_chat(self, old_chat_id: int, new_chat_id: int) -> None:
        await self._call(
            self._client.set_payload,
            {"chat_id": new_chat_id},
            points=qm.Filter(must=[_chat_condition(old_chat_id)]),
            wait=True,
        )

    async def _scroll_all(self, flt: qm.Filter | None, with_payload=True) -> list[qm.Record]:
        records: list[qm.Record] = []
        offset = None
        while True:
            batch, offset = await self._call(
                self._client.scroll,
                scroll_filter=flt,
                limit=1024,
                offset=offset,
                with_payload=with_payload,
            )
            records.extend(batch)
            if offset is None:
                return records


def _chat_condition(chat_id: int) -> qm.FieldCondition:
    return qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id))


def _index_is(index: int) -> qm.FieldCondition:
    return qm.FieldCondition(key="chunk_index", match=qm.MatchValue(value=index))


def _doc_filter(doc_id: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])


def _chat_filter(chat_id: int, chunk_index: int | None = None) -> qm.Filter:
    must = [_chat_condition(chat_id), _READY]
    if chunk_index is not None:
        must.append(_index_is(chunk_index))
    return qm.Filter(must=must)


def _doc_from_payload(payload: dict) -> DocumentInfo:
    return DocumentInfo(
        doc_id=payload["doc_id"],
        file_name=payload["file_name"],
        chunk_count=payload["chunk_count"],
        sha256=payload["sha256"],
        uploaded_at=datetime.fromisoformat(payload["uploaded_at"]),
    )

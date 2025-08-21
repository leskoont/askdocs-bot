import asyncio
import contextlib
import hashlib
import re
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import structlog

from askdocs.config import Settings
from askdocs.rag import prompts
from askdocs.rag.chunking import chunk_sections
from askdocs.rag.embeddings import Embedder
from askdocs.rag.parsing import DocumentTooLargeError, EmptyDocumentError, Section, parse_document
from askdocs.rag.retrieval import fit_to_budget, mmr
from askdocs.rag.store import ChunkStore, DocumentInfo, Hit

log = structlog.get_logger(__name__)

_CITATION = re.compile(r"\[(\d{1,2}(?:\s*[-–,]\s*\d{1,2})*)\]")
_EMBED_BATCH = 64

Progress = Callable[[float], Awaitable[None]]


class DuplicateDocumentError(Exception):
    def __init__(self, existing: DocumentInfo):
        super().__init__(f"already uploaded as {existing.file_name}")
        self.existing = existing


class DocumentLimitError(Exception):
    def __init__(self, limit: int):
        super().__init__(f"document limit reached: {limit}")
        self.limit = limit


class ChatBusyError(Exception):
    pass


class NoDocumentsError(Exception):
    pass


class Completion(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True, slots=True)
class Source:
    number: int
    file_name: str
    page: int | None
    heading: str | None

    @property
    def location(self) -> str:
        return source_location(self.page, self.heading)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: list[Source]
    found: bool


def source_location(page: int | None, heading: str | None) -> str:
    parts = []
    if page is not None:
        parts.append(f"стр. {page}")
    if heading:
        parts.append(f"раздел «{heading}»")
    return ", ".join(parts)


class RagService:
    def __init__(self, store: ChunkStore, embedder: Embedder, llm: Completion, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.settings = settings
        self._chat_locks: weakref.WeakValueDictionary[int, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._uploads = asyncio.Semaphore(settings.max_concurrent_uploads)
        self._answers = asyncio.Semaphore(settings.max_concurrent_answers)

    def _lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = self._chat_locks[chat_id] = asyncio.Lock()
        return lock

    async def add_document(
        self, chat_id: int, file_name: str, data: bytes, progress: Progress | None = None
    ) -> DocumentInfo:
        s = self.settings
        sha256 = hashlib.sha256(data).hexdigest()

        async with self._lock(chat_id):
            existing = await self.store.find_by_hash(chat_id, sha256)
            if existing is not None:
                raise DuplicateDocumentError(existing)
            docs = await self.store.list_documents(chat_id)
            replaced = [d for d in docs if d.file_name == file_name]
            if len(docs) - len(replaced) >= s.max_docs_per_chat:
                raise DocumentLimitError(s.max_docs_per_chat)

            async with self._uploads:
                started = time.perf_counter()
                chunks = await asyncio.to_thread(self._chunk, file_name, data)
                vectors = await self._embed(chunks, progress)
                doc = DocumentInfo(
                    doc_id=uuid.uuid4().hex,
                    file_name=file_name,
                    chunk_count=len(chunks),
                    sha256=sha256,
                    uploaded_at=datetime.now(UTC),
                )
                try:
                    await self.store.add_document(chat_id, doc, chunks, vectors)
                    await self.store.mark_ready(doc.doc_id)
                except BaseException:
                    # shield: runs even when the upload task is cancelled
                    with contextlib.suppress(Exception):
                        await asyncio.shield(self.store.drop_points(doc.doc_id))
                    raise
                for old in replaced:
                    try:
                        await self.store.drop_points(old.doc_id)
                    except Exception:
                        # the new version is already indexed, don't fail the upload over this
                        log.warning("old version not removed", doc_id=old.doc_id, exc_info=True)

        log.info(
            "document indexed",
            chat_id=chat_id,
            doc_id=doc.doc_id,
            file_name=file_name,
            size=len(data),
            chunks=len(chunks),
            seconds=round(time.perf_counter() - started, 2),
        )
        return doc

    def _chunk(self, file_name: str, data: bytes) -> list[Section]:
        s = self.settings
        sections = parse_document(file_name, data, s.max_chunks_per_doc * s.chunk_size)
        chunks = chunk_sections(sections, s.chunk_size, s.chunk_overlap, s.max_chunks_per_doc)
        if not chunks:
            raise EmptyDocumentError("no chunks produced")
        if len(chunks) > s.max_chunks_per_doc:
            raise DocumentTooLargeError(f"more than {s.max_chunks_per_doc} chunks")
        return chunks

    async def _embed(self, chunks: list[Section], progress: Progress | None) -> list[list[float]]:
        passages = [f"{c.heading}\n{c.text}" if c.heading else c.text for c in chunks]
        vectors: list[list[float]] = []
        for start in range(0, len(passages), _EMBED_BATCH):
            vectors.extend(
                await self.embedder.embed_passages(passages[start : start + _EMBED_BATCH])
            )
            if progress is not None:
                await progress(len(vectors) / len(passages))
        return vectors

    async def answer(self, chat_id: int, question: str) -> Answer:
        s = self.settings
        if not await self.store.count_documents(chat_id):
            raise NoDocumentsError

        async with self._answers:
            query_vector = await self.embedder.embed_query(question)
            hits = await self.store.search(
                chat_id, query_vector, limit=s.fetch_k, score_threshold=s.score_threshold
            )
            log.info(
                "retrieved",
                chat_id=chat_id,
                hits=len(hits),
                best_score=round(hits[0].score, 3) if hits else None,
            )
            if not hits:
                return Answer(prompts.NO_ANSWER, [], found=False)

            # e5 scores sit in a narrow band, so a fixed threshold alone lets weak neighbours in.
            hits = [h for h in hits if h.score >= hits[0].score - s.score_margin]
            selected = mmr(query_vector, hits, s.top_k, s.mmr_lambda)
            context = fit_to_budget(selected, s.max_context_chars)

            messages = prompts.build_messages(question, [(_label(h), h.text) for h in context])
            text = await self.llm.complete(messages)
        return _make_answer(text, context)

    async def list_documents(self, chat_id: int) -> list[DocumentInfo]:
        return await self.store.list_documents(chat_id)

    async def delete_document(self, chat_id: int, doc_id: str) -> bool:
        async with self._free_lock(chat_id):
            return await self.store.delete_document(chat_id, doc_id)

    async def clear(self, chat_id: int) -> int:
        async with self._free_lock(chat_id):
            return await self.store.clear(chat_id)

    async def move_chat(self, old_chat_id: int, new_chat_id: int) -> None:
        async with self._lock(old_chat_id):
            await self.store.move_chat(old_chat_id, new_chat_id)

    def _free_lock(self, chat_id: int) -> asyncio.Lock:
        # an upload can hold the lock for minutes, a button press can't wait that long
        lock = self._lock(chat_id)
        if lock.locked():
            raise ChatBusyError
        return lock


def _label(hit: Hit) -> str:
    location = source_location(hit.page, hit.heading)
    return f"{hit.file_name}, {location}" if location else hit.file_name


def _make_answer(text: str, context: list[Hit]) -> Answer:
    sources = [
        Source(number, hit.file_name, hit.page, hit.heading)
        for number, hit in enumerate(context, start=1)
    ]
    cited = _cited_numbers(text)
    cited_sources = [s for s in sources if s.number in cited]
    if cited_sources:
        return Answer(text, cited_sources, found=True)
    if _is_no_answer(text):
        return Answer(text, [], found=False)
    return Answer(text, sources[:3], found=True)


def _cited_numbers(text: str) -> set[int]:
    numbers = set()
    for group in _CITATION.findall(text):
        for part in group.split(","):
            # [1-2-3] or a date like [12-01-26] is not a citation
            match = re.fullmatch(r"(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?", part.strip())
            if match is None:
                continue
            start = int(match[1])
            end = int(match[2]) if match[2] else start
            numbers.update(range(start, end + 1))
    return numbers


def _is_no_answer(text: str) -> bool:
    normalized = text.strip().strip("«»\"' .").lower()
    for phrase in (prompts.NO_ANSWER, prompts.NO_ANSWER_EN):
        phrase = phrase.rstrip(".").lower()
        # "нет ответа. Но в разделе 5 сказано..." is still an answer
        if normalized.startswith(phrase) and len(normalized) <= len(phrase) + 40:
            return True
    return False

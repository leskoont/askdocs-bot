import re
from collections.abc import Iterable, Iterator

from askdocs.rag.parsing import Section

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_SPACES = re.compile(r"[ \t ]+")


def chunk_sections(
    sections: Iterable[Section], size: int, overlap: int, limit: int | None = None
) -> list[Section]:
    """Stops after limit + 1 chunks: the caller only needs to know the document is too long."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    for section in sections:
        for text in split_text(section.text, size, overlap):
            chunks.append(Section(text, page=section.page, heading=section.heading))
            if limit is not None and len(chunks) > limit:
                return chunks
    return chunks


def split_text(text: str, size: int, overlap: int) -> Iterator[str]:
    # Pieces of an oversized paragraph leave room for the overlap prepended to them.
    piece_size = max(size - overlap - 2, size // 2) if overlap else size
    current: list[str] = []
    length = 0

    for unit in _units(text, size, piece_size):
        added = len(unit) + (2 if current else 0)
        if current and length + added > size:
            chunk = "\n\n".join(current)
            yield chunk
            tail = _overlap_tail(chunk, overlap)
            if tail and len(tail) + 2 + len(unit) <= size:
                current, length = [tail], len(tail)
            else:
                current, length = [], 0
            added = len(unit) + (2 if current else 0)
        current.append(unit)
        length += added

    if current:
        yield "\n\n".join(current)


def _units(text: str, size: int, piece_size: int) -> Iterator[str]:
    for paragraph in _paragraphs(text):
        if len(paragraph) <= size:
            yield paragraph
        else:
            yield from _split_long(paragraph, piece_size)


def _paragraphs(text: str) -> list[str]:
    result = []
    for block in _PARAGRAPH_BREAK.split(text.replace("\r\n", "\n")):
        lines = [_SPACES.sub(" ", line).strip() for line in block.split("\n")]
        paragraph = "\n".join(line for line in lines if line)
        if paragraph:
            result.append(paragraph)
    return result


def _split_long(paragraph: str, size: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_words(sentence, size))
            continue
        candidate = f"{current} {sentence}" if current else sentence
        if len(candidate) > size:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_words(text: str, size: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > size:
            if current:
                pieces.append(current)
                current = ""
            last = (len(word) - 1) // size * size
            pieces.extend(word[i : i + size] for i in range(0, last, size))
            word = word[last:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(chunk: str, overlap: int) -> str:
    if overlap <= 0 or len(chunk) <= overlap:
        return ""
    tail = chunk[-overlap:]
    match = _SENTENCE_END.search(tail)
    if match and match.end() < len(tail):
        return tail[match.end() :].strip()
    space = tail.find(" ")
    if space == -1:
        return ""
    return tail[space + 1 :].strip()

import codecs
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")

MIN_TEXT_CHARS = 20
# python-docx keeps every part of the package in memory, images included
MAX_DOCX_UNPACKED = 64 * 2**20

# Single-byte encodings Russian text files still come in (Windows, old Unix, DOS).
_LEGACY_ENCODINGS = ("cp1251", "koi8_r", "cp866")
# The wrong ones turn these frequent lowercase letters into capitals or pseudographics.
_RU_COMMON = frozenset("оеаинтсрвлкмдпу")
_GARBLED = re.compile("[\ufffd\u2500-\u259f]")
_HIGH_BYTES = bytes(range(0x80, 0x100))


class ParseError(Exception):
    pass


class EmptyDocumentError(ParseError):
    def __init__(self, message: str, *, scanned: bool = False):
        super().__init__(message)
        self.scanned = scanned


class DocumentTooLargeError(ParseError):
    def __init__(self, message: str, *, unpacked: bool = False):
        super().__init__(message)
        self.unpacked = unpacked


@dataclass(frozen=True, slots=True)
class Section:
    text: str
    page: int | None = None
    heading: str | None = None


def file_extension(file_name: str) -> str:
    return PurePath(file_name).suffix.lower()


def is_supported(file_name: str | None) -> bool:
    return file_name is not None and file_extension(file_name) in SUPPORTED_EXTENSIONS


def parse_document(file_name: str, data: bytes, max_chars: int | None = None) -> list[Section]:
    ext = file_extension(file_name)
    if ext == ".pdf":
        sections = _parse_pdf(data, max_chars)
    elif ext == ".docx":
        sections = _parse_docx(data, max_chars)
    elif ext == ".md":
        sections = _parse_markdown(decode_text(data))
    elif ext == ".txt":
        sections = [Section(decode_text(data))]
    else:
        raise ParseError(f"unsupported extension: {ext or 'none'}")

    sections = [s for s in sections if s.text.strip()]
    if sum(len(s.text.split()) for s in sections) == 0:
        raise EmptyDocumentError("document contains no text", scanned=ext == ".pdf")
    return sections


def decode_text(data: bytes) -> str:
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16", errors="replace")
    elif b"\x00" in data[:8192]:
        raise ParseError("binary data in a text file")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("utf-8-sig", errors="replace")
            # A stray broken byte doesn't make a UTF-8 file cp1251. One cp1251 word in Latin
            # text does: there nearly every non-ASCII byte fails to decode.
            bad = text.count("\ufffd")
            non_ascii = len(data) - len(data.translate(None, _HIGH_BYTES))
            if bad > 1 and bad * 10 > non_ascii:
                candidates = [data.decode(enc, errors="replace") for enc in _LEGACY_ENCODINGS]
                text = max(candidates, key=_russian_score)

    sample = text[:8192]
    junk = sum(1 for ch in sample if ch == "\ufffd" or _is_control(ch))
    if junk > len(sample) / 100:
        raise ParseError("text file is binary or in an unknown encoding")
    return text


def _russian_score(text: str) -> int:
    sample = text[:8192]
    return sum(ch in _RU_COMMON for ch in sample) - 2 * len(_GARBLED.findall(sample))


def _is_control(ch: str) -> bool:
    return ch not in "\t\n\r\f" and unicodedata.category(ch) == "Cc"


def _parse_pdf(data: bytes, max_chars: int | None) -> list[Section]:
    # pypdf raises far more than PdfReadError on damaged files (KeyError, ValueError, ...).
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ParseError("pdf is password protected")
        pages = []
        length = 0
        for page in reader.pages:
            pages.append(page.extract_text() or "")
            length += len(pages[-1])
            if max_chars is not None and length > max_chars:
                raise DocumentTooLargeError(f"pdf has more than {max_chars} characters")
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"cannot read pdf: {exc}") from exc

    total = sum(len(re.sub(r"\s", "", text)) for text in pages)
    if total < MIN_TEXT_CHARS:
        raise EmptyDocumentError("pdf has no text layer", scanned=True)

    return [
        Section(_fix_pdf_text(text), page=number)
        for number, text in enumerate(pages, start=1)
        if text.strip()
    ]


def _fix_pdf_text(text: str) -> str:
    # Join words broken by hyphenation at line ends: "докумен-\nтация" -> "документация".
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _parse_docx(data: bytes, max_chars: int | None) -> list[Section]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            unpacked = sum(i.file_size for i in archive.infolist())
        if unpacked > MAX_DOCX_UNPACKED:
            raise DocumentTooLargeError(f"docx unpacks to {unpacked} bytes", unpacked=True)
        document = docx.Document(io.BytesIO(data))
        styles = {s.style_id: s.name.lower() for s in document.styles if s.name}
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"cannot read docx: {exc}") from exc

    sections: list[Section] = []
    heading: str | None = None
    buffer: list[str] = []
    length = 0

    def flush() -> None:
        if buffer:
            sections.append(Section("\n\n".join(buffer), heading=heading))
            buffer.clear()

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            if _is_heading(styles.get(block._p.style, "")):
                flush()
                heading = text
            else:
                buffer.append(text)
        elif isinstance(block, Table):
            rows = [_table_row(row) for row in block.rows]
            text = "\n".join(row for row in rows if row)
            if text:
                buffer.append(text)
        length += len(text)
        if max_chars is not None and length > max_chars:
            raise DocumentTooLargeError(f"docx has more than {max_chars} characters")
    flush()
    return sections


def _table_row(row) -> str:
    # A horizontally merged cell is returned once per grid column it spans.
    cells = []
    for cell in row.cells:
        if not cells or cell._tc is not cells[-1]._tc:
            cells.append(cell)
    texts = [cell.text.strip() for cell in cells]
    return " | ".join(texts) if any(texts) else ""


def _is_heading(style: str) -> bool:
    return style.startswith(("heading", "заголовок")) or style == "title"


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _parse_markdown(text: str) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    lines: list[str] = []
    in_code = False

    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        match = None if in_code else _MD_HEADING.match(line)
        if match:
            if "".join(lines).strip():
                sections.append(Section("\n".join(lines).strip(), heading=heading))
            heading = match.group(2)
            lines = []
        else:
            lines.append(line)

    if "".join(lines).strip():
        sections.append(Section("\n".join(lines).strip(), heading=heading))
    return sections

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")

MIN_TEXT_CHARS = 20


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
    return data.decode("utf-8-sig", errors="replace")


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
    texts = [cell.text.strip() for cell in row.cells]
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

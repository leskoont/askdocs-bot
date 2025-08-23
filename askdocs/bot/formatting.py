import re
from html import escape

from askdocs.bot import texts
from askdocs.rag.service import Answer, Source

# Telegram limits the text after entity parsing, counted in UTF-16 code units.
MESSAGE_LIMIT = 4096

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_SEPARATORS = ("\n\n", "\n", ". ", " ")


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    parts: list[str] = []
    text = text.strip()
    while text:
        window = _utf16_prefix(text, limit)
        if window == len(text):
            parts.append(text)
            break
        cut = window
        for sep in _SEPARATORS:
            pos = text.rfind(sep, 0, window)
            if pos >= window // 2:
                cut = pos + (1 if sep == ". " else 0)
                break
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return [p for p in parts if p]


def _utf16_prefix(text: str, limit: int) -> int:
    """Length of the longest prefix of text that fits into limit UTF-16 code units."""
    used = 0
    for index, char in enumerate(text):
        used += 2 if ord(char) > 0xFFFF else 1
        if used > limit:
            return index
    return len(text)


def to_html(text: str) -> str:
    # Models sometimes emit **bold** despite the prompt; render it instead of showing asterisks.
    return _BOLD.sub(r"<b>\1</b>", escape(text, quote=False))


def render_sources(sources: list[Source]) -> list[str]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for source in sources:
        grouped.setdefault((source.file_name, source.location), []).append(source.number)
    lines = []
    for (file_name, location), numbers in grouped.items():
        refs = "".join(f"[{n}]" for n in numbers)
        lines.append(f"{refs} {file_name}, {location}" if location else f"{refs} {file_name}")
    return lines


def render_answer(answer: Answer, limit: int = MESSAGE_LIMIT) -> list[str]:
    plain_parts = split_message(answer.text, limit)
    html_parts = [to_html(part) for part in plain_parts]
    if not answer.sources:
        return html_parts

    blocks = _source_blocks(render_sources(answer.sources), limit)
    rendered = [_sources_html(block, header=index == 0) for index, block in enumerate(blocks)]
    first = "\n".join(blocks[0])
    if plain_parts and utf16_len(plain_parts[-1]) + 2 + utf16_len(first) <= limit:
        html_parts[-1] = f"{html_parts[-1]}\n\n{rendered.pop(0)}"
    return html_parts + rendered


def _source_blocks(lines: list[str], limit: int) -> list[list[str]]:
    blocks = [[texts.SOURCES_HEADER]]
    size = utf16_len(texts.SOURCES_HEADER)
    for line in lines:
        line = line[: _utf16_prefix(line, limit)]
        if size + 1 + utf16_len(line) > limit:
            blocks.append([])
            size = -1
        blocks[-1].append(line)
        size += 1 + utf16_len(line)
    return blocks


def _sources_html(lines: list[str], *, header: bool) -> str:
    escaped = [escape(line, quote=False) for line in lines]
    if header:
        escaped[0] = f"<b>{escaped[0]}</b>"
    return "\n".join(escaped)

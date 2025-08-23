from askdocs.bot.formatting import (
    render_answer,
    render_sources,
    split_message,
    to_html,
    utf16_len,
)
from askdocs.rag.service import Answer, Source


def test_short_message_is_untouched():
    assert split_message("Короткий ответ.") == ["Короткий ответ."]


def test_split_respects_limit_and_keeps_text():
    paragraphs = [f"Абзац {i}. " + "слово " * 40 for i in range(40)]
    text = "\n\n".join(p.strip() for p in paragraphs)
    parts = split_message(text, 1000)
    assert len(parts) > 1
    assert all(len(p) <= 1000 for p in parts)
    assert " ".join(" ".join(parts).split()) == " ".join(text.split())


def test_split_prefers_paragraph_boundaries():
    text = "а" * 600 + "\n\n" + "б" * 600
    assert split_message(text, 1000) == ["а" * 600, "б" * 600]


def test_split_without_separators_cuts_hard():
    parts = split_message("я" * 2500, 1000)
    assert [len(p) for p in parts] == [1000, 1000, 500]


def test_split_counts_utf16_units():
    # characters outside the BMP take two UTF-16 code units
    text = "\U0001f600" * 3000
    parts = split_message(text, 4096)
    assert all(utf16_len(p) <= 4096 for p in parts)
    assert "".join(parts) == text


def test_html_is_escaped():
    assert to_html("a < b && c > d") == "a &lt; b &amp;&amp; c &gt; d"
    assert to_html("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_markdown_bold_becomes_html():
    assert to_html("Срок — **30 дней** <важно>") == "Срок — <b>30 дней</b> &lt;важно&gt;"


def test_sources_are_grouped_by_location():
    sources = [
        Source(1, "a.pdf", 3, None),
        Source(2, "b.docx", None, "Оплата"),
        Source(3, "a.pdf", 3, None),
        Source(4, "notes.txt", None, None),
    ]
    assert render_sources(sources) == [
        "[1][3] a.pdf, стр. 3",
        "[2] b.docx, раздел «Оплата»",
        "[4] notes.txt",
    ]


def test_render_answer_appends_escaped_sources():
    answer = Answer("Ответ [1].", [Source(1, "R&D <draft>.pdf", 2, None)], found=True)
    assert render_answer(answer) == [
        "Ответ [1].\n\n<b>Источники:</b>\n[1] R&amp;D &lt;draft&gt;.pdf, стр. 2"
    ]


def test_sources_go_to_separate_message_when_no_room():
    answer = Answer("x" * 990, [Source(1, "a.pdf", 1, None)], found=True)
    parts = render_answer(answer, limit=1000)
    assert parts[0] == "x" * 990
    assert parts[1].startswith("<b>Источники:</b>")


def test_many_sources_are_split_across_messages():
    sources = [Source(i, f"{'документ' * 20}_{i}.pdf", i, None) for i in range(1, 41)]
    parts = render_answer(Answer("Ответ.", sources, found=True), limit=1000)
    assert len(parts) > 2
    assert all(utf16_len(p) <= 1000 for p in parts)
    assert parts[0].startswith("Ответ.\n\n<b>Источники:</b>")
    assert "".join(parts).count(".pdf, стр.") == 40


def test_oversized_source_line_is_trimmed():
    answer = Answer("Ответ.", [Source(1, "я" * 2000 + ".pdf", 1, None)], found=True)
    assert all(utf16_len(p) <= 1000 for p in render_answer(answer, limit=1000))

from itertools import pairwise

import pytest

from askdocs.rag.chunking import chunk_sections
from askdocs.rag.chunking import split_text as _split_text
from askdocs.rag.parsing import Section


def split_text(text: str, size: int, overlap: int) -> list[str]:
    return list(_split_text(text, size, overlap))


def paragraph(n: int, words: int = 20) -> str:
    return " ".join(f"слово{n}_{i}" for i in range(words)) + "."


def test_short_text_is_one_chunk():
    assert split_text("Первый абзац.\n\nВторой абзац.", 500, 50) == [
        "Первый абзац.\n\nВторой абзац."
    ]


def test_chunks_respect_size():
    text = "\n\n".join(paragraph(i) for i in range(30))
    chunks = split_text(text, 400, 80)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_paragraphs_are_not_cut_when_they_fit():
    paragraphs = [paragraph(i, words=12) for i in range(10)]
    chunks = split_text("\n\n".join(paragraphs), 450, 0)
    for chunk in chunks:
        for part in chunk.split("\n\n"):
            assert part in paragraphs


def test_consecutive_chunks_overlap():
    text = " ".join(f"Предложение номер {i} про договор." for i in range(60))
    chunks = split_text(text, 300, 100)
    assert len(chunks) > 2
    for prev, nxt in pairwise(chunks):
        head = nxt.split("\n\n")[0]
        assert head in prev, "next chunk should start with the tail of the previous one"
        assert 0 < len(head) <= 100


def test_overlap_starts_on_sentence_boundary():
    text = " ".join(f"Фраза {i} закончилась." for i in range(80))
    chunks = split_text(text, 250, 60)
    for chunk in chunks[1:]:
        assert chunk.startswith("Фраза")


def test_long_sentence_is_split_by_words():
    sentence = " ".join(["очень"] * 300)
    chunks = split_text(sentence, 200, 0)
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).split() == sentence.split()


def test_huge_token_is_hard_split():
    chunks = split_text("x" * 1000, 300, 0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]


def test_whitespace_is_normalized_and_empty_dropped():
    text = "  первая   строка \t с пробелами \n\n\n\n   \n\nвторая строка  "
    assert split_text(text, 500, 0) == ["первая строка с пробелами\n\nвторая строка"]
    assert split_text("   \n\n  ", 500, 0) == []


def test_sections_keep_page_and_heading():
    sections = [
        Section("Текст первой страницы. " * 30, page=1),
        Section("Короткий раздел.", heading="Оплата"),
    ]
    chunks = chunk_sections(sections, 300, 50)
    assert {c.page for c in chunks[:-1]} == {1}
    assert chunks[-1].heading == "Оплата"
    assert chunks[-1].page is None
    assert chunks[-1].text == "Короткий раздел."


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_sections([Section("текст")], 100, 100)


def test_huge_word_with_overlap_stays_within_size():
    chunks = split_text("начало " + "y" * 5000 + " конец", 500, 100)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks).count("y") >= 5000


def test_limit_stops_early():
    sections = [Section(paragraph(i, words=40)) for i in range(100)]
    assert len(chunk_sections(sections, 300, 0, limit=5)) == 6

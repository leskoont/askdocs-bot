import io

import pytest
from helpers import make_docx, make_pdf
from pypdf import PdfReader, PdfWriter

from askdocs.rag.parsing import (
    DocumentTooLargeError,
    EmptyDocumentError,
    ParseError,
    is_supported,
    parse_document,
)


def test_supported_extensions():
    assert is_supported("Договор.PDF")
    assert is_supported("notes.md")
    assert not is_supported("table.xlsx")
    assert not is_supported("README")


def test_txt_utf8():
    sections = parse_document("a.txt", "Привет, мир".encode())
    assert sections[0].text == "Привет, мир"
    assert sections[0].page is None


def test_markdown_sections_by_heading():
    md = (
        "Вступление без заголовка.\n\n"
        "# Оплата\n\nОплата в течение 10 дней.\n\n"
        "## Штрафы ##\n\nПеня 0,1% в день.\n\n"
        "```bash\n# это комментарий, не заголовок\n```\n"
    )
    sections = parse_document("doc.md", md.encode())
    assert [s.heading for s in sections] == [None, "Оплата", "Штрафы"]
    assert "Оплата в течение 10 дней." in sections[1].text
    assert "# это комментарий" in sections[2].text


def test_docx_headings_and_tables():
    data = make_docx(
        [
            ("p", "Общие положения договора."),
            ("h1", "Сроки"),
            ("p", "Работы выполняются за 30 дней."),
            ("table", "Этап|Срок;Дизайн|5 дней"),
        ]
    )
    sections = parse_document("contract.docx", data)
    assert sections[0].heading is None
    assert sections[1].heading == "Сроки"
    assert "Работы выполняются за 30 дней." in sections[1].text
    assert "Дизайн | 5 дней" in sections[1].text


def test_pdf_pages_are_numbered():
    data = make_pdf(
        [
            ["Payment terms", "The invoice is paid within ten days."],
            [],
            ["Liability", "Penalty is one percent per day."],
        ]
    )
    sections = parse_document("contract.pdf", data)
    assert [s.page for s in sections] == [1, 3]
    assert "invoice is paid within ten days" in sections[0].text
    assert "Penalty" in sections[1].text


def test_pdf_hyphenation_is_joined():
    data = make_pdf([["The docu-", "mentation is complete and long enough."]])
    assert "documentation" in parse_document("a.pdf", data)[0].text


def test_pdf_without_text_is_reported_as_scanned():
    with pytest.raises(EmptyDocumentError) as info:
        parse_document("scan.pdf", make_pdf([[], []]))
    assert info.value.scanned


def test_empty_txt_is_not_scanned():
    with pytest.raises(EmptyDocumentError) as info:
        parse_document("empty.txt", b"  \n\n ")
    assert not info.value.scanned


@pytest.mark.parametrize("name", ["broken.pdf", "broken.docx"])
def test_corrupted_files(name):
    with pytest.raises(ParseError):
        parse_document(name, b"definitely not a real file")


def test_unsupported_format():
    with pytest.raises(ParseError):
        parse_document("photo.jpg", b"\xff\xd8\xff")


TEXT = "Съешь же ещё этих мягких французских булок, да выпей чаю."


@pytest.mark.parametrize("text", [TEXT, "The “quick” brown fox — jumps over the dog."])
def test_utf8_with_one_broken_byte(text):
    data = (text * 5).encode()
    data = data[:100] + b"\xff" + data[100:]
    decoded = parse_document("a.txt", data)[0].text
    assert decoded.count("\ufffd") == 1
    assert text in decoded


def test_utf8_glued_to_cp1251_is_not_read_as_koi8():
    data = (TEXT * 6).encode() + (TEXT * 2).encode("cp1251")
    decoded = parse_document("a.txt", data)[0].text
    assert TEXT in decoded
    assert not any("\u2500" <= ch <= "\u259f" for ch in decoded)


def encrypt(pdf: bytes, algorithm: str, user_password: str = "") -> bytes:
    reader = PdfReader(io.BytesIO(pdf))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm=algorithm)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-128", "AES-256"])
def test_pdf_with_owner_password_only(algorithm):
    # Such PDFs open without a password in any viewer; only editing is restricted.
    data = encrypt(make_pdf([["Payment is due within ten days."]]), algorithm)
    assert "within ten days" in parse_document("a.pdf", data)[0].text


def test_pdf_with_user_password():
    data = encrypt(make_pdf([["Payment is due within ten days."]]), "AES-256", "secret")
    with pytest.raises(ParseError):
        parse_document("a.pdf", data)


def test_long_documents_stop_early():
    paragraphs = [("p", f"Пункт {i}. " + "текст " * 20) for i in range(50)]
    with pytest.raises(DocumentTooLargeError):
        parse_document("big.docx", make_docx(paragraphs), max_chars=1000)
    pages = [[f"Page {i} " + "text " * 20] for i in range(20)]
    with pytest.raises(DocumentTooLargeError):
        parse_document("big.pdf", make_pdf(pages), max_chars=1000)
    assert len(parse_document("big.pdf", make_pdf(pages))) == 20

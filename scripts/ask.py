"""Run the RAG pipeline on local files without Telegram.

    python scripts/ask.py examples/dogovor-uslug.md "Какой размер предоплаты?"

Uses the same settings (.env) as the bot, but an in-memory Qdrant, so the bot's data is untouched.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from askdocs.bot.app import create_rag, load_embedder
from askdocs.bot.formatting import render_sources
from askdocs.config import Settings
from askdocs.log import setup_logging
from askdocs.rag.llm import LLMError


async def main(files: list[Path], questions: list[str], show_context: bool) -> int:
    settings = Settings(bot_token="0:unused")
    setup_logging("WARNING", secrets=[settings.llm_api_key.get_secret_value()])

    embedder = await load_embedder(settings)
    qdrant = QdrantClient(location=":memory:")
    rag = create_rag(settings, qdrant, embedder)
    await rag.store.setup(create_indexes=False)

    chat_id = 1
    for path in files:
        doc = await rag.add_document(chat_id, path.name, path.read_bytes())
        print(f"indexed {doc.file_name}: {doc.chunk_count} chunks")

    print(f"llm: {settings.llm_model} at {settings.llm_base_url}\n")
    for question in questions:
        print(f"Q: {question}")
        if show_context:
            vector = await embedder.embed_query(question)
            hits = await rag.store.search(chat_id, vector, limit=settings.fetch_k)
            for hit in hits[:8]:
                print(f"   {hit.score:.3f}  {hit.text[:70].replace(chr(10), ' ')}")
        try:
            answer = await rag.answer(chat_id, question)
        except LLMError as exc:
            print(f"   LLM error: {exc}\n")
            continue
        print(f"A: {answer.text}")
        for line in render_sources(answer.sources):
            print(f"   {line}")
        print()

    await rag.llm.close()
    rag.store.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, nargs="+", help="documents and then questions")
    parser.add_argument("-q", "--question", action="append", default=[])
    parser.add_argument("--show-context", action="store_true", help="print retrieval scores")
    args = parser.parse_args()

    files = [p for p in args.file if p.is_file()]
    questions = [str(p) for p in args.file if not p.is_file()] + args.question
    if not files or not questions:
        parser.error("need at least one existing file and one question")
    sys.exit(asyncio.run(main(files, questions, args.show_context)))

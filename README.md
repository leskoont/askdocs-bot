# askdocs-bot

A Telegram bot that answers questions about the documents you send it. Send a PDF, DOCX, TXT or MD file to the chat, then ask questions as plain messages. The bot finds the relevant fragments and passes them to a language model, and lists the sources under the answer: the file and the page or section. If the documents don't contain the answer, the bot says so.

Embeddings are computed locally with `intfloat/multilingual-e5-small`, fragments are stored in Qdrant, and the answer is written by any model with an OpenAI-compatible API: Ollama, vLLM, LM Studio, OpenRouter, OpenAI. The bot is built on aiogram 3 and runs with polling or a webhook. Each chat's documents are visible only in that chat. The bot's replies and source labels are in Russian.

## Example

A run of `scripts/ask.py` on the service contract in [`examples/dogovor-uslug.md`](examples/dogovor-uslug.md), model `minimax/minimax-m3.1-flash:free` through OpenRouter. The contract and the questions are in Russian. The questions ask for the prepayment amount and deadline, the contractor's late-work penalty, and the customer's CEO. The last answer means "The uploaded documents contain no answer to this question."

```
$ python scripts/ask.py examples/dogovor-uslug.md \
    "Какой размер предоплаты и в какой срок её нужно внести?" \
    "Какая пеня за просрочку работ у исполнителя?" \
    "Кто генеральный директор Заказчика?"
indexed dogovor-uslug.md: 10 chunks
llm: minimax/minimax-m3.1-flash:free at https://openrouter.ai/api/v1

Q: Какой размер предоплаты и в какой срок её нужно внести?
A: Предоплата составляет 90 000 рублей (50% стоимости) и вносится в течение 5 банковских дней после подписания договора. [1]
   [1] dogovor-uslug.md, раздел «3. Стоимость и порядок оплаты»

Q: Какая пеня за просрочку работ у исполнителя?
A: За просрочку выполнения работ Исполнитель уплачивает пеню в размере 0,1% от стоимости работ за каждый день просрочки, но не более 10% от стоимости работ [1].
   [1] dogovor-uslug.md, раздел «6. Ответственность сторон»

Q: Кто генеральный директор Заказчика?
A: В загруженных документах нет ответа на этот вопрос.
```

In Telegram the answer arrives as a reply to the question.

Commands: `/docs` lists the chat's documents with delete buttons, `/clear` deletes all documents after confirmation, `/ask <question>` is used in groups, `/help` prints the help text.

## How it works

On upload, the file is split into sections: PDF pages, and sections by heading in DOCX and Markdown. Sections are cut into fragments of about 1000 characters with a 150-character overlap, at paragraph and sentence boundaries, so a fragment never spans two pages or two sections. The fragments are embedded and stored in Qdrant together with `chat_id`. A document is not visible to search until all its fragments are written; leftovers from an interrupted upload or deletion are cleaned up at startup. Duplicates are detected by sha256, and a file with the same name replaces the previous version.

For a question, the bot takes the 20 nearest fragments of that chat, drops the weak ones, picks up to 5 mutually dissimilar ones from the rest (MMR) and fits them into 6000 characters. The model receives the numbered fragments and cites them as `[1]`; only the fragments it cited end up in the sources, and if it cited none, the three closest to the question are shown. If no suitable fragments are found, the model is not called at all.

A fragment is weak if its cosine similarity to the question is below `SCORE_THRESHOLD` (0.78) or it trails the best one by more than `SCORE_MARGIN` (0.06). With e5-small the similarities are squeezed into a narrow band, so a single threshold is not enough. The best similarity for the example above:

| Question | Best fragment |
|---|---|
| Какая пеня за просрочку работ у исполнителя? | 0.894 |
| Сколько длится гарантия? | 0.874 |
| Какой размер предоплаты и в какой срок её нужно внести? | 0.855 |
| Кто генеральный директор Заказчика? | 0.781 |
| Как сварить борщ? | 0.772 |
| Какая столица Австралии? | 0.716 |
| Сколько стоит доставка пиццы? | 0.811 |
| What is the penalty for late delivery? | 0.773 |

The threshold filters out clearly unrelated questions without calling the model, but not all of them: the pizza question passes because of the word "стоимость" (cost), and then it is the model that says there is no answer. The 0.78 threshold was tuned on this contract; the CEO question clears it by a thousandth, so on other documents and with a different embedding model the threshold should be rechecked. `scripts/ask.py --show-context` prints the similarities.

## Running

You need a bot token from [@BotFather](https://t.me/BotFather) and access to an LLM: a local Ollama or a key for an OpenAI-compatible API. All parameters are listed with comments in [`.env.example`](.env.example).

### Docker

```bash
cp .env.example .env      # fill in BOT_TOKEN and the LLM settings
docker compose up -d --build
docker compose logs -f bot
```

This starts the bot and Qdrant; Qdrant data and the embedding model cache live in volumes. The model is downloaded on first start. By default the bot looks for Ollama on the host machine at `http://host.docker.internal:11434/v1`. On Linux, Ollama listens only on `127.0.0.1` by default and is not reachable from the container: start it with `OLLAMA_HOST=0.0.0.0`.

### Without Docker

Python 3.11–3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
cp .env.example .env
python -m askdocs
```

Without `QDRANT_URL`, the embedded Qdrant in `data/qdrant` is used. It is meant for a single process: a second bot using the same folder exits with an error saying the folder is locked. Embedded mode is fine for small volumes; for larger ones, run a separate Qdrant and set `QDRANT_URL`.

You can check the pipeline without Telegram: `scripts/ask.py` takes its settings from `.env` and keeps the data in memory.

```bash
python scripts/ask.py examples/dogovor-uslug.md "Сколько длится гарантия?" --show-context
pytest
ruff check .
```

On Windows with Python from Anaconda, torch may fail to import (`WinError 1114` while loading `c10.dll`) because of an old `vcruntime140.dll` in the distribution. A venv on regular CPython helps, for example `uv venv --python 3.12`.

### Groups

In a group the bot answers only to `/ask`, so it doesn't react to every message. It sees files only if privacy mode is disabled in [@BotFather](https://t.me/BotFather) (`/setprivacy` → Disable) or the bot is made a group administrator; after changing the mode, remove the bot from the group and add it again. Only the group's administrators can delete documents in a group.

## Ollama on a 6 GB GPU

`qwen2.5:7b-instruct` quantized as q4_K_M (the default tag in Ollama) takes about 4.7 GB and fits in 6 GB together with an 8k-token context.

```bash
ollama pull qwen2.5:7b-instruct
```

In `.env`:

```
LLM_BASE_URL=http://localhost:11434/v1        # from docker: http://host.docker.internal:11434/v1
LLM_MODEL=qwen2.5:7b-instruct
EMBEDDING_DEVICE=cpu
```

Ollama's default context may be smaller than needed for 6000 characters of fragments plus the answer, in which case the start of the prompt is silently truncated. Set `OLLAMA_CONTEXT_LENGTH=8192` in the Ollama server's environment, and if VRAM is short, lower it to 4096 and reduce `MAX_CONTEXT_CHARS` to 3500. It is better to compute embeddings on the CPU so they don't take VRAM from the LLM.

Ollama can also be run in docker-compose: the `ollama` service in `docker-compose.yml` is commented out. It needs the NVIDIA Container Toolkit, and `LLM_BASE_URL=http://ollama:11434/v1` in `.env`.

## Limitations

- Scans without a text layer are not read; there is no OCR. The bot says so on upload.
- Each question is handled on its own, so follow-ups like "and how much is that in rubles?" won't work.
- A question in one language about a document in another gets noticeably lower similarity and may not pass the threshold, like the English question in the table above. A lower `SCORE_THRESHOLD` or `multilingual-e5-base` helps.
- The rate limit is kept in process memory, so the bot is meant to run as a single replica.

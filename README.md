# askdocs-bot

Telegram-бот, который отвечает на вопросы по присланным документам. В чат отправляется PDF, DOCX, TXT или MD, после этого можно спрашивать обычными сообщениями. Бот ищет подходящие фрагменты и передаёт их языковой модле, а под ответом перечисляет источники: файл и страницу или раздел. Если в документах ответа нет, бот так и пишет.

Эмбеддинги считаются локально моделью `intfloat/multilingual-e5-small`, фрагменты хранятся в Qdrant, ответ пишет любая модель с OpenAI-совместимым API: Ollama, vLLM, LM Studio, OpenRouter, OpenAI. Бот написан на aiogram 3 и работает через polling. Документы каждого чата видны только в этом чате.

## Запуск

Нужен токен бота от [@BotFather](https://t.me/BotFather) и доступ к LLM: локальный Ollama или ключ OpenAI-совместимого API. Все параметры с пояснениями перечислены в [`.env.example`](.env.example).

### Без Docker

Python 3.11–3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # или CUDA-сборку
pip install -e ".[dev]"
cp .env.example .env
python -m askdocs
```

Без `QDRANT_URL` используется встроенный Qdrant в папке `data/qdrant`. Он рассчитан на один процесс: второй бот с той же папкой завершится с ошибкой, что папка занята. Встроенный режим подходит для небольших объёмов, для больших нужен отдельный Qdrant через `QDRANT_URL`.

Пайплайн можно проверить без Telegram: `scripts/ask.py` берёт настройки из `.env` и держит данные в памяти.

```bash
python scripts/ask.py examples/dogovor-uslug.md "Сколько длится гарантия?" --show-context
pytest
ruff check .
```

На Windows с Python из Anaconda torch может не импортироваться (`WinError 1114` при загрузке `c10.dll`) из-за старого `vcruntime140.dll` в дистрибутиве. Помогает venv на обычном CPython, например `uv venv --python 3.12`.

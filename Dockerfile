FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app

# CPU build of torch: the default wheel pulls ~2 GB of CUDA libraries,
# and a small embedding model runs fine on CPU.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# Dependencies first, so that a code change does not reinstall them.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY README.md LICENSE ./
COPY askdocs ./askdocs
RUN pip install --no-deps .

RUN useradd --create-home --uid 1000 bot \
    && mkdir -p /models /app/data \
    && chown bot:bot /models /app/data
USER bot

CMD ["python", "-m", "askdocs"]

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
COPY ui/ ui/
COPY data/ data/

RUN pip install --no-cache-dir -e . \
    # bake the tokenizer into the image so the first request doesn't need the network
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && useradd --create-home app && mkdir logs && chown -R app /app

USER app
EXPOSE 8000

# Render injects $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn arbiter.api:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]

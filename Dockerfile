FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV HOME=/home/adept
WORKDIR /app

RUN addgroup --system adept \
    && adduser --system --ingroup adept --home /home/adept adept \
    && mkdir -p /home/adept/.cache/uv \
    && chown -R adept:adept /home/adept

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --no-install-project \
    && chown -R adept:adept /home/adept \
    && chown -R adept:adept /app

COPY app app

RUN chown -R adept:adept /app

USER adept

COPY app app
USER adept
EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
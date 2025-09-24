# Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (opzionale) se servono wheels per sentence-transformers / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps prima -> cache build
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt


# codice
COPY . /app

# porte comuni (FastAPI e Streamlit)
EXPOSE 8000 8501

# di default non avviamo nulla: ogni servizio imposta il proprio "command" su compose
CMD ["bash", "-lc", "echo 'Set command in docker-compose.yml (uvicorn/streamlit)' && sleep infinity"]

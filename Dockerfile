FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-dev.txt ./

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-dev.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgles2 \
        libegl1 && \
    rm -rf /var/lib/apt/lists/*

COPY src/ ./src/
COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "tests/", "-q"]
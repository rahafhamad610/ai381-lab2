FROM python:3.14-slim@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade apt libapt-pkg7.0 zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS builder

RUN python -m venv /opt/venv

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY tests/ tests/
COPY artifacts/ artifacts/

RUN python src/train/train.py
RUN python -m pytest -v

FROM base AS runtime

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --chown=appuser:appuser artifacts/ /app/artifacts/
ARG GIT_SHA=local
ENV GIT_SHA=${GIT_SHA}

# Package installation is complete; omit installers from runtime.
RUN /opt/venv/bin/python -m pip uninstall -y pip \
    && /usr/local/bin/python -m pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.14/ensurepip

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# AMR Resistance Prediction — Dockerfile
# Base: python:3.11-slim-bookworm (Debian 12)
#
# Build args:
#   DEVICE=cpu  →  PyTorch CPU   (por defecto)
#   DEVICE=gpu  →  PyTorch CUDA 12.1
#
# Uso directo:
#   docker build --build-arg DEVICE=cpu -t amr-pred:cpu .
#   docker build --build-arg DEVICE=gpu -t amr-pred:gpu .
#
# Uso con docker compose (recomendado):
#   docker compose --profile cpu up -d
#   docker compose --profile gpu up -d
# =============================================================================

ARG DEVICE=cpu

# ---------- Stage 1: builder -------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ARG DEVICE

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 1. Dependencias base (sin torch) — capa cacheada independientemente de DEVICE
COPY requirements/base.txt requirements/base.txt
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements/base.txt

# 2. Dependencias específicas del dispositivo (cpu o gpu)
#    El condicional se resuelve en build-time con el ARG.
COPY requirements/cpu.txt requirements/cpu.txt
COPY requirements/gpu.txt requirements/gpu.txt
RUN pip install --prefix=/install --no-cache-dir \
        -r requirements/${DEVICE}.txt


# ---------- Stage 2: runtime -------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ARG DEVICE
# Guardamos DEVICE como variable de entorno para que los scripts puedan leerla
ENV DEVICE=${DEVICE}
ENV PYTHONPATH=/workspace/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

LABEL maintainer="amr-project"
LABEL description="AMR resistance prediction — device=${DEVICE}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    jellyfish \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd --create-home --shell /bin/bash amr
USER amr
WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]

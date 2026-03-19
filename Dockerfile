# ============================================================================
# AMR Resistance Prediction — Dockerfile
# Multi-stage build con pip. Soporta CPU y GPU via build arg.
#
# Build args:
#   DEVICE=cpu (default) | gpu
#
# Uso:
#   docker compose --profile cpu up -d --build
#   docker compose --profile gpu up -d --build
# ============================================================================

# ── Stage 1: base con dependencias del sistema ────────────────────────────
FROM python:3.11-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        jellyfish \
        procps \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: instalar dependencias Python ──────────────────────────────────
FROM base AS deps

WORKDIR /tmp/reqs

# Copiar requirements (cache layer — solo se reconstruye si cambian)
COPY requirements/ ./

# Argumento para seleccionar CPU o GPU
ARG DEVICE=cpu

# Instalar PyTorch primero (es el paquete más pesado)
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip install -r torch-gpu.txt ; \
    else \
        pip install -r torch-cpu.txt ; \
    fi

# Instalar el resto de dependencias
RUN pip install -r base.txt
RUN pip install -r dev.txt

# ── Stage 3: imagen de desarrollo ─────────────────────────────────────────
FROM deps AS dev

WORKDIR /workspace

# Crear usuario no-root
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && mkdir -p /workspace/datos/brutos/genomas_fasta /workspace/datos/procesados \
                /workspace/informes/modelos /workspace/informes/figuras \
    && chown -R ${USERNAME}:${USERNAME} /workspace

# Jupyter sin token ni password
ENV JUPYTER_TOKEN="" \
    JUPYTER_CONFIG_DIR="/home/${USERNAME}/.jupyter"

RUN mkdir -p ${JUPYTER_CONFIG_DIR} \
    && echo "c.ServerApp.token = ''" > ${JUPYTER_CONFIG_DIR}/jupyter_server_config.py \
    && echo "c.ServerApp.password = ''" >> ${JUPYTER_CONFIG_DIR}/jupyter_server_config.py \
    && echo "c.ServerApp.disable_check_xsrf = True" >> ${JUPYTER_CONFIG_DIR}/jupyter_server_config.py \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8888/api/status || exit 1

EXPOSE 8888

USER ${USERNAME}

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

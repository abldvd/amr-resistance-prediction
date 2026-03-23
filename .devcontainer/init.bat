@echo off
REM ── .devcontainer/init.bat ──────────────────────────────────────────────────
REM Se ejecuta en el host (Windows CMD) antes de arrancar el contenedor.
REM 1. Crea .env si no existe.
REM 2. Detecta GPU NVIDIA y escribe docker-compose.devcontainer.yml en consecuencia.

REM ── .env ────────────────────────────────────────────────────────────────────
if not exist .env (
    echo [devcontainer] Creando .env desde .env.example...
    copy .env.example .env >nul
)

REM ── Deteccion GPU ────────────────────────────────────────────────────────────
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo [devcontainer] GPU NVIDIA detectada - configurando entorno GPU...
    copy /y .devcontainer\docker-compose.gpu.yml docker-compose.devcontainer.yml >nul
) else (
    echo [devcontainer] Sin GPU NVIDIA - configurando entorno CPU...
    copy /y .devcontainer\docker-compose.cpu.yml docker-compose.devcontainer.yml >nul
)

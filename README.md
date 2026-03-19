# Predicción de Resistencia Antibiótica (AMR)

### Aprendizaje automático sobre datos de secuenciación genómica completa

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-required-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Licencia](https://img.shields.io/badge/Licencia-MIT-green?style=flat-square)](LICENSE)
[![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-orange?style=flat-square)]()

Predicción de fenotipos de resistencia antibiótica (Susceptible / Intermedio / Resistente) en *Klebsiella pneumoniae* a partir de datos de secuenciación genómica completa (WGS) procedentes de [BV-BRC](https://www.bv-brc.org/). El pipeline combina vectores de frecuencia de k-mers, matrices de presencia/ausencia de genes AMR y tres modelos de ML supervisado — Random Forest, XGBoost y **TabNet** — con interpretabilidad basada en valores SHAP y atención aprendida.

> **Entorno de desarrollo:** Docker + VSCode Dev Containers · Python 3.11 · JupyterLab

---

## Índice

- [Contexto científico](#contexto-científico)
- [Modelos](#modelos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Entorno de desarrollo](#entorno-de-desarrollo)
  - [Requisitos previos](#1-requisitos-previos)
  - [Clonar el repositorio](#2-clonar-el-repositorio)
  - [Configurar variables de entorno](#3-configurar-variables-de-entorno)
  - [Crear directorios de datos](#4-crear-directorios-de-datos)
  - [Construir la imagen Docker](#5-construir-la-imagen-docker)
  - [Opción A — VSCode Dev Containers](#opción-a--vscode-dev-containers-recomendado)
  - [Opción B — Terminal sin VSCode](#opción-b--terminal-sin-vscode)
  - [Verificar la instalación](#7-verificar-la-instalación)
  - [Setup GPU (opcional)](#setup-gpu-opcional)
- [Descarga de datos](#descarga-de-datos)
- [Visión general del pipeline](#visión-general-del-pipeline)
- [Resultados](#resultados)
- [Solución de problemas](#solución-de-problemas)
- [Referencias](#referencias)

---

## Contexto científico

La resistencia antimicrobiana (AMR) es una de las amenazas sanitarias globales más urgentes, con una proyección de 10 millones de muertes anuales para 2050 (OMS, 2019). Las pruebas fenotípicas tradicionales (microdilución en caldo) tardan entre 24 y 72 horas y requieren cultivo vivo. La secuenciación genómica completa (WGS) permite predecir la resistencia en menos de 2 horas a partir de un aislado clínico, posibilitando una terapia dirigida más rápida.

Este proyecto replica y amplía el enfoque de predicción de resistencia basado en k-mers descrito en:

> Moradigaravand et al. (2018). *Prediction of antibiotic resistance in Escherichia coli from large-scale pan-genome data*. PLOS Computational Biology.

**Organismo objetivo:** *Klebsiella pneumoniae* (patógeno de prioridad crítica según la OMS)  
**Antibiótico objetivo:** Ciprofloxacino (fluoroquinolona — terapia de primera línea)  
**Tarea:** Clasificación binaria — Susceptible (S) vs Resistente (R)

---

## Modelos

Se comparan tres enfoques de complejidad creciente sobre la misma matriz de features.

### Random Forest — baseline sólido

Ensemble de árboles de decisión con bootstrap. Modelo de referencia estándar en la literatura AMR-ML por su robustez ante features irrelevantes y su facilidad de interpretación vía importancia de Gini. Se usa con `class_weight='balanced'` para compensar el desbalance S:R.

### XGBoost — gradient boosting optimizado

Gradient boosting con regularización L1/L2, manejo nativo de sparsidad y paralelización eficiente. En datos genómicos de alta dimensionalidad supera consistentemente a Random Forest en AUC-ROC gracias a la construcción secuencial de árboles que corrige los errores del modelo anterior. Se usa `scale_pos_weight` para el desbalance de clases.

### TabNet — red neuronal con atención secuencial *(SOTA)*

> Arik, S. & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI. Google Research.

TabNet es una arquitectura de deep learning diseñada específicamente para datos tabulares de alta dimensionalidad. Utiliza un **mecanismo de atención secuencial** que selecciona en cada paso de decisión qué subconjunto de features activar — especialmente idóneo para matrices k-mer dispersas donde la señal útil está distribuida en miles de columnas.

```
Matriz k-mer sparse (N × 100k)
        │
        ▼
  BatchNorm inicial
        │
        ▼
  ┌─────────────────────┐
  │  Paso de atención 1 │ ← máscara aprendida M₁ (qué k-mers mirar)
  └─────────────────────┘
        │  ...  (N_steps = 5 por defecto)
        ▼
  Agregación de pasos
        │
        ▼
  Clasificador final → P(Resistente)
```

---

## Estructura del proyecto

```
amr-resistance-prediction/
│
├── .devcontainer/
│   ├── devcontainer.json        # configuración VSCode Dev Containers
│   └── post-create.sh           # setup automático al crear el contenedor
│
├── requirements/
│   ├── base.txt                 # dependencias comunes (scikit-learn, pandas…)
│   ├── cpu.txt                  # PyTorch CPU-only
│   └── gpu.txt                  # PyTorch CUDA 12.1
│
├── datos/
│   ├── brutos/                  # datos BV-BRC — no versionados (.gitignore)
│   │   ├── genomas_fasta/       # secuencias WGS por lotes (.fasta)
│   │   ├── amr_fenotipos.csv    # etiquetas S/I/R + valores MIC
│   │   └── amr_matriz_genes.csv # presencia/ausencia de genes AMR
│   └── procesados/              # outputs del preprocesado — no versionados
│       ├── informe_qc_genomas.csv
│       ├── matriz_kmers.npz
│       ├── X_final.npz
│       └── y_final.npy
│
├── notebooks/
│   ├── 01_descarga_datos.ipynb
│   ├── 02_control_calidad_genomica.ipynb
│   ├── 03_variable_target.ipynb
│   ├── 04_extraccion_kmers.ipynb
│   ├── 05_matriz_features.ipynb
│   ├── 06_preprocesado.ipynb
│   ├── 07_entrenamiento_rf_xgb.ipynb
│   ├── 08_entrenamiento_tabnet.ipynb
│   └── 09_interpretabilidad.ipynb
│
├── src/
│   └── amr/                     # código reutilizable (importado desde notebooks)
│
├── outputs/
│   ├── models/                  # modelos serializados — no versionados
│   └── figures/                 # gráficos exportados — no versionados
│
├── informes/
│   └── figuras/
│
├── Dockerfile                   # multi-stage, soporta DEVICE=cpu|gpu
├── docker-compose.yml           # perfiles: cpu · gpu
├── .env.example                 # plantilla de variables de entorno
├── .gitignore
└── README.md
```

---

## Entorno de desarrollo

El entorno completo corre dentro de un contenedor Docker. No se instala nada en el sistema host salvo Docker y VSCode. Esto garantiza reproducibilidad total independientemente del sistema operativo del host.

**Tiempo estimado de setup inicial:** 15–25 minutos (dependiendo de la conexión; PyTorch pesa ~2 GB).

---

### 1. Requisitos previos

Instala las siguientes herramientas en tu sistema **antes** de continuar.

#### Docker Desktop (o Docker Engine en Linux)

| Sistema | Descarga | Versión mínima |
|---|---|---|
| Windows (WSL2) | [docs.docker.com/desktop/windows](https://docs.docker.com/desktop/windows/install/) | 4.x |
| macOS | [docs.docker.com/desktop/mac](https://docs.docker.com/desktop/mac/install/) | 4.x |
| Linux (Ubuntu) | [docs.docker.com/engine/install/ubuntu](https://docs.docker.com/engine/install/ubuntu/) | 24.x |

Tras instalar, verifica que Docker funciona correctamente:

```bash
docker --version
# Docker version 26.x.x, build ...

docker compose version
# Docker Compose version v2.x.x

docker run --rm hello-world
# Hello from Docker!
```

> **WSL2 (Windows):** Asegúrate de que en Docker Desktop → Settings → Resources → WSL Integration esté habilitada la distribución que uses (Ubuntu 22.04 recomendado). Clona el repositorio **dentro** del filesystem de WSL (`~/`), no en `/mnt/c/`. El acceso cruzado entre Windows y WSL es muy lento para operaciones I/O intensivas como el procesado de FASTA.

#### VSCode + extensión Dev Containers

Descarga VSCode desde [code.visualstudio.com](https://code.visualstudio.com/).

Instala la extensión **Dev Containers** desde la terminal:

```bash
code --install-extension ms-vscode-remote.remote-containers
```

O búscala en el panel de extensiones de VSCode (`Ctrl+Shift+X`) con el ID: `ms-vscode-remote.remote-containers`.

---

### 2. Clonar el repositorio

```bash
# WSL2 / Linux / macOS — clona dentro del filesystem nativo
cd ~
git clone https://github.com/<tu-usuario>/amr-resistance-prediction.git
cd amr-resistance-prediction
```

---

### 3. Configurar variables de entorno

El proyecto usa un archivo `.env` para gestionar rutas y parámetros. Nunca se versiona porque puede contener tokens privados.

```bash
cp .env.example .env
```

Abre `.env` con cualquier editor y revisa los valores. Los únicos que probablemente tengas que cambiar:

```bash
# Ruta en el HOST donde están (o estarán) los datos.
# Por defecto apunta a ~/amr-data — cámbiala si tus FASTAs están en otro disco.
AMR_DATA_DIR=~/amr-data

# Puerto local para JupyterLab. Cámbialo si el 8888 ya está ocupado.
JUPYTER_PORT=8888

# Token BV-BRC — solo necesario si usas endpoints autenticados de la API.
# Déjalo comentado si no lo tienes aún.
# BVBRC_TOKEN=tu_token_aqui
```

---

### 4. Crear directorios de datos

Los datos se montan en el contenedor desde fuera del repositorio para no versionar gigabytes de FASTA. Crea la estructura en el host:

```bash
mkdir -p ~/amr-data/brutos/genomas_fasta
mkdir -p ~/amr-data/procesados
```

Si configuraste una ruta distinta en `AMR_DATA_DIR`, crea la misma estructura allí.

---

### 5. Construir la imagen Docker

La imagen se construye en dos variantes: **CPU** (por defecto, cualquier máquina) y **GPU** (requiere NVIDIA, ver [Setup GPU](#setup-gpu-opcional)).

#### Construir imagen CPU

```bash
docker compose --profile cpu build
```

La primera build descarga la imagen base de Python y todos los paquetes (~2.5 GB de descarga, imagen final ~3.5 GB). Las builds posteriores son rápidas gracias al sistema de caché por capas de Docker.

Para forzar una rebuild completa sin caché (por ejemplo, tras actualizar `requirements/base.txt`):

```bash
docker compose --profile cpu build --no-cache
```

#### Construir imagen GPU

```bash
docker compose --profile gpu build
```

> Ver la sección [Setup GPU](#setup-gpu-opcional) antes de intentar esto.

---

### Opción A — VSCode Dev Containers (recomendado)

Esta es la forma preferida de trabajar. VSCode se conecta al contenedor y todas las extensiones, el linter, el formateador y el kernel de Jupyter corren **dentro** del contenedor — no en tu máquina local.

**Paso 1.** Abre la carpeta del proyecto en VSCode:

```bash
code .
```

**Paso 2.** VSCode detectará automáticamente el archivo `.devcontainer/devcontainer.json` y mostrará una notificación en la esquina inferior derecha:

```
Folder contains a Dev Container configuration file.
Reopen folder to develop in a container  [Reopen in Container]
```

Haz clic en **Reopen in Container**.

Si la notificación no aparece, ábrela manualmente con el Command Palette:

```
F1  →  Dev Containers: Reopen in Container
```

**Paso 3.** VSCode construirá el contenedor (si no está construido), lo arrancará y abrirá una nueva ventana conectada a él. En la esquina inferior izquierda verás:

```
>< Dev Container: AMR — CPU
```

**Paso 4.** El script `post-create.sh` se ejecuta automáticamente la primera vez. Verás en el terminal integrado de VSCode que:

- Se crean los directorios del proyecto dentro del contenedor
- Se registra el kernel de Jupyter (`AMR Prediction (py3.11)`)
- Se verifican las dependencias críticas (scikit-learn, xgboost, tabnet, shap, biopython)
- Se verifica que `jellyfish` está disponible en el PATH

**Paso 5.** Para abrir JupyterLab, usa el terminal integrado de VSCode (`` Ctrl+` ``) y ejecuta:

```bash
jupyter lab
```

VSCode redirigirá el puerto automáticamente y ofrecerá abrir el navegador. También puedes acceder directamente a [http://localhost:8888](http://localhost:8888).

#### Seleccionar CPU o GPU en el Dev Container

El archivo `.devcontainer/devcontainer.json` apunta a CPU por defecto. Para cambiar a GPU, edita las dos líneas indicadas:

```jsonc
// Antes (CPU):
"service": "dev-cpu",
"runServices": ["dev-cpu"],

// Después (GPU):
"service": "dev-gpu",
"runServices": ["dev-gpu"],
```

Luego reconstruye el contenedor:

```
F1  →  Dev Containers: Rebuild Container
```

---

### Opción B — Terminal sin VSCode

Si prefieres no usar VSCode, puedes arrancar el contenedor directamente y acceder a JupyterLab desde el navegador.

**Arrancar contenedor CPU:**

```bash
docker compose --profile cpu up -d
```

**Arrancar contenedor GPU:**

```bash
docker compose --profile gpu up -d
```

El flag `-d` arranca el contenedor en background. JupyterLab estará disponible en [http://localhost:8888](http://localhost:8888) en unos segundos.

**Ver logs en tiempo real** (útil para confirmar que JupyterLab arrancó bien):

```bash
docker compose logs -f dev-cpu   # o dev-gpu
```

Deberías ver una línea como:

```
[JupyterServerApp] Jupyter Server 2.x.x is running at:
[JupyterServerApp] http://0.0.0.0:8888/lab
```

**Abrir una shell interactiva en el contenedor** (para ejecutar scripts, jellyfish, etc.):

```bash
docker compose exec dev-cpu bash   # o dev-gpu
```

**Parar el contenedor** (los datos en volúmenes no se pierden):

```bash
docker compose --profile cpu down
```

---

### 7. Verificar la instalación

Desde un terminal dentro del contenedor (ya sea el de VSCode o el de `docker compose exec`):

```bash
# Verificar versiones de las dependencias ML principales
python -c "
import sklearn, xgboost, pytorch_tabnet, shap, Bio, torch
print(f'scikit-learn  : {sklearn.__version__}')
print(f'xgboost       : {xgboost.__version__}')
print(f'pytorch-tabnet: ok')
print(f'shap          : {shap.__version__}')
print(f'biopython     : {Bio.__version__}')
print(f'torch         : {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Verificar jellyfish (conteo de k-mers)
jellyfish --version

# Verificar que los datos están montados correctamente
ls /workspace/datos/brutos/
```

La salida esperada de `torch.cuda.is_available()` es `False` en la imagen CPU y `True` en la imagen GPU (con una GPU NVIDIA detectada).

---

### Setup GPU (opcional)

La imagen GPU requiere configuración adicional en el host **antes** de construirla.

#### Requisitos de hardware y software

- GPU NVIDIA con arquitectura Turing o superior (RTX 20xx / RTX 30xx / RTX 40xx / A100 / H100)
- Driver NVIDIA **>= 525** instalado en el host

#### Paso 1 — Verificar el driver en el host

```bash
# En WSL2 o Linux nativo
nvidia-smi
```

La salida debe mostrar tu GPU y el driver instalado. Si el comando falla, instala o actualiza el driver NVIDIA en el sistema host (no dentro de WSL).

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.x         Driver Version: 535.x    CUDA Version: 12.2      |
|-------------------------------+---------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| 0   RTX 3080         Off     | 00000000:01:00.0  On |                  N/A |
```

#### Paso 2 — Instalar NVIDIA Container Toolkit

Sigue la guía oficial para tu sistema operativo:

- **Ubuntu / WSL2:** [docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Resumen para Ubuntu/WSL2:

```bash
# Añadir repositorio NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Instalar
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configurar Docker para usar el runtime NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker

# Reiniciar Docker
sudo systemctl restart docker
```

#### Paso 3 — Verificar que Docker ve la GPU

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Si este comando muestra tu GPU, el toolkit está correctamente instalado. Si falla, no sigas adelante — la imagen GPU no funcionará.

#### Paso 4 — Construir y arrancar la imagen GPU

```bash
docker compose --profile gpu build
docker compose --profile gpu up -d
```

#### Verificar CUDA dentro del contenedor

```bash
docker compose exec dev-gpu python -c "
import torch
print(f'CUDA disponible : {torch.cuda.is_available()}')
print(f'GPU detectada   : {torch.cuda.get_device_name(0)}')
print(f'VRAM total      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

---

## Descarga de datos

Los datos **no están versionados** en este repositorio por razones de tamaño y licencia de BV-BRC. Una vez el entorno esté funcionando, ejecuta el notebook `01_descarga_datos.ipynb` para descargarlos automáticamente vía la API de BV-BRC.

```bash
# Asegúrate de que los directorios existen en el host antes de descargar
ls ~/amr-data/brutos/genomas_fasta   # debe existir (creado en el paso 4)
```

| Recurso | Tiempo estimado | Espacio en disco |
|---|---|---|
| Fenotipos AMR (CSV) | < 1 min | ~5 MB |
| Genes AMR (CSV) | < 2 min | ~20 MB |
| 500 genomas FASTA | 2–4 horas | ~3–5 GB |

---

## Visión general del pipeline

```
API BV-BRC
    │
    ├── endpoint genome_amr        →  amr_fenotipos.csv    (etiquetas S/I/R + MIC)
    └── endpoint genome_sequence   →  genomas_fasta/        (contigs WGS)
            │
            ├── [QC] N50 > 20 kb · contigs < 500 · GC 50–64%
            │
            ├── Conteo de k-mers (k=31, jellyfish)
            │       └── Top 100k k-mers por document frequency
            │
            ├── Detección de genes AMR (endpoint specialty_gene)
            │       └── Matriz pivote: genoma × gen (0/1)
            │
            └── Matriz de features  (sparse CSR, ~5000 × 100k+)
                    │
                    ├── VarianceThreshold + Chi2 SelectKBest(50k)
                    ├── MaxAbsScaler (preserva sparsidad)
                    ├── StratifiedKFold(n=5)  ← evita data leakage
                    │
                    ├── Random Forest      (class_weight='balanced')
                    ├── XGBoost            (scale_pos_weight)
                    ├── TabNet             (atención secuencial — SOTA)
                    │
                    └── Evaluación: AUC-ROC · F1-macro · VME/ME
                            ├── SHAP TreeExplainer   (RF + XGBoost)
                            └── Máscaras de atención (TabNet)
```

---

## Resultados

> Sección pendiente — se actualizará tras completar el entrenamiento.

| Modelo | AUC-ROC | F1-macro | VME (%) | ME (%) |
|---|---|---|---|---|
| Baseline (most_frequent) | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| **TabNet** | — | — | — | — |

**VME** (very major error): Resistente clasificado como Susceptible — el error de mayor riesgo clínico.  
**ME** (major error): Susceptible clasificado como Resistente.

---

## Solución de problemas

#### `docker compose` no reconoce el flag `--profile`

Estás usando la versión antigua (`docker-compose` con guión). Este proyecto requiere Docker Compose V2 (`docker compose` sin guión). Actualiza Docker Desktop o instala el plugin manualmente.

#### JupyterLab no carga en el navegador

Comprueba que el contenedor está corriendo y sano:

```bash
docker compose ps
```

El campo `Status` debe mostrar `healthy`. Si muestra `starting`, espera 30 segundos y vuelve a comprobar. Si muestra `unhealthy`, revisa los logs:

```bash
docker compose logs dev-cpu
```

#### El kernel de Jupyter aparece como "dead" o no arranca

El kernel usa el Python del contenedor. Si abres un notebook desde fuera del contenedor (por ejemplo desde un JupyterLab local), el kernel no existirá. Abre siempre los notebooks desde dentro del Dev Container en VSCode o desde el JupyterLab que corre en el contenedor.

#### `jellyfish: command not found` dentro del contenedor

Estás usando la imagen antigua. Reconstruye con:

```bash
docker compose --profile cpu build --no-cache
```

#### En WSL2, el montaje de datos es muy lento

El volumen `AMR_DATA_DIR` apunta a una ruta de Windows (`/mnt/c/...`). Mueve los datos al filesystem de WSL (`~/amr-data`) y actualiza `.env`. La diferencia de velocidad en operaciones I/O intensivas puede ser de 10x.

#### `nvidia-smi` falla dentro del contenedor GPU

El NVIDIA Container Toolkit no está instalado correctamente o Docker no fue reiniciado tras la instalación. Sigue los pasos de la sección [Setup GPU](#setup-gpu-opcional) desde el principio.

---

## Referencias

1. Moradigaravand et al. (2018). *Prediction of antibiotic resistance in Escherichia coli from large-scale pan-genome data*. PLOS Comput Biol. https://doi.org/10.1371/journal.pcbi.1006258
2. Nguyen et al. (2019). *Using machine learning to predict antimicrobial MICs and associated genomic features for nontyphoidal Salmonella*. J Clin Microbiol. https://doi.org/10.1128/JCM.01260-18
3. Arik, S. & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI. https://arxiv.org/abs/1908.07442
4. Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS. https://arxiv.org/abs/2106.11959
5. BV-BRC: Bacterial and Viral Bioinformatics Resource Center. https://www.bv-brc.org/
6. EUCAST Clinical Breakpoint Tables v14.0. https://www.eucast.org/clinical_breakpoints/

---

<p align="center">
  Proyecto académico · <em>Ciencia de datos aplicada al ámbito biosanitario</em>
</p>

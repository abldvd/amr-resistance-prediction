# Predicción de Resistencia Antibiótica (AMR)

### Aprendizaje automático sobre datos de secuenciación genómica completa

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-required-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Licencia](https://img.shields.io/badge/Licencia-MIT-green?style=flat-square)](LICENSE)
[![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-orange?style=flat-square)]()

Predicción de fenotipos de resistencia antibiótica (Susceptible / Intermedio / Resistente) en *Klebsiella pneumoniae* a partir de datos de secuenciación genómica completa (WGS) procedentes de [BV-BRC](https://www.bv-brc.org/). El pipeline combina vectores de frecuencia de k-mers, matrices de presencia/ausencia de genes AMR y tres modelos de ML supervisado — Random Forest, XGBoost y **TabNet** — con interpretabilidad basada en valores SHAP y atención aprendida.

> **Entorno de desarrollo:** Windows 11 · Docker Desktop · VSCode Dev Containers · Python 3.11 · JupyterLab

---

## Índice

- [Contexto científico](#contexto-científico)
- [Modelos](#modelos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Entorno de desarrollo](#entorno-de-desarrollo)
  - [Requisitos previos](#1-requisitos-previos)
  - [Primeros pasos](#2-primeros-pasos)
  - [Arrancar el contenedor](#3-arrancar-el-contenedor)
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
├── data/
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

El entorno corre dentro de un contenedor Docker. En el host solo necesitas Docker Desktop y VSCode — nada más.

**Tiempo estimado de setup inicial:** 15–25 min (PyTorch pesa ~2 GB en descarga).

---

### 1. Requisitos previos

| Herramienta | Descarga | Notas |
|---|---|---|
| Docker Desktop 4.x | [docs.docker.com/desktop/windows](https://docs.docker.com/desktop/windows/install/) | Durante la instalación, usa el backend **Hyper-V**, no WSL2 |
| VSCode | [code.visualstudio.com](https://code.visualstudio.com/) | |
| Extensión Dev Containers | `Ctrl+Shift+X` → busca `ms-vscode-remote.remote-containers` | |
| Git para Windows | [git-scm.com](https://git-scm.com/download/win) | |

Tras instalar Docker, verifica en PowerShell:

```powershell
docker --version          # Docker version 26.x.x
docker compose version    # Docker Compose version v2.x.x
docker run --rm hello-world
```

---

### 2. Primeros pasos

Abre PowerShell y ejecuta:

```powershell
# Clonar el repositorio
git clone https://github.com/<tu-usuario>/amr-resistance-prediction.git
cd amr-resistance-prediction

# Crear el .env desde la plantilla
copy .env.example .env

# Crear la estructura de datos dentro del proyecto
mkdir data\brutos\genomas_fasta
mkdir data\procesados
```

El `.env` no necesita cambios para empezar — `AMR_DATA_DIR` ya apunta a `./data` por defecto.

---

### 3. Arrancar el contenedor

**Opción A — VSCode Dev Containers (recomendado)**

```powershell
code .
```

VSCode detectará `.devcontainer/devcontainer.json` y mostrará la notificación *"Reopen in Container"* en la esquina inferior derecha. Haz clic en ella. Si no aparece:

```
F1  →  Dev Containers: Reopen in Container
```

VSCode construirá la imagen (solo la primera vez), arrancará el contenedor y abrirá una ventana conectada a él. El terminal integrado ya está dentro del contenedor. La esquina inferior izquierda mostrará `>< Dev Container: AMR — CPU`.

Para abrir JupyterLab:

```bash
jupyter lab   # ejecuta esto en el terminal integrado de VSCode
```

El puerto se reenvía automáticamente → [http://localhost:8888](http://localhost:8888).

**Opción B — Solo terminal**

```powershell
docker compose --profile cpu up -d          # arranca en background
docker compose logs -f dev-cpu              # sigue los logs
docker compose exec dev-cpu bash            # shell dentro del contenedor
docker compose --profile cpu down           # para el contenedor
```

**Verificar que todo funciona** (desde el terminal del contenedor):

```bash
python -c "import sklearn, xgboost, shap, torch; print('OK')"
jellyfish --version
ls /workspace/datos/brutos/
```

---

### Setup GPU (opcional)

Requiere GPU NVIDIA (arquitectura Turing o superior) y driver **>= 525** instalado en Windows.

**1. Verifica el driver en PowerShell:**

```powershell
nvidia-smi   # debe mostrar tu GPU y la versión del driver
```

**2. Habilita soporte GPU en Docker Desktop:**

Docker Desktop en Windows con Hyper-V expone la GPU al contenedor sin necesidad de instalar el NVIDIA Container Toolkit manualmente — ya está integrado. Solo asegúrate de tener Docker Desktop **>= 4.x** y el driver NVIDIA actualizado.

**3. Verifica que Docker ve la GPU:**

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**4. Construye y arranca la imagen GPU:**

```powershell
docker compose --profile gpu build
docker compose --profile gpu up -d
```

Para cambiar el Dev Container de CPU a GPU, edita dos líneas en `.devcontainer/devcontainer.json`:

```jsonc
"service": "dev-gpu",       // antes: dev-cpu
"runServices": ["dev-gpu"], // antes: ["dev-cpu"]
```

Luego: `F1 → Dev Containers: Rebuild Container`.

**Verificar CUDA dentro del contenedor:**

```bash
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU : {torch.cuda.get_device_name(0)}')
"
```

---

## Descarga de datos

Los datos **no están versionados** en este repositorio por razones de tamaño y licencia de BV-BRC. Una vez el entorno esté funcionando, ejecuta el notebook `01_descarga_datos.ipynb` para descargarlos automáticamente vía la API de BV-BRC. Se guardarán en `data/brutos/` dentro del propio proyecto.

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

Estás usando la versión antigua (`docker-compose` con guión). Este proyecto requiere Docker Compose V2 (`docker compose` sin guión). Actualiza Docker Desktop.

#### JupyterLab no carga en el navegador

Comprueba que el contenedor está corriendo y sano:

```powershell
docker compose ps
```

El campo `Status` debe mostrar `healthy`. Si muestra `unhealthy`, revisa los logs:

```powershell
docker compose logs dev-cpu
```

#### El kernel de Jupyter aparece como "dead"

El kernel usa el Python del contenedor. Abre siempre los notebooks desde dentro del Dev Container en VSCode o desde el JupyterLab que corre en el contenedor — nunca desde un Jupyter local del host.

#### `jellyfish: command not found` dentro del contenedor

Imagen desactualizada. Reconstruye forzando sin caché:

```powershell
docker compose --profile cpu build --no-cache
```

#### Docker no puede montar la ruta de datos

Los datos están en `./data` dentro del proyecto. Docker Desktop necesita acceso a la unidad donde está clonado el repositorio. En Docker Desktop → Settings → Resources → File Sharing, verifica que la unidad (por ejemplo `C:`) aparece en la lista.

#### `nvidia-smi` falla dentro del contenedor GPU

Verifica que el driver NVIDIA está actualizado en Windows y que Docker Desktop es versión 4.x o superior. Prueba primero el comando de verificación antes de construir la imagen GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

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

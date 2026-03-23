# Predicción de Resistencia Antibiótica (AMR)

### Aprendizaje automático sobre datos de secuenciación genómica completa

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-required-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Dev Container](https://img.shields.io/badge/Dev%20Container-ready-0DB7ED?style=flat-square&logo=visualstudiocode)](https://containers.dev/)
[![Licencia](https://img.shields.io/badge/Licencia-MIT-green?style=flat-square)](LICENSE)
[![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-orange?style=flat-square)]()

Predicción de fenotipos de resistencia antibiótica (Susceptible / Resistente) en *Klebsiella pneumoniae* a partir de datos de secuenciación genómica completa (WGS) procedentes de [BV-BRC](https://www.bv-brc.org/). El pipeline combina vectores de frecuencia de k-mers, matrices de presencia/ausencia de genes AMR y tres modelos de ML supervisado — Random Forest, XGBoost y **TabNet** — con interpretabilidad basada en valores SHAP y atención aprendida.

> **Stack:** Docker · VSCode Dev Containers · pip · Python 3.11 · PyTorch · JupyterLab  
> **Plataforma principal:** Windows 11 + WSL2

---

## Índice

- [Contexto científico](#contexto-científico)
- [Modelos](#modelos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Guía de instalación paso a paso](#guía-de-instalación-paso-a-paso)
  - [Paso 1 — Verificar Docker y WSL2](#paso-1--verificar-docker-y-wsl2)
  - [Paso 2 — Instalar la extensión Dev Containers](#paso-2--instalar-la-extensión-dev-containers)
  - [Paso 3 — Clonar el repositorio](#paso-3--clonar-el-repositorio)
  - [Paso 4 — Abrir el proyecto en el contenedor](#paso-4--abrir-el-proyecto-en-el-contenedor)
  - [Paso 5 — Verificar que todo funciona](#paso-5--verificar-que-todo-funciona)
  - [Paso 6 — Abrir JupyterLab](#paso-6--abrir-jupyterlab)
- [Setup GPU (opcional)](#setup-gpu-opcional)
- [Descarga de datos](#descarga-de-datos)
- [Pipeline](#pipeline)
- [Resultados](#resultados)
- [Makefile — referencia de comandos](#makefile--referencia-de-comandos)
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
│   ├── devcontainer.json          # configuración Dev Container
│   └── post-create.sh             # setup automático tras crear el contenedor
│
├── requirements/
│   ├── base.txt                   # dependencias comunes (scikit-learn, pandas…)
│   ├── dev.txt                    # herramientas de desarrollo (ruff, pytest)
│   ├── torch-cpu.txt              # PyTorch CPU-only
│   └── torch-gpu.txt              # PyTorch CUDA 12.1
│
├── datos/
│   ├── brutos/                    # datos BV-BRC — no versionados (.gitignore)
│   │   ├── genomas_fasta/         # secuencias WGS por lotes (.fasta)
│   │   ├── amr_fenotipos.csv      # etiquetas S/I/R + valores MIC
│   │   └── amr_matriz_genes.csv   # presencia/ausencia de genes AMR
│   └── procesados/                # outputs del preprocesado — no versionados
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
├── modelos/                       # modelos serializados — no versionados
├── figuras/                       # gráficos exportados — no versionados
│
├── Dockerfile                     # multi-stage, soporta DEVICE=cpu|gpu
├── docker-compose.yml             # perfiles: cpu · gpu
├── Makefile                       # atajos: make up, make jupyter, make shell…
├── .env.example                   # plantilla de variables de entorno
├── .gitignore
└── README.md
```

---

## Guía de instalación paso a paso

> **Punto de partida:** tienes instalado **VSCode**, **Docker Desktop** y **WSL2** en Windows 11.  
> No necesitas Python, pip, ni ninguna otra herramienta en tu máquina — todo corre dentro del contenedor.

**Tiempo estimado:** 15–25 minutos (la primera vez, porque PyTorch pesa ~2 GB).

---

### Paso 1 — Verificar Docker y WSL2

Abre **PowerShell** (o Windows Terminal) y ejecuta estos comandos:

```powershell
wsl --version
```

Deberías ver algo como:

```
Versión de WSL: 2.x.x.x
Versión de kernel: 5.15.x
```

Si dice `wsl: command not found` o muestra la versión 1, ejecuta `wsl --install` y reinicia Windows.

```powershell
docker --version
```

```
Docker version 27.x.x, build xxxxxxx
```

```powershell
docker compose version
```

```
Docker Compose version v2.x.x
```

Ahora verifica que Docker Desktop está configurado con WSL2. Abre **Docker Desktop** y comprueba:

```
Settings  →  General  →  ✅ "Use the WSL 2 based engine"
```

```
Settings  →  Resources  →  WSL Integration  →  ✅ "Enable integration with my default WSL distro"
```

Si has hecho cambios, pulsa **Apply & restart**.

Confirma que Docker funciona:

```powershell
docker run --rm hello-world
```

Deberías ver `Hello from Docker!`. Si falla, reinicia Docker Desktop y vuelve a intentar.

---

### Paso 2 — Instalar la extensión Dev Containers

Abre **VSCode** y pulsa `Ctrl+Shift+X` para abrir el panel de extensiones. Busca e instala:

```
Dev Containers
```

El autor es **Microsoft** (ID: `ms-vscode-remote.remote-containers`). Solo necesitas esta extensión — las demás (Python, Jupyter, Ruff, GitLens…) se instalarán automáticamente dentro del contenedor.

---

### Paso 3 — Clonar el repositorio

Abre **PowerShell** y ejecuta:

```powershell
git clone https://github.com/<tu-usuario>/amr-resistance-prediction.git
cd amr-resistance-prediction
copy .env.example .env
```

> **Tip de rendimiento:** si más adelante notas que el contenedor va lento accediendo a ficheros, puedes mover el repositorio al filesystem de WSL. Abre una terminal WSL (escribe `wsl` en PowerShell) y clona en `~/`:
>
> ```bash
> cd ~
> git clone https://github.com/<tu-usuario>/amr-resistance-prediction.git
> ```
>
> Luego en VSCode usa `F1 → Dev Containers: Open Folder in Container` y navega a `\\wsl$\Ubuntu\home\<tu-usuario>\amr-resistance-prediction`.

---

### Paso 4 — Abrir el proyecto en el contenedor

Abre el directorio del proyecto en VSCode:

```powershell
code .
```

VSCode detectará la carpeta `.devcontainer/` y mostrará una notificación en la esquina inferior derecha:

```
📦 Folder contains a Dev Container configuration file.
   Reopen in Container
```

**Haz clic en "Reopen in Container".**

Si la notificación no aparece, usa la paleta de comandos:

```
F1  →  Dev Containers: Reopen in Container
```

**Lo que ocurre ahora (todo automático, no tienes que hacer nada):**

1. **Docker construye la imagen** — descarga Python 3.11, instala PyTorch, scikit-learn, XGBoost, SHAP, JupyterLab, jellyfish y el resto de dependencias. **Esto tarda 15–20 minutos la primera vez** porque PyTorch pesa ~2 GB. Puedes seguir el progreso en la barra inferior de VSCode — haz clic en *"Starting Dev Container (show log)"* para ver los detalles.

2. **Docker arranca el contenedor** con tu código montado dentro.

3. **Se ejecuta `post-create.sh`** — crea los directorios de datos, modelos y figuras, verifica las dependencias y configura git.

4. **VSCode se reconecta al contenedor** — instala las extensiones (Python, Jupyter, Ruff, GitLens) dentro del contenedor automáticamente.

Cuando termine, la esquina inferior izquierda de VSCode mostrará:

```
>< Dev Container: AMR Prediction
```

**Ya estás dentro del contenedor.** El terminal integrado (`Ctrl+ñ` o `` Ctrl+` ``) ejecuta comandos dentro del contenedor Linux, no en tu Windows.

---

### Paso 5 — Verificar que todo funciona

Abre el terminal integrado de VSCode (`Ctrl+ñ`) y ejecuta:

```bash
make check
```

Deberías ver:

```
  scikit-learn : 1.x.x
  xgboost      : 2.x.x
  shap         : 0.4x.x
  pytorch      : 2.x.x
  cuda         : False
  jellyfish    : 2.x.x
✓ Dependencias OK
```

`cuda: False` es correcto si no tienes GPU NVIDIA — el perfil CPU funciona perfectamente.

---

### Paso 6 — Abrir JupyterLab

Tienes dos opciones para trabajar con notebooks:

**Opción A — Notebooks directamente en VSCode (recomendado)**

Haz doble clic en cualquier `.ipynb` del explorador de archivos de VSCode. El kernel de Jupyter ya está disponible dentro del contenedor — no necesitas lanzar nada.

**Opción B — JupyterLab en el navegador**

Desde el terminal integrado de VSCode:

```bash
make jupyter
```

Abre http://localhost:8888 en tu navegador. JupyterLab se abre sin pedir token ni contraseña. Para pararlo: `Ctrl+C` en el terminal.

---

### Resumen visual

```
Windows (tu máquina)
│
├── Docker Desktop (motor WSL2)
│   └── Contenedor Linux
│       ├── Python 3.11 + pip
│       ├── PyTorch, scikit-learn, XGBoost, SHAP
│       ├── JupyterLab (puerto 8888)
│       ├── jellyfish (conteo de k-mers)
│       └── Tu código (montado desde Windows)
│
└── VSCode
    ├── Extensión Dev Containers → conectado al contenedor
    ├── Python, Jupyter, Ruff → corren DENTRO del contenedor
    └── Terminal integrado → bash del contenedor
```

---

## Setup GPU (opcional)

Requiere GPU NVIDIA con driver ≥ 525 instalado **en Windows** (no dentro de WSL).

**1. Verifica el driver en PowerShell:**

```powershell
nvidia-smi
```

Debe mostrar tu GPU y la versión del driver. Si falla, descarga el driver desde [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx).

**2. Verifica que Docker ve la GPU:**

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Si funciona, ya tienes soporte GPU. WSL2 + Docker Desktop lo manejan automáticamente.

**3. Cambia el Dev Container a GPU:**

Edita `.devcontainer/devcontainer.json` y cambia dos líneas:

```jsonc
"service": "amr-dev-gpu",          // antes: amr-dev
"runServices": ["amr-dev-gpu"],    // antes: ["amr-dev"]
```

Luego: `F1 → Dev Containers: Rebuild Container`. La primera vez construirá una imagen con PyTorch+CUDA (~3 GB extra).

**4. Verificar CUDA dentro del contenedor:**

```bash
python -c "
import torch
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## Descarga de datos

Los datos **no están incluidos** en el repositorio por tamaño y licencia de BV-BRC. Una vez el entorno esté funcionando, ejecuta el notebook `01_descarga_datos.ipynb` para descargarlos vía la API de BV-BRC.

| Recurso | Tiempo estimado | Espacio en disco |
|---|---|---|
| Fenotipos AMR (CSV) | < 1 min | ~5 MB |
| Genes AMR (CSV) | < 2 min | ~20 MB |
| ~500 genomas FASTA | 2–4 horas | ~3–5 GB |

---

## Pipeline

```
API BV-BRC
    │
    ├── endpoint genome_amr         →  amr_fenotipos.csv     (etiquetas S/I/R + MIC)
    └── endpoint genome_sequence    →  genomas_fasta/          (contigs WGS)
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

## Makefile — referencia de comandos

| Comando | Descripción |
|---|---|
| `make up` | Construye imagen CPU y arranca el contenedor |
| `make up-gpu` | Construye imagen GPU y arranca el contenedor |
| `make down` | Para y elimina los contenedores |
| `make shell` | Abre bash dentro del contenedor |
| `make jupyter` | Lanza JupyterLab en http://localhost:8888 |
| `make check` | Verifica dependencias instaladas |
| `make logs` | Sigue los logs del contenedor |
| `make build` | Solo construye la imagen (sin arrancar) |
| `make clean` | Elimina contenedores, volúmenes y caché del proyecto |

---

## Solución de problemas

#### La notificación "Reopen in Container" no aparece en VSCode

Verifica que la extensión **Dev Containers** está instalada (`Ctrl+Shift+X` → busca `Dev Containers`). Después: `F1 → Dev Containers: Reopen in Container`.

#### El build de la imagen se queda colgado o es muy lento

PyTorch pesa ~2 GB. Si tu conexión es lenta, la descarga puede tardar bastante. Verifica los logs: `F1 → Dev Containers: Show Log`. Si ves errores de red, reinicia Docker Desktop y vuelve a intentar con `F1 → Dev Containers: Rebuild Container`.

#### `docker compose` no reconoce `--profile`

Estás usando Docker Compose V1 (`docker-compose` con guión). Este proyecto requiere V2 (`docker compose` sin guión). Actualiza Docker Desktop a la última versión.

#### JupyterLab no carga en http://localhost:8888

```bash
docker compose ps
```

Si `Status` no muestra `healthy`, revisa los logs con `make logs`.

#### El kernel de Jupyter aparece como "dead"

Abre los notebooks desde dentro del Dev Container en VSCode o desde el JupyterLab del contenedor — nunca desde un Jupyter local instalado en Windows.

#### `jellyfish: command not found`

Imagen desactualizada. Reconstruye sin caché:

```bash
make build ARGS="--no-cache"
```

#### Performance lenta en Windows

Docker Desktop debe usar WSL2 (Settings → General → *"Use the WSL 2 based engine"*). Para mejor rendimiento, clona el repo dentro del filesystem de WSL en vez de en `C:\`:

```bash
# En terminal WSL
cd ~ && git clone https://github.com/<tu-usuario>/amr-resistance-prediction.git
```

#### `nvidia-smi` falla dentro del contenedor GPU

Verifica el driver NVIDIA en **Windows** (no dentro de WSL). Prueba primero:

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

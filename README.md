# Predicción de Resistencia Antibiótica (AMR)

### Aprendizaje automático sobre datos de secuenciación genómica completa

[!\[Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square\&logo=python)](https://www.python.org/)
[!\[Licencia](https://img.shields.io/badge/Licencia-MIT-green?style=flat-square)](LICENSE)
\[!\[Estado](https://img.shields.io/badge/Estado-En%20desarrollo-orange?style=flat-square)]()
[!\[Plataforma](https://img.shields.io/badge/Plataforma-WSL2%20%2B%20Ubuntu-E95420?style=flat-square\&logo=ubuntu)](https://ubuntu.com/wsl)

Predicción de fenotipos de resistencia antibiótica (Susceptible / Intermedio / Resistente) en *Klebsiella pneumoniae* a partir de datos de secuenciación genómica completa (WGS) procedentes de la base de datos [BV-BRC](https://www.bv-brc.org/). El pipeline combina vectores de frecuencia de k-mers, matrices de presencia/ausencia de genes AMR y tres modelos de ML supervisado — Random Forest, XGBoost y **TabNet** — con interpretabilidad basada en valores SHAP y atención aprendida.

> \*\*Entorno de desarrollo:\*\* WSL2 (Ubuntu 22.04) · Miniconda · JupyterLab

\---

## Índice

* [Contexto científico](#contexto-científico)
* [Modelos](#modelos)
* [Estructura del proyecto](#estructura-del-proyecto)
* [Configuración del entorno](#configuración-del-entorno)
* [Uso](#uso)
* [Descarga de datos](#descarga-de-datos)
* [Visión general del pipeline](#visión-general-del-pipeline)
* [Resultados](#resultados)
* [Referencias](#referencias)

\---

## Contexto científico

La resistencia antimicrobiana (AMR) es una de las amenazas sanitarias globales más urgentes, con una proyección de 10 millones de muertes anuales para 2050 (OMS, 2019). Las pruebas fenotípicas tradicionales (microdilución en caldo) tardan entre 24 y 72 horas y requieren cultivo vivo. La secuenciación genómica completa (WGS) permite predecir la resistencia en menos de 2 horas a partir de un aislado clínico, posibilitando una terapia dirigida más rápida.

Este proyecto replica y amplía el enfoque de predicción de resistencia basado en k-mers descrito en:

> Moradigaravand et al. (2018). \*Prediction of antibiotic resistance in Escherichia coli from large-scale pan-genome data\*. PLOS Computational Biology.

**Organismo objetivo:** *Klebsiella pneumoniae* (patógeno de prioridad crítica según la OMS)  
**Antibiótico objetivo:** Ciprofloxacino (fluoroquinolona — terapia de primera línea)  
**Tarea:** Clasificación binaria — Susceptible (S) vs Resistente (R)

\---

## Modelos

Se comparan tres enfoques de complejidad creciente sobre la misma matriz de features.

### Random Forest — baseline sólido

Ensemble de árboles de decisión con bootstrap. Es el modelo de referencia estándar en la literatura AMR-ML por su robustez ante features irrelevantes y su facilidad de interpretación vía importancia de Gini. Se usa con `class\_weight='balanced'` para compensar el desbalance S:R.

### XGBoost — gradient boosting optimizado

Gradient boosting con regularización L1/L2, manejo nativo de sparsidad y paralelización eficiente. En datos genómicos de alta dimensionalidad supera consistentemente a Random Forest en AUC-ROC gracias a la construcción secuencial de árboles que corrige los errores del modelo anterior. Se usa `scale\_pos\_weight` para el desbalance de clases.

### TabNet — red neuronal con atención secuencial *(SOTA)*

> Arik, S. \& Pfister, T. (2021). \*TabNet: Attentive Interpretable Tabular Learning\*. AAAI. Google Research.

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
        │  ...  (N\_steps = 5 por defecto)
        ▼
  Agregación de pasos
        │
        ▼
  Clasificador final → P(Resistente)
```

\---

## Estructura del proyecto

```
amr-resistance-prediction/
│
├── datos/
│   ├── brutos/                      # datos BV-BRC — no versionados (.gitignore)
│   │   ├── genomas\_fasta/           # secuencias WGS por lotes (.fasta)
│   │   ├── amr\_fenotipos.csv        # etiquetas S/I/R + valores MIC
│   │   └── amr\_matriz\_genes.csv     # presencia/ausencia de genes AMR
│   └── procesados/                  # outputs del preprocesado — no versionados
│       ├── informe\_qc\_genomas.csv
│       ├── matriz\_kmers.npz
│       ├── X\_final.npz
│       └── y\_final.npy
│
├── notebooks/
│   ├── 01\_descarga\_datos.ipynb           # descarga API BV-BRC
│   ├── 02\_control\_calidad\_genomica.ipynb # QC de ensamblajes FASTA
│   ├── 03\_variable\_target.ipynb          # procesamiento S/I/R y MIC
│   ├── 04\_extraccion\_kmers.ipynb         # vectorización k-mer (k=31)
│   ├── 05\_matriz\_features.ipynb          # integración y selección de features
│   ├── 06\_preprocesado.ipynb             # normalización, split, validación
│   ├── 07\_entrenamiento\_rf\_xgb.ipynb     # Random Forest + XGBoost + CV
│   ├── 08\_entrenamiento\_tabnet.ipynb     # TabNet con pytorch-tabnet
│   └── 09\_interpretabilidad.ipynb        # SHAP + máscaras de atención TabNet
│
├── informes/
│   └── figuras/                     # gráficos EDA, curvas ROC, SHAP plots
│
├── environment.yml                  # entorno conda reproducible
├── .gitignore
└── README.md
```

\---

## Configuración del entorno

### 1\. Clonar el repositorio

> Clona dentro del filesystem de WSL (`\~/`), no en `/mnt/c/`. El acceso cruzado es más lento para operaciones I/O intensivas como el procesado de FASTA.

```bash
cd \~
git clone https://github.com/<tu-usuario>/amr-resistance-prediction.git
cd amr-resistance-prediction
```

### 2\. Crear el entorno conda

```bash
conda env create -f environment.yml
conda activate amr-pred
```

### 3\. Registrar el kernel en JupyterLab

```bash
python -m ipykernel install --user \\
  --name amr-pred \\
  --display-name "AMR Prediction (py3.11)"
```

\---


## Descarga de datos

Los datos **no están versionados** en este repositorio por razones de tamaño y licencia de BV-BRC. Para reproducir el estudio desde cero, crear las carpetas y ejecutar el notebook `01`:

```bash
mkdir -p datos/brutos/genomas\_fasta datos/procesados informes/figuras
```

|Recurso|Tiempo estimado|Espacio en disco|
|-|-|-|
|Fenotipos AMR (CSV)|< 1 minuto|\~5 MB|
|Genes AMR (CSV)|< 2 minutos|\~20 MB|
|500 genomas FASTA|2–4 horas|\~3–5 GB|

\---

## Visión general del pipeline

```
API BV-BRC
    │
    ├── endpoint genome\_amr        →  amr\_fenotipos.csv    (etiquetas S/I/R + MIC)
    └── endpoint genome\_sequence   →  genomas\_fasta/        (contigs WGS)
            │
            ├── \[QC] N50 > 20 kb · contigs < 500 · GC 50–64 %
            │
            ├── Conteo de k-mers (k=31, jellyfish)
            │       └── Top 100 k k-mers por document frequency
            │
            ├── Detección de genes AMR (endpoint specialty\_gene)
            │       └── Matriz pivote: genoma × gen (0/1)
            │
            └── Matriz de features  (sparse CSR, \~5000 × 100 k+)
                    │
                    ├── VarianceThreshold + Chi2 SelectKBest(50 k)
                    ├── MaxAbsScaler (preserva sparsidad)
                    ├── StratifiedKFold(n=5)  ← evita data leakage
                    │
                    ├── Random Forest      (class\_weight='balanced')
                    ├── XGBoost            (scale\_pos\_weight)
                    ├── TabNet             (atención secuencial — SOTA)
                    │
                    └── Evaluación: AUC-ROC · F1-macro · VME/ME
                            ├── SHAP TreeExplainer   (RF + XGBoost)
                            └── Máscaras de atención (TabNet)
```

\---

## Resultados

> Sección pendiente — se actualizará tras completar el entrenamiento.

|Modelo|AUC-ROC|F1-macro|VME (%)|ME (%)|
|-|-|-|-|-|
|Baseline (most\_frequent)|—|—|—|—|
|Random Forest|—|—|—|—|
|XGBoost|—|—|—|—|
|**TabNet**|—|—|—|—|

**VME** (very major error): Resistente clasificado como Susceptible — el error de mayor riesgo clínico.  
**ME** (major error): Susceptible clasificado como Resistente.

\---

## Referencias

1. Moradigaravand et al. (2018). *Prediction of antibiotic resistance in Escherichia coli from large-scale pan-genome data*. PLOS Comput Biol. https://doi.org/10.1371/journal.pcbi.1006258
2. Nguyen et al. (2019). *Using machine learning to predict antimicrobial MICs and associated genomic features for nontyphoidal Salmonella*. J Clin Microbiol. https://doi.org/10.1128/JCM.01260-18
3. Arik, S. \& Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI. https://arxiv.org/abs/1908.07442
4. Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS. https://arxiv.org/abs/2106.11959
5. BV-BRC: Bacterial and Viral Bioinformatics Resource Center. https://www.bv-brc.org/
6. EUCAST Clinical Breakpoint Tables v14.0. https://www.eucast.org/clinical\_breakpoints/

\---

<p align="center">
  Proyecto académico · <em>Ciencia de datos aplicada al ámbito biosanitario</em>
</p>


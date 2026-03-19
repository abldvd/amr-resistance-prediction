#!/usr/bin/env bash
# =============================================================================
# post-create.sh — Se ejecuta UNA VEZ tras crear el Dev Container
# No uses conda aquí. Todo ya está instalado en la imagen.
# =============================================================================
set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AMR Project — Post-create setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Crear estructura de directorios si no existe
echo "→ Creando estructura de directorios..."
mkdir -p datos/brutos/genomas_fasta
mkdir -p datos/procesados
mkdir -p informes/figuras
mkdir -p src/amr
mkdir -p outputs/models
mkdir -p outputs/figures

# 2. Crear .env a partir del ejemplo si no existe
if [ ! -f .env ]; then
    echo "→ Creando .env desde .env.example..."
    cp .env.example .env
    echo "  ⚠️  Edita .env y añade tu token BV-BRC si lo necesitas."
fi

# 3. Registrar kernel de Jupyter (ya no es conda, es el Python del contenedor)
echo "→ Registrando kernel de Jupyter..."
python -m ipykernel install \
    --user \
    --name amr-pred \
    --display-name "AMR Prediction (py3.11)"

# 4. Verificar dependencias críticas
echo "→ Verificando instalación..."
python -c "
import sklearn, xgboost, pytorch_tabnet, shap, Bio
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  xgboost      : {xgboost.__version__}')
print(f'  pytorch-tabnet: ok')
print(f'  shap         : {shap.__version__}')
print(f'  biopython    : {Bio.__version__}')
"

# 5. Verificar jellyfish (herramienta de k-mers)
echo "→ Verificando jellyfish..."
jellyfish --version || echo "  ⚠️  jellyfish no encontrado — revisa el Dockerfile"

echo ""
echo "✅ Setup completado. JupyterLab disponible en http://localhost:8888"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

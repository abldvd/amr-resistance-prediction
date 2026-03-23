#!/usr/bin/env bash
# .devcontainer/post-create.sh
# Se ejecuta una vez tras crear el contenedor.
set -euo pipefail

echo "╔══════════════════════════════════════════════╗"
echo "║  AMR Resistance Prediction — Post-create     ║"
echo "╚══════════════════════════════════════════════╝"

# Crear directorios si no existen
mkdir -p datos/brutos/genomas_fasta datos/procesados
mkdir -p modelos figuras

# Configurar git
git config --global --add safe.directory /workspace

# Verificación rápida de dependencias
echo ""
echo "Verificando dependencias..."
python -c "
import sklearn, xgboost, shap, torch
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  xgboost      : {xgboost.__version__}')
print(f'  shap         : {shap.__version__}')
print(f'  pytorch      : {torch.__version__}')
print(f'  cuda         : {torch.cuda.is_available()}')
"

if command -v jellyfish &> /dev/null; then
    echo "  jellyfish    : $(jellyfish --version 2>&1 | head -1)"
else
    echo "  jellyfish    : ⚠ no encontrado"
fi

echo ""
echo "✓ Entorno listo."
echo "  → Notebooks: haz doble clic en cualquier .ipynb en el explorador de VSCode"
echo "  → JupyterLab: ejecuta 'make jupyter' en el terminal"

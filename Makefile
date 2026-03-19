# ============================================================================
# Makefile — AMR Resistance Prediction
# Atajos para Docker Compose. Uso: make <target>
# ============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

CYAN := \033[36m
RESET := \033[0m

.PHONY: help up up-gpu down build build-gpu shell jupyter logs check clean

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'

up: ## Construye y arranca el contenedor (CPU)
	docker compose --profile cpu up -d --build

up-gpu: ## Construye y arranca el contenedor (GPU)
	docker compose --profile gpu up -d --build

down: ## Para y elimina los contenedores
	docker compose --profile cpu --profile gpu down

build: ## Solo construye la imagen CPU
	docker compose --profile cpu build $(ARGS)

build-gpu: ## Solo construye la imagen GPU
	docker compose --profile gpu build $(ARGS)

shell: ## Abre bash dentro del contenedor
	@if docker compose ps --format json | grep -q amr-dev-gpu; then \
		docker compose exec amr-dev-gpu bash; \
	else \
		docker compose exec amr-dev bash; \
	fi

jupyter: ## Lanza JupyterLab → http://localhost:8888
	@echo "→ JupyterLab disponible en http://localhost:8888"
	@if docker compose ps --format json | grep -q amr-dev-gpu; then \
		docker compose exec amr-dev-gpu jupyter lab --ip=0.0.0.0 --port=8888 --no-browser; \
	else \
		docker compose exec amr-dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser; \
	fi

logs: ## Sigue los logs del contenedor
	docker compose logs -f

check: ## Verifica dependencias instaladas
	@if docker compose ps --format json | grep -q amr-dev-gpu; then \
		docker compose exec amr-dev-gpu bash -c '\
			python -c "import sklearn, xgboost, shap, torch; \
			print(f\"  scikit-learn : {sklearn.__version__}\"); \
			print(f\"  xgboost      : {xgboost.__version__}\"); \
			print(f\"  shap         : {shap.__version__}\"); \
			print(f\"  pytorch      : {torch.__version__}\"); \
			print(f\"  cuda         : {torch.cuda.is_available()}\")"; \
			echo "  jellyfish    : $$(jellyfish --version 2>&1 | head -1)"; \
			echo "✓ Dependencias OK"'; \
	else \
		docker compose exec amr-dev bash -c '\
			python -c "import sklearn, xgboost, shap, torch; \
			print(f\"  scikit-learn : {sklearn.__version__}\"); \
			print(f\"  xgboost      : {xgboost.__version__}\"); \
			print(f\"  shap         : {shap.__version__}\"); \
			print(f\"  pytorch      : {torch.__version__}\"); \
			print(f\"  cuda         : {torch.cuda.is_available()}\")"; \
			echo "  jellyfish    : $$(jellyfish --version 2>&1 | head -1)"; \
			echo "✓ Dependencias OK"'; \
	fi

clean: ## Elimina contenedores, volúmenes y caché del proyecto
	docker compose --profile cpu --profile gpu down -v --rmi local
	@echo "✓ Limpieza completada"

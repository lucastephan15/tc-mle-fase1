# Comandos do projeto — a mesma sequência que o CI roda.
#
# Existe para responder a uma pergunta específica da avaliação: "alguém clona o
# repositório e reproduz seu melhor modelo com um comando?". Se a resposta mora
# na cabeça de quem escreveu, a reprodutibilidade é declarativa, não real.

.PHONY: help setup lint test ci gate eda baseline comparacao finalistas limpar

PY := python

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Cria o venv e instala as versões TRAVADAS
	$(PY) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt

lint:  ## Checagem estática (ruff)
	ruff check .

test:  ## Suíte de testes
	pytest

gate:  ## Treina o modelo de referência e falha se ficar abaixo do piso
	$(PY) -m src.gate

ci: lint test gate  ## Tudo que o CI roda, na mesma ordem — rodar ANTES de dar push

# --- Etapas do pipeline, na ordem em que foram executadas -------------------

eda:  ## Etapa 1 — análise exploratória
	$(PY) notebooks/01_eda.py

baseline:  ## Etapa 3 — baseline (LogReg) com registro no MLflow
	$(PY) -m src.train --modelo logreg

comparacao:  ## Etapa 6 — 11 candidatos em CV repetida no treino
	$(PY) -m src.comparacao

finalistas:  ## Etapa 6 — finalistas na validação: limiar, Brier e importâncias
	$(PY) -m src.finalistas

limpar:  ## Remove caches e saídas derivadas (data/raw NUNCA é tocado)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/__pycache__ tests/__pycache__
	rm -f data/processed/*.csv

# Comandos do projeto — a mesma sequência que o CI roda.
#
# Existe para responder a uma pergunta específica da avaliação: "alguém clona o
# repositório e reproduz seu melhor modelo com um comando?". Se a resposta mora
# na cabeça de quem escreveu, a reprodutibilidade é declarativa, não real.

.PHONY: help setup exige-venv lint test ci gate eda baseline comparacao finalistas limpar

# Os alvos usam os binários DO VENV, não os do shell. Motivo medido em 11/08/2026:
# `make ci` fora do venv ativado pegava o ruff GLOBAL do sistema (0.6.4) em vez do
# travado em requirements.txt (0.16.2) e reportava 6 erros PD901 que o CI não vê —
# um falso positivo que quase barrou um push legítimo. Ferramenta de validação cujo
# resultado depende de o shell estar "certo" não valida nada: ela vira a segunda
# verdade que o pyproject.toml existe para impedir.
VENV   := .venv
PY     := $(VENV)/bin/python
RUFF   := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest

# Alvos que exigem o ambiente instalado dependem disto: falha com instrução, não
# com "command not found".
exige-venv:
	@test -x $(PY) || { echo "venv ausente ou incompleto. Rode: make setup"; exit 1; }

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Cria o venv e instala as versões TRAVADAS
	python3 -m venv $(VENV)          # o único alvo que usa o Python do SISTEMA
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

lint: exige-venv  ## Checagem estática (ruff)
	$(RUFF) check .

test: exige-venv  ## Suíte de testes
	$(PYTEST)

gate: exige-venv  ## Treina o modelo de referência e falha se ficar abaixo do piso
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

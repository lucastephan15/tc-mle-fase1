# Comandos do projeto — a mesma sequência que o CI roda.
#
# Existe para responder a uma pergunta específica da avaliação: "alguém clona o
# repositório e reproduz seu melhor modelo com um comando?". Se a resposta mora
# na cabeça de quem escreveu, a reprodutibilidade é declarativa, não real.

.PHONY: help setup exige-venv lint test ci gate promover artefato api \
        monitorar simular-drift \
        docker-build docker-run docker-teste docker-limpar \
        eda baseline comparacao finalistas tuning mlp limpar

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

promover: exige-venv  ## Etapa 9c — grava models/campeao.joblib (só se passar no gate)
	$(PY) -m src.promover

artefato: exige-venv  ## Mostra o que está promovido hoje (versão, sha256, features)
	$(PY) -m src.artefato

auditar: exige-venv  ## Etapa 10.5 — fairness desagregado por grupo sensível
	# Mede no limiar de OPERAÇÃO (0,29), não no 0,5: a fila real é cortada ali,
	# e auditar no limiar errado descreve um modelo que o projeto não usa.
	# NÃO retorna 1 quando estoura o limite: a decisão registrada (§5g) é
	# aceitar e declarar. Quem barra mudança é tests/test_fairness.py.
	$(PY) -m src.fairness

api: exige-venv  ## Sobe a API local em http://localhost:8000 (docs em /docs)
	# Sem --reload: é servidor de desenvolvimento e recarregador de código, não
	# um modo "verbose". Em produção quem define o nº de workers é a RAM
	# (193,6 MB por worker, 93% import), não a vazão — a 1,7 ms por predição a
	# vazão sobra desde o primeiro.
	$(VENV)/bin/uvicorn src.api.app:criar_app --factory --host 127.0.0.1 --port 8000 --no-access-log

# --- Monitoramento (Etapa 10) -----------------------------------------------

LOG ?= logs/inferencia.jsonl

monitorar: exige-venv ## Etapa 10 — lê um .jsonl de inferência e mostra o painel
	# A API loga em STDOUT (filesystem de container é efêmero), então quem quer
	# arquivo redireciona: `make api > $(LOG)`. O caminho é sobrescrevível com
	# LOG=... — inclusive para os arquivos que o simulate_drift deixa em logs/.
	$(PY) -m src.monitoring $(LOG)

simular-drift: exige-venv ## Etapa 10c-bis — fabrica drift, prova que o detector dispara
	# Sobe a própria API num subprocesso, manda uma janela normal e uma
	# deslocada, e roda a detecção sobre os logs das duas. ~20 s.
	#
	# 🚨 Roda com TC_LOG_FEATURES=1 (o script liga sozinho), que produção mantém
	# desligado: data drift precisa das 13 colunas de entrada, e 4 delas são
	# demográficas. Aceitável aqui (máquina local, logs/ está no .gitignore) e
	# não no PaaS, onde o stdout é coletado por um terceiro.
	PYTHONPATH=. $(PY) scripts/simulate_drift.py

# --- Container (Etapa 9f) ---------------------------------------------------

# A tag da imagem acompanha a VERSÃO DO MODELO servido, não a do código: é o
# artefato dentro dela que define o que ela faz.
IMAGEM   ?= tc-churn
TAG      ?= 1.0.0
PLATAFORMA ?= linux/amd64
PORTA    ?= 8010

docker-build: ## Etapa 9f — constrói a imagem de serviço (amd64 por padrão)
	# 🚨 `--platform` explícito. Um contêiner Linux no macOS roda numa VM arm64,
	# então o build aqui produz arm64 por padrão — e o runner do Actions e a
	# maioria das clouds são x86_64: `exec format error`, e o erro aparece no
	# DEPLOY, não no build local que passou. Sobrescreva com PLATAFORMA=linux/arm64
	# para um build nativo (mais rápido) quando o alvo for só a máquina local.
	docker buildx build --platform $(PLATAFORMA) -t $(IMAGEM):$(TAG) --load .

docker-run: ## Sobe a imagem em http://localhost:$(PORTA) (docs em /docs)
	docker run --rm --name $(IMAGEM) --platform $(PLATAFORMA) \
		-p $(PORTA):8000 $(IMAGEM):$(TAG)

docker-teste: exige-venv ## Etapa 9e — sobe o container e testa contra ele
	# A metade que a suíte do pytest NÃO cobre: ela exercita o pipeline em
	# memória, e o que a API serve é o objeto SERIALIZADO dentro de uma imagem.
	# Testar o que você treina não é testar o que você serve.
	@docker rm -f $(IMAGEM)-teste >/dev/null 2>&1 || true
	docker run -d --name $(IMAGEM)-teste --platform $(PLATAFORMA) \
		-p $(PORTA):8000 $(IMAGEM):$(TAG)
	@echo "esperando o healthcheck (PRONTIDÃO, não vitalidade)..."
	@for i in $$(seq 1 60); do \
		s=$$(docker inspect --format '{{.State.Health.Status}}' $(IMAGEM)-teste); \
		[ "$$s" = "healthy" ] && break; \
		[ "$$s" = "unhealthy" ] && { docker logs $(IMAGEM)-teste; exit 1; }; \
		sleep 1; \
	done; echo "healthy em $${i}s"
	# `PYTHONPATH=.` porque `sys.path[0]` de um script é a pasta DELE (`scripts/`),
	# não a raiz do repo — o `import src` falharia. É o item 17 numa terceira
	# porta, e a correção definitiva é a 9g (tornar o projeto instalável), não
	# mais uma variável de ambiente por chamador.
	@TC_API_BASE=http://localhost:$(PORTA) PYTHONPATH=. $(PY) scripts/integracao_container.py; \
		codigo=$$?; docker rm -f $(IMAGEM)-teste >/dev/null; exit $$codigo

docker-limpar: ## Remove containers e imagens do projeto
	-docker rm -f $(IMAGEM) $(IMAGEM)-teste 2>/dev/null
	-docker rmi $(IMAGEM):$(TAG) 2>/dev/null

# --- Etapas do pipeline, na ordem em que foram executadas -------------------

eda:  ## Etapa 1 — análise exploratória
	$(PY) notebooks/01_eda.py

baseline:  ## Etapa 3 — baseline (LogReg) com registro no MLflow
	$(PY) -m src.train --modelo logreg

comparacao:  ## Etapa 6 — 11 candidatos em CV repetida no treino
	$(PY) -m src.comparacao

finalistas:  ## Etapa 6 — finalistas na validação: limiar, Brier e importâncias
	$(PY) -m src.finalistas

tuning:  ## Etapa 7 — grid (LogReg) + random search (HGB) com regra 1-SE
	$(PY) -m src.tuning

mlp:  ## Etapa 8 — MLP em PyTorch (grade + 5 seeds + validação)
	$(PY) -m src.mlp

limpar:  ## Remove caches e saídas derivadas (data/raw NUNCA é tocado)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/__pycache__ tests/__pycache__
	rm -f data/processed/*.csv

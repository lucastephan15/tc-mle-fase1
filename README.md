# Tech Challenge — Fase 1 · Predição de Churn (Telecom)

[![CI](https://github.com/lucastephan15/tc-mle-fase1/actions/workflows/ci.yml/badge.svg)](https://github.com/lucastephan15/tc-mle-fase1/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![ruff](https://img.shields.io/badge/lint-ruff-261230)](pyproject.toml)

Pós-graduação em **Machine Learning Engineering** — FIAP + Alura PosTech
Entrega: **01/09/2026** · Modalidade: individual

---

## O problema em uma frase

Priorizar clientes de telecom com maior risco de cancelamento, **ordenando-os numa fila de
trabalho** para a equipe de Retenção agir antes da decisão final do cliente.

> ⚠️ **O artefato é um ranqueador, não um classificador.** A pergunta do negócio não é
> *"esse cliente vai cancelar: sim ou não?"* — é ***"quem eu ligo primeiro?"***. Essa distinção
> determina a métrica (PR-AUC, independente de limiar), o formato do relatório (curva de ganho
> acumulado) e o fato de o limiar de corte ser um **parâmetro de negócio**, alterável sem
> retreinar nada.

| Item | Definição |
|---|---|
| **Alvo** | `Churn = Yes` — cliente cancelou o serviço |
| **Janela de predição** | 30 dias |
| **Ação disparada** | fila priorizada de tarefas no CRM → contato humano (**apoio à decisão, nunca ação automática**) |
| **Métrica primária** | **PR-AUC** (average precision) |
| **Métricas de negócio** | recall@10% e recall@20% da base pontuada |
| **Assimetria de custo** | falso negativo ≈ **R$ 194** × falso positivo ≈ **R$ 62** → **≈ 3:1** |

📄 **A documentação da entrega é [`docs/RELATORIO.md`](docs/RELATORIO.md)** — a narrativa
completa, do enquadramento à operação, com o número que sustenta cada decisão. Este README é
o manual de operação; o relatório é a leitura.

A conta que sustenta a razão 3:1, as premissas e os planos B estão em
**[`docs/decision-log.md`](docs/decision-log.md)** — o **registro de decisões**, matéria-prima
de tudo isso, preenchido **durante** a execução, nunca depois.

| documento | o que é |
|---|---|
| [`docs/RELATORIO.md`](docs/RELATORIO.md) | 📄 **a entrega** — leitura principal |
| [`docs/ML_CANVAS.md`](docs/ML_CANVAS.md) | o enquadramento de negócio: stakeholders, economia do erro, métricas e pressupostos |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | a EDA executada, com as figuras e o que cada achado decidiu |
| [`MODEL_CARD.md`](MODEL_CARD.md) | uso pretendido, usos **proibidos**, fairness, LGPD |
| [`docs/decision-log.md`](docs/decision-log.md) | o dossiê do processo, com o que não deu certo |
| [`docs/resultado-teste-final.json`](docs/resultado-teste-final.json) | o registro da leitura única do teste |

---

## Estrutura

```
src/            código modularizado — o que o CI importa e testa
  preprocess.py   limpeza + feature engineering
  train.py        treino, executável e parametrizável
  evaluate.py     métricas + auditoria de fairness
  artefato.py     empacotar/carregar o modelo servido — e VERIFICAR que é ele
  promover.py     grava models/campeao.joblib, só se passar no gate
  api/            FastAPI — schema (contrato) · servico (pontuar) · app (HTTP)
data/raw/       ⛔ IMUTÁVEL — nenhum script escreve aqui
data/processed/ tudo que é derivado (não versionado: reprodutível)
models/         artefatos serializados (o Registry é a fonte de verdade)
notebooks/      exploração — e SÓ exploração
  01_eda.py       a FONTE, em formato percent (`# %%`) — é o que entra em diff
  01_eda.ipynb    o derivado EXECUTADO (`make notebook`) — é o que se lê
tests/          unitários + integração
scripts/        integracao_container.py — a Etapa 9e, contra a imagem de pé
docs/           documentação da entrega — e o rastro de como ela foi decidida
  RELATORIO.md    o relatório: resultado, comparação campeão × baseline, teste tocado 1x
  ML_CANVAS.md    enquadramento do problema — stakeholders, métricas, premissas
  decision-log.md POR QUE cada decisão foi tomada; o git diz o que mudou, este diz por quê
  revisita.md     backlog DECLARADO — melhoria vista depois de uma etapa fechada não é
                  refeita na hora, é anotada aqui com origem, etapa afetada e resultado
                  quando executada (114 itens, 31 executados). É o mecanismo contra a
                  refatoração perpétua: dívida com dono e endereço não é dívida esquecida
Dockerfile      a receita; a IMAGEM é que é o artefato reprodutível
.dockerignore   allowlist — o Docker NÃO lê o .gitignore
.github/workflows/  CI/CD
```

**Duas regras que sustentam a estrutura:**
1. **`data/raw` é read-only.** Sobrescrever o bruto destrói a reprodutibilidade de todo modelo
   anterior — não se pode reverter só o código, é preciso poder reverter os dados.
2. **Notebook explora; `src/` produz.** Lógica que só existe numa célula não é testável, nem
   importável pelo CI, nem revisável — nenhuma linha de `notebooks/` é importada pelo pipeline.
   O `01_eda.ipynb` é entregue **como leitura**, e sua fonte versionada é o `.py` em formato
   percent: JSON de notebook é uma linha só, inmergeável, e muda de conteúdo a cada abertura.

---

## Stack

PyTorch (MLP) · scikit-learn · MLflow (Tracking + Model Registry) · FastAPI · Fairlearn · SHAP ·
GitHub Actions · Docker

---

## Reprodutibilidade

```bash
make setup      # venv + versões travadas
make ci         # lint + 140 testes + gate de promoção — o mesmo que o CI roda
make promover   # grava models/campeao.joblib — só se passar no gate
make artefato   # mostra o que está promovido (versão, sha256, features, limiar)
make reportar   # Etapa 11 — a leitura ÚNICA do conjunto de teste (+ figuras)
make notebook   # regenera notebooks/01_eda.ipynb executado, a partir do .py
make api        # sobe a API em http://localhost:8000 (docs em /docs)
make help       # todos os alvos
```

### A API

**No ar:** https://tc-churn-api.onrender.com — a raiz leva à **documentação interativa**
(`/docs`), gerada da OpenAPI. Rotas: `/health` · `/predict` · `/v1/predict` · `/v1/predict-batch`.
`/predict` e `/v1/predict` são **a mesma rota** (mesmo handler) em dois caminhos: o `/v1` é o
canônico e é a estratégia de release declarada; o caminho sem versão existe porque é por ele que
o integrador chama.

⚠️ Plano gratuito: o serviço **dorme após 15 min sem tráfego** e a primeira requisição depois
disso leva cerca de um minuto. Acorde-o antes de demonstrar.

Localmente:

```bash
make promover && make api

curl localhost:8000/health
curl -X POST localhost:8000/v1/predict -H 'Content-Type: application/json' -d '{
  "Total Charges": 108.15, "Tenure Months": 2.0, "Monthly Charges": 53.85,
  "Gender": "Male", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
  "Multiple Lines": "No", "Internet Service": "DSL", "Online Security": "Yes",
  "Tech Support": "No", "Contract": "Month-to-month", "Paperless Billing": "Yes"}'
# -> {"request_id":"…","versao_modelo":"1.0.0",
#     "predicoes":[{"probabilidade":0.3700,"decisao":true,"limiar":0.29}]}
```

`POST /v1/predict-batch` (`{"clientes": [...]}`, até 5.000) é o endpoint **principal**: 1.409
clientes custam 2,9 ms em lote contra 2.353 ms em chamadas unitárias. A resposta traz
**probabilidade**, não classe — o limiar de operação é parâmetro de negócio, vem do artefato e
acompanha cada predição.

O contrato de entrada é **derivado do artefato** (13 features, 25 valores categóricos, 3 faixas):
categoria inexistente, `-999` e `1e9` recebem **422**, não 200 com predição corrompida.

⚠️ **Limitações declaradas:** não há autenticação (API interna), e `/docs` publica o contrato —
que é a descrição do modelo, não dado pessoal.

⚠️ **`models/` não é versionado** (o `.gitignore` diz por quê: o registry é a fonte de verdade,
não o repositório) — **com uma exceção nomeada**: `models/campeao.joblib`, o artefato promovido,
8.306 bytes, porque a plataforma de deploy constrói a imagem a partir do clone. `make promover`
o reconstrói a qualquer momento, e ele só é gravado se o gate aprovar nos dois eixos.

Versões pinadas + seeds fixados (`random_state`, `np.random.seed`, `torch.manual_seed`).
Prova real de reprodutibilidade: clonar numa máquina limpa e obter **o número exato**.

### O container

```bash
make docker-build    # buildx --platform linux/amd64, base fixada por digest
make docker-teste    # sobe a imagem e roda a Etapa 9e contra ela
make docker-run      # sobe em http://localhost:8010
```

🚨 **`--platform linux/amd64` não é detalhe.** Um contêiner Linux no macOS roda numa VM **arm64**,
então o build no Apple Silicon produz imagem arm64 — e o runner do Actions e a maioria das clouds
são `x86_64`: `exec format error`, que aparece **no deploy**, não no build local que passou.

**A imagem de serviço não é a de experimentação.** Medido: venv completo **1.400 MB** (118 dists)
× imagem **510 MB** (30 dists) — `torch` sozinho são 1.057 MB e o campeão é uma **regressão
logística que não o importa**. O modelo servido tem **8.306 bytes**; `scipy`, `pandas`, `sklearn`
e `numpy` são ~70% da imagem. *O que não cabe numa função serverless não é o modelo, é o que ele
precisa para existir.*

**O `.dockerignore` é uma allowlist, e veio antes do Dockerfile.** O Docker **não lê o
`.gitignore`**: um `COPY . .` levaria `data/raw` (7.043 clientes reais), `mlruns/`, `.git` e o
`.venv` (1,5 GB de binários arm64 de macOS) para dentro da imagem — medido, **2,03 GB de contexto
contra 2,61 kB**. E camada Docker é imutável: apagar depois não devolve o dado.

O `HEALTHCHECK` verifica **prontidão**, não vitalidade — exige `status == "pronto"`, e foi
verificado reprovando: um servidor que responde **200 com `status: degradado`** falha a probe.
Sem artefato, o container **morre no boot** com mensagem acionável (exit 1), em vez de subir e
responder 200 com o modelo de outra pessoa.

⚠️ **`models/campeao.joblib` é a única exceção nomeada** ao `models/*` do `.gitignore` — 8.306
bytes, porque a plataforma de deploy constrói a imagem a partir do clone do Git. A regra continua
valendo para todo o resto; exceção declarada com motivo é decisão, exceção silenciosa é a regra
apodrecendo.

### O deploy

O serviço roda a partir do `render.yaml` versionado — **o deploy é código, não memória de quem
clicou num painel**. A plataforma constrói a imagem a partir do clone do Git, e é por isso que o
artefato promovido precisou ser versionado.

🎯 **`autoDeployTrigger: checksPass` fecha o CI/CD:** nenhum commit chega à nuvem sem o
`make ci` ter passado no GitHub Actions. O default da plataforma publicaria a cada push —
entrega contínua **sem** integração contínua.

🚨 **Um defeito que só a nuvem revelou, e vale a leitura:** o serviço passou a responder
`x-render-routing: no-server` em **48% das requisições**, com a aplicação respondendo 200 a tudo
que lhe chegava e zero restarts. Causa: a porta estava declarada **em dois lugares com valores
diferentes** — `EXPOSE 8000` no Dockerfile, processo escutando em `${PORT:-8000}` = 10000 (o
default da plataforma), e a plataforma *detectando* qual era. Declarar `PORT: 8000` alinhou os
três e levou o roteamento a **120/120**. *O `EXPOSE` é documentação, e documentação que discorda
do processo é uma afirmação que alguém vai ler — aqui, uma máquina.*

**Verificado em produção com o mesmo script da Etapa 9e:** PR-AUC **0,6646020519** local ×
**0,6646020519** na nuvem — idêntico nos 10 dígitos, com 0 decisões trocadas no limiar.

### O que o CI faz

| Job | Faz | Falha quando |
|---|---|---|
| **QA** | `ruff check` + `pytest` (140 testes) | lint sujo ou qualquer teste vermelho |
| **Gate de promoção** | treina o modelo de referência e mede na **validação**, em **dois eixos** | PR-AUC < 0,66 **ou** Brier > 0,14 |

Três decisões que valem a leitura, todas em `.github/workflows/ci.yml` e no decision log:

- **O gate mede na validação, não no teste.** Decidir promoção olhando o teste a cada push o
  converte em validação depois de alguns commits — ele deixa de estimar generalização, e o
  sintoma é traiçoeiro porque o gap treino-teste continua bonito. É divergência deliberada do
  enunciado da disciplina.
- **O piso é absoluto (0,66), não relativo.** Um gate do tipo *"≥ 80% do baseline"* aceitaria
  0,53 — pior que modelos que já foram rejeitados.
- **São dois eixos, não um: desempenho E calibração.** Um gate só de PR-AUC aprova o modelo que
  ordena igual e calibra pior — e como a fila é ordenada por `P(churn) × CLTV`, a probabilidade é
  multiplicada por reais. Os dois limites já foram verificados **reprovando**, porque gate que
  nunca falhou é decoração.
- **Continuous Delivery, não Deployment.** O último passo para produção é humano, porque a
  predição dispara ação comercial com custo real. O job de registro no Model Registry
  ainda **não** existe, e o porquê está comentado no próprio workflow: sem backend persistente
  do MLflow, ele declararia sucesso sem ter feito nada.

---

### O monitoramento

A API emite **uma linha JSON por requisição em stdout** (não em arquivo: o filesystem do container
é efêmero, e num PaaS o que a plataforma coleta é o stream). Quem quer arquivo redireciona:

```bash
make api > logs/inferencia.jsonl        # a API, com o log indo para disco
make monitorar                          # o painel: latência em percentis, erros, drift
make simular-drift                      # ~20 s — fabrica drift e mostra o alarme nascer
```

**O que vai na linha, e o que não vai.** `scores` **sempre** (probabilidade sem atributo ao lado
não identifica ninguém, e dá prediction drift de graça); as 13 features **só** com
`TC_LOG_FEATURES=1`, porque quatro delas são demográficas e o stdout, num PaaS, é coletado por um
terceiro. O default é a direção segura, e a decisão foi tomada **antes** da primeira linha ser
emitida: log emitido não volta.

**O baseline mora dentro do artefato** (média/desvio/quantis e bordas de PSI das numéricas,
frequências das categóricas, distribuição dos scores). É propriedade daquele modelo, não do
repositório — e como guarda proporções por bin em vez de dados, roda no container **sem o
dataset**, que é justamente o que não pode estar lá.

`make simular-drift` é o teste do sistema de vigilância: manda à API uma janela normal e uma
deslocada de propósito (base envelhecendo, reajuste de preço, plano novo no catálogo) e mostra o
antes/depois. O resultado mais instrutivo veio do terceiro cenário — **categoria inédita devolve
422 e nunca chega ao modelo**, porque o schema é derivado do artefato. Nesta API, categoria nova é
evento de **serviço**, não de drift. Detalhes e limitações em `docs/decision-log.md` §5f.

---

## Status

| Etapa | Estado |
|---|---|
| 0 · Enquadramento do problema | ✅ concluída — `docs/decision-log.md` §0 |
| 1 · Data Understanding (EDA) | ✅ concluída — `docs/decision-log.md` §1 · `notebooks/01_eda.py` |
| 2 · Data Preparation | ✅ concluída — §2 · split 60/20/20, teste intocado |
| 3 · Baseline / MVP | ✅ concluída — §3 · **LogReg, PR-AUC 0,6623** contra piso de 0,2654 |
| 4 · Feature Engineering | ✅ concluída — §3 · 4 features medidas por ablação e **todas descartadas** |
| 5 · Seleção de features | ✅ concluída — §4 · **19 → 13 features**, por custo operacional |
| 6 · Comparação de algoritmos | ✅ concluída — §5b · **empate técnico** entre LogReg, HGB e RF (0,07 dp) |
| 7 · Tuning | ✅ concluída — §5c · **ganho zero**: o default da LogReg já era o pico da grade |
| 8 · MLP em PyTorch | ✅ concluída — §5d · **a rede não superou** (0,6615 × 0,6646), e a regra 1-SE elegeu profundidade **zero** |
| 8-bis · MLPClassifier (sklearn) | ✅ **concluída** — §5d-bis · a rede **que o enunciado nomeia**, sob protocolo idêntico. A 1-SE elegeu **profundidade zero outra vez**, e o `early_stopping` que pontua por acurácia custou **−0,0304 de PR-AUC e +R$ 1.838/ciclo**, medido |
| 9 · Pipeline serializado + API | ✅ **concluída (9c → 9f-quater)** — §5e · artefato promovido com identidade verificada na carga, API FastAPI de pé (`/health` · `/predict` · `/v1/predict` · `/v1/predict-batch`) e **imagem `linux/amd64` servindo o mesmo modelo** (PR-AUC idêntico nos 10 dígitos entre macOS e Linux, 0 decisões trocadas). **No ar em https://tc-churn-api.onrender.com**, com o deploy versionado em `render.yaml` e travado atrás do CI |
| 9.5 · CI/CD | ✅ **concluída** — QA + gate de dois eixos + teste de caracterização rodando a cada push, com deploy travado atrás dos checks. O job de registro no Model Registry foi **deliberadamente omitido**, com os dois bloqueios escritos dentro do próprio YAML: `mlruns/` morre com o runner (exigiria backend persistente) e o artefato só passou a existir na Etapa 9. Escrevê-lo antes disso produziria um job que declara sucesso sem ter feito nada |
| 10 · Monitoramento | ✅ **concluída** — §5f · log estruturado (10a), baseline de drift **dentro do artefato** (10a-2), **drift fabricado com detector verificado** (10c-bis), tabela das 4 famílias com limiar E ação (10b), política de retreino (10d) e rollback em duas camadas (10e) |
| 10.5 · Governança e fairness | ✅ **concluída** — §5g · `MODEL_CARD.md` com pré-registro **commitado antes** da auditoria, e ela **achou**: 58,89 pp de disparidade de recall em `Dependents`, aceita e declarada com o preço das três saídas medido |
| 11 · Documentação | ✅ **concluída** — [`docs/RELATORIO.md`](docs/RELATORIO.md) · §6b · **teste tocado uma única vez** (PR-AUC 0,6496, IC95 [0,5960; 0,7016]) · curva de ganho · seções 7 e 8 do decision log |

---

## Licença

Código, notebooks e documentação deste repositório estão sob a **Licença MIT** — ver
[`LICENSE`](LICENSE).

O que **não** está coberto por ela: o dataset em `data/raw/`, que é o *Telco Customer
Churn* da **IBM**, versionado aqui porque o `sha256` dele é o que amarra cada modelo ao
snapshot exato do dado. A separação está escrita em [`NOTICE.md`](NOTICE.md), junto com
as licenças de terceiros das dependências.

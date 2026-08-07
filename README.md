# Tech Challenge — Fase 1 · Predição de Churn (Telecom)

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

A conta que sustenta a razão 3:1, as premissas e os planos B estão em
**[`docs/decision-log.md`](docs/decision-log.md)** — que é a matéria-prima desta documentação e
é preenchido **durante** a execução, nunca depois.

---

## Estrutura

```
src/            código modularizado — o que o CI importa e testa
  preprocess.py   limpeza + feature engineering
  train.py        treino, executável e parametrizável
  evaluate.py     métricas + auditoria de fairness
  api/            FastAPI (/predict, /health)
data/raw/       ⛔ IMUTÁVEL — nenhum script escreve aqui
data/processed/ tudo que é derivado (não versionado: reprodutível)
models/         artefatos serializados (o Registry é a fonte de verdade)
notebooks/      exploração — e SÓ exploração
tests/          unitários + integração
docs/           decision log, model card
.github/workflows/  CI/CD
```

**Duas regras que sustentam a estrutura:**
1. **`data/raw` é read-only.** Sobrescrever o bruto destrói a reprodutibilidade de todo modelo
   anterior — não se pode reverter só o código, é preciso poder reverter os dados.
2. **Notebook não é entregável.** Ele explora; `src/` produz. Lógica que só existe numa célula
   não é testável, nem importável pelo CI, nem revisável.

---

## Stack

PyTorch (MLP) · scikit-learn · MLflow (Tracking + Model Registry) · FastAPI · Fairlearn · SHAP ·
GitHub Actions · Docker

---

## Reprodutibilidade

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # versões travadas
```

Versões pinadas + seeds fixados (`random_state`, `np.random.seed`, `torch.manual_seed`).
Prova real de reprodutibilidade: clonar numa máquina limpa e obter **o número exato**.

---

## Status

| Etapa | Estado |
|---|---|
| 0 · Enquadramento do problema | ✅ concluída — `docs/decision-log.md` §0 |
| 1 · Data Understanding (EDA) | ⬜ |
| 2 · Data Preparation | ⬜ |
| 3 · Baseline / MVP | ⬜ |
| 4-8 · FE → seleção → comparação → tuning → MLP | ⬜ |
| 9 · Pipeline serializado + API | ⬜ |
| 9.5 · CI/CD | ⬜ |
| 10 · Monitoramento | ⬜ |
| 10.5 · Governança e fairness | ⬜ |
| 11 · Documentação | ⬜ |

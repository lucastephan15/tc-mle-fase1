# NOTICE — direitos autorais e materiais de terceiros

Este repositório reúne conteúdo com dois regimes distintos de direitos. Este arquivo
separa os dois, para que a Licença MIT não seja lida como se cobrisse o que não é meu.

---

## 1. Conteúdo autoral deste projeto — Licença MIT

Todo o código-fonte (`src/`, `tests/`, `scripts/`), os notebooks, a documentação
(`README.md`, `MODEL_CARD.md`, `docs/`), as configurações de build, container e CI, e os
artefatos de modelo aqui versionados são de minha autoria e estão licenciados sob a
**Licença MIT** — ver [`LICENSE`](./LICENSE).

---

## 2. Dataset — IBM Telco Customer Churn

O arquivo [`data/raw/Telco_customer_churn.xlsx`](./data/raw/Telco_customer_churn.xlsx)
(7.043 clientes, 33 colunas, `sha256 1bcbc0cc…`) é a variante estendida do **Telco
Customer Churn**, dado de amostra publicado pela **IBM** e amplamente redistribuído para
fins educacionais.

- **Não está coberto pela Licença MIT** deste repositório. Os direitos sobre o dado
  pertencem à IBM e/ou seus licenciantes.
- Está versionado aqui por uma razão técnica, não por conveniência: `data/raw` é a
  entrada imutável do pipeline, e o `sha256` do arquivo é recalculado a cada carregamento
  (`src/data.py`) para amarrar cada modelo treinado ao snapshot exato do dado. Sem o
  arquivo, nenhum resultado deste repositório é reproduzível.
- Nenhum uso comercial ou redistribuição fora do contexto acadêmico é autorizado por
  este NOTICE.
- A coluna `Churn Score` do dataset é a **saída de um modelo proprietário da IBM**
  embutida no dado. Ela foi deliberadamente excluída do treino e usada apenas como
  benchmark de referência — o motivo está em [`MODEL_CARD.md`](./MODEL_CARD.md).

---

## 3. Enunciado do Tech Challenge

O enunciado oficial do desafio é de autoria da **FIAP / PosTech** e **não está
versionado neste repositório** — apenas os requisitos que ele impõe são citados na
documentação, para tornar rastreável o que foi atendido.

---

## 4. Dependências de terceiros

As bibliotecas declaradas em `requirements.in` / `requirements.txt` e
`requirements-serve.in` / `requirements-serve.txt` mantêm cada uma a sua própria licença
de origem (scikit-learn, PyTorch, FastAPI, Fairlearn, SHAP, MLflow, entre outras). Nada
neste repositório altera ou substitui esses termos.

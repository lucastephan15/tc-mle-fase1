"""
Auditoria de fairness — Etapa 10.5.

    python -m src.fairness        (ou `make auditar`)

Mede as métricas do modelo **desagregadas por grupo sensível**, porque uma
métrica agregada boa é compatível com um grupo sendo mal atendido: um recall
global de 0,77 pode ser 0,81 num grupo e 0,22 em outro — e é exatamente o que
este repositório mede.

⛔ **Mede no limiar de OPERAÇÃO, não no 0,5 implícito.** A fila de retenção real
é cortada em `config`/artefato (0,29), e é nesse ponto que se decide quem recebe
campanha. Auditar em 0,5 produziria uma tabela tecnicamente correta sobre um
modelo que este projeto não usa.

⛔ **Mede na VALIDAÇÃO.** Mesmo motivo do gate: o teste segue intocado.

🔑 **A métrica que carrega o dano é o recall (equivalentemente, a FNR), não a
acurácia.** Em churn, o prejuízo mora no falso negativo: o cliente que ia
cancelar e não foi marcado não entra na fila, não recebe oferta e vai embora —
e isso nunca vira linha de planilha, porque ninguém registra a campanha que não
foi feita. `selection_rate` entra como **diagnóstico** (explica a disparidade),
nunca como critério: taxa de seleção maior num grupo de prevalência maior é o
modelo funcionando, não viés.

Os atributos sensíveis estão **dentro** das features de propósito (decisão da
Etapa 5). Removê-los não removeria o viés — o modelo os reconstrói pelos proxies
(`Contract`, `Tenure Months`) — removeria a capacidade de medi-lo. Medido nesta
etapa: sem `Dependents`, a disparidade cai de 58,89 pp para 13,20 pp (atenua, não
resolve) e a PR-AUC vai a 0,6427, **abaixo do piso do gate**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from src import config, data, evaluate, gate

# Atributos protegidos auditados. Ficam separados de `config.CAT` de propósito:
# esta lista responde "o que precisa ser auditado", não "o que o modelo usa".
SENSIVEIS = ["Gender", "Senior Citizen", "Partner", "Dependents"]

# Política declarada no PRÉ-REGISTRO (decision log §5g), escrita antes da
# primeira medição. Não é atingida por 3 dos 4 atributos, e a decisão registrada
# é aceitar com declaração — ver o Model Card. O número continua aqui porque uma
# política que some quando é violada não é política.
LIMITE_DISPARIDADE = 0.10

# Abaixo disto, a disparidade do grupo é reportada com aviso de incerteza: com
# poucos churners o recall do grupo tem intervalo de confiança largo demais para
# sustentar conclusão sozinho. `Dependents=Yes` tem 23 na validação.
MIN_CHURNERS_CONFIAVEL = 30


def _ic_binomial(k: int, n: int) -> tuple[float, float]:
    """IC95 de Clopper-Pearson. Existe para que grupo pequeno não vire conclusão.

    Sem ele, um recall de 0,2174 sobre 23 churners tem a mesma aparência de um
    sobre 2.300 — e só um dos dois sustenta uma decisão.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def auditar(
    y_true, y_score, sensiveis: pd.DataFrame, limiar: float
) -> dict[str, dict]:
    """Métricas por grupo para cada atributo sensível, no limiar de operação.

    Devolve um dicionário em vez de imprimir pelo mesmo motivo que
    `gate.aprovado()` devolve: para poder ser chamado por um teste. O relatório
    é uma função separada, e o Model Card cita **estes** números — não uma
    segunda contagem mantida igual pela memória de quem escreveu.
    """
    y = np.asarray(y_true)
    pred = (np.asarray(y_score) >= limiar).astype(int)
    saida: dict[str, dict] = {}

    for attr in sensiveis.columns:
        s = sensiveis[attr].to_numpy()
        grupos: dict[str, dict] = {}
        for g in sorted(pd.unique(s), key=str):
            m = s == g
            churners = int(y[m].sum())
            pegos = int(((pred == 1) & (y == 1))[m].sum())
            lo, hi = _ic_binomial(pegos, churners)
            grupos[str(g)] = {
                "n": int(m.sum()),
                "churners": churners,
                "recall": pegos / churners if churners else float("nan"),
                "fnr": 1 - (pegos / churners) if churners else float("nan"),
                "selection_rate": float(pred[m].mean()),
                "prevalencia": float(y[m].mean()),
                "ic95_recall": (lo, hi),
                "confiavel": churners >= MIN_CHURNERS_CONFIAVEL,
            }
        recalls = [v["recall"] for v in grupos.values()]
        saida[attr] = {
            "grupos": grupos,
            "disparidade_recall": float(max(recalls) - min(recalls)),
            "dentro_do_limite": bool(max(recalls) - min(recalls) <= LIMITE_DISPARIDADE),
            # Qual grupo é o prejudicado importa: uma disparidade a FAVOR do
            # grupo protegido é disparidade e é reportada, mas não é o mesmo
            # dano. Em `Senior Citizen` o recall maior é o do grupo idoso.
            "grupo_pior": min(grupos, key=lambda k: grupos[k]["recall"]),
        }
    return saida


def medir_do_campeao() -> tuple[dict[str, dict], float]:
    """Treina o campeão pelo caminho canônico e audita. Fonte única do número.

    Usa `gate.treinar_campeao` — a mesma função que o gate do CI e a promoção
    usam — para que a auditoria não caracterize um terceiro modelo construído
    com o mesmo código. É a lição do item 85, aplicada à fairness.
    """
    np.random.seed(config.SEED)
    d = data.dividir()
    pipe = gate.treinar_campeao(d)
    p = pipe.predict_proba(d.validacao.X)[:, 1]
    limiar = evaluate.curva_custo(d.validacao.y, p)["limiar_otimo"]
    return auditar(d.validacao.y, p, d.validacao.X[SENSIVEIS], limiar), limiar


def relatorio(aud: dict[str, dict], limiar: float) -> str:
    linhas = [
        f"Auditoria de fairness — validação, limiar de operação {limiar:.2f}",
        f"Política pré-registrada: disparidade de recall <= {LIMITE_DISPARIDADE:.0%}",
        "",
    ]
    for attr, r in aud.items():
        marca = "✅" if r["dentro_do_limite"] else "🚨"
        linhas.append(f"{marca} {attr}  —  disparidade {r['disparidade_recall'] * 100:.2f} pp")
        for g, v in r["grupos"].items():
            aviso = "" if v["confiavel"] else f"  ⚠️ só {v['churners']} churners"
            lo, hi = v["ic95_recall"]
            linhas.append(
                f"     {g:<6} n={v['n']:>5}  prev={v['prevalencia']:.4f}  "
                f"recall={v['recall']:.4f} [IC95 {lo:.3f}–{hi:.3f}]  "
                f"sel={v['selection_rate']:.4f}{aviso}"
            )
        linhas.append("")
    return "\n".join(linhas)


def main() -> int:
    aud, limiar = medir_do_campeao()
    print(relatorio(aud, limiar))
    fora = [a for a, r in aud.items() if not r["dentro_do_limite"]]
    if fora:
        # NÃO retorna 1. A decisão registrada na Etapa 10.5 é aceitar com
        # declaração, e um alvo que derruba o CI depois de a decisão ter sido
        # tomada seria teatro. Quem barra mudança é o teste de caracterização.
        print(f"⚠️  Fora do limite de {LIMITE_DISPARIDADE:.0%}: {', '.join(fora)}")
        print("   Decisão registrada (decision log §5g): aceitar e DECLARAR no Model Card.")
        print("   O CI barra a MUDANÇA destes números (tests/test_fairness.py), não o nível.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

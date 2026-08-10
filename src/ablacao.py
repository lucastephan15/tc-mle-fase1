"""
Estudo de ablação das features — Etapa 4.

    python -m src.ablacao

Responde à provocação do runbook: *"você vai medir o ganho dessa feature ou só
assumir que ajudou?"*.

DUAS DECISÕES METODOLÓGICAS, ambas deliberadas:

1. **Mede por validação cruzada estratificada DENTRO DO TREINO**, não na
   validação. Se cada feature candidata fosse julgada olhando a validação, cinco
   decisões depois a validação estaria gasta — exatamente o mesmo problema que
   nos fez manter o teste intocado, só que um andar acima. A validação continua
   reservada para a escolha final entre modelos.

2. **Ablação por remoção (leave-one-out), não por adição isolada.** Adicionar uma
   feature sozinha superestima sua contribuição quando ela é redundante com
   outra. Medir quanto se PERDE ao removê-la do conjunto completo responde a
   pergunta certa: "esta feature agrega algo que as outras já não dizem?".

O desvio-padrão entre folds é reportado junto porque sem ele não há como
distinguir ganho real de variância do sorteio.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config, data, features
from src.preprocess import construir_pipeline

TODAS = features.NOVAS_NUM + features.NOVAS_CAT
N_FOLDS = 5


def montar(nome: str):
    """O estimador da ablação e se ele precisa de escalonamento.

    Rodar a ablação com DOIS algoritmos não é comparar algoritmos (isso é a
    Etapa 6) — é testar o princípio de que feature engineering depende do
    algoritmo alvo. Uma contagem que é combinação linear das dummies já
    presentes é redundante por construção para um modelo linear, mas pode não
    ser para uma árvore, que precisaria de vários splits para reconstruí-la.
    """
    if nome == "logreg":
        return LogisticRegression(max_iter=1000, random_state=config.SEED), True
    if nome == "rf":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            random_state=config.SEED, n_jobs=-1,
        ), False
    raise ValueError(nome)


def cv_pr_auc(X, y, novas: list[str], modelo: str = "logreg") -> tuple[float, float]:
    """PR-AUC média e desvio-padrão em K folds estratificados do treino."""
    estimador, escalonar = montar(modelo)
    pipe = construir_pipeline(estimador, escalonar=escalonar, novas=novas)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
    # 'average_precision' é o nome do PR-AUC no sklearn — a métrica primária
    # decidida na Etapa 0. Deixar 'accuracy' aqui por inércia mediria outra coisa.
    notas = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    return float(notas.mean()), float(notas.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="logreg", choices=["logreg", "rf"])
    args = ap.parse_args()

    dados = data.dividir()
    X, y = dados.treino.X, dados.treino.y

    print(f"Ablação por CV estratificada ({N_FOLDS} folds) no TREINO — "
          f"métrica: PR-AUC — modelo: {args.modelo}\n")

    base_m, base_s = cv_pr_auc(X, y, [], args.modelo)
    todas_m, todas_s = cv_pr_auc(X, y, TODAS, args.modelo)

    print(f"{'conjunto':<34} {'PR-AUC':>8} {'±dp':>7} {'Δ vs base':>11}")
    print("-" * 63)
    print(f"{'só as 19 originais (base)':<34} {base_m:>8.4f} {base_s:>7.4f} {'—':>11}")
    print(f"{'+ as 4 novas':<34} {todas_m:>8.4f} {todas_s:>7.4f} "
          f"{todas_m - base_m:>+11.4f}")
    print()

    print("Contribuição marginal (remove UMA do conjunto completo):")
    print(f"{'sem qual feature':<34} {'PR-AUC':>8} {'±dp':>7} {'perda':>11}")
    print("-" * 63)
    marginais = []
    for f in TODAS:
        restantes = [x for x in TODAS if x != f]
        m, s = cv_pr_auc(X, y, restantes, args.modelo)
        perda = todas_m - m
        marginais.append((f, perda))
        print(f"{'sem ' + f:<34} {m:>8.4f} {s:>7.4f} {perda:>+11.4f}")

    print()
    print("Leitura: 'perda' positiva = a feature agrega algo que as outras não dizem.")
    print(f"Referência de ruído: o desvio entre folds do modelo completo é {todas_s:.4f}")
    print("— ganho menor que isso não é ganho, é sorteio.\n")

    uteis = [f for f, p in marginais if p > todas_s / 2]
    print(f"Acima de meio desvio: {uteis or 'nenhuma'}")


if __name__ == "__main__":
    np.random.seed(config.SEED)
    main()

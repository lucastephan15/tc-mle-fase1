"""
MLPClassifier do scikit-learn — Etapa 8-bis.

    python -m src.mlp_sklearn            # o experimento completo
    python -m src.mlp_sklearn --rapido   # 1 repetição da CV

POR QUE ESTE MÓDULO EXISTE, se a Etapa 8 já entregou uma rede treinada. O
enunciado da fase nomeia a implementação: *"treinar uma Rede Neural simples
utilizando o MLPClassifier do Scikit-Learn"*. A Etapa 8 usou PyTorch, e a razão
está escrita em `src/mlp.py`: o `early_stopping=True` do `MLPClassifier` pontua
a validação interna com `accuracy_score` — está no fonte e a classe **não expõe
`scoring`** —, o que é otimizar a métrica errada dentro do laço de treino, num
problema de prevalência 26,5% com custo de erro 3:1.

As duas coisas não são alternativas. O que este módulo faz é **medir** esse
argumento em vez de afirmá-lo, rodando o `MLPClassifier` sob protocolo idêntico
ao das Etapas 6, 7 e 8 — mesmas 13 features, mesmo pipeline (one-hot +
padronização), mesma CV estratificada repetida 5x3, mesma seed, `scoring=
"average_precision"`. Produz três coisas:

1. **a linha do `MLPClassifier` na tabela comparativa** — o requisito cumprido
   com o número ao lado dos demais candidatos, e não com uma justificativa;
2. **o custo medido do early stopping que pontua por acurácia** — a defesa da
   implementação em PyTorch deixa de ser leitura de código-fonte e vira número;
3. uma **quinta medição independente** de *"o sinal do Telco é essencialmente
   linear no logit"* — agora com a rede de catálogo, escrita por outra equipe,
   com outro otimizador, outra inicialização e outro critério de parada. As
   quatro anteriores: FE inútil (Etapa 4), gap 0,005 x 0,083 (Etapa 6), toco de
   2 folhas (Etapa 7), 1-SE elegendo profundidade zero (Etapa 8).

⚠️ **Não promove modelo e não toca o conjunto de teste.** Mede no treino (CV) e
na validação da Etapa 2, como todo o resto do repositório.
"""

from __future__ import annotations

import argparse
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.neural_network import MLPClassifier

from src import config, data, evaluate
from src.comparacao import EXPERIMENTO, N_FOLDS, N_REPS
from src.config import SEEDS
from src.preprocess import construir_pipeline
from src.tuning import envelopes, refit_1se

# `max_iter` alto de propósito: o default 200 do sklearn faz a rede parar por
# ORÇAMENTO e não por convergência, e uma comparação em que um candidato foi
# interrompido no meio do treino mede o corte, não o modelo. O preço é o
# ConvergenceWarning, que aqui é CONTADO em vez de engolido (§4 do relatório).
MAX_ITER = 800


def grade() -> dict:
    """A MESMA grade da Etapa 8, traduzida para os nomes do sklearn.

    `alpha` é a L2 do `MLPClassifier` — o mesmo papel do `weight_decay` do Adam
    na implementação em PyTorch, e o terceiro alvo da penalização que a M02-A02
    estabeleceu (coeficientes -> folhas -> pesos). Os três valores {0, 1e-3,
    1e-2} espelham a grade do PyTorch para que as duas tabelas se comparem
    linha a linha; **1e-4 entra por ser o default do sklearn**, isto é, a
    configuração que este experimento teria se ninguém tivesse escolhido nada.

    `()` é o CONTROLE, pelo mesmo motivo de lá: zero camadas ocultas é uma
    regressão logística, e o número dela tem de bater com o 0,6904 da Etapa 6.
    Se não bater, o que muda entre os candidatos não é só a profundidade — e
    nada abaixo significa coisa alguma.
    """
    return {
        "modelo__hidden_layer_sizes": [(), (8,), (16,), (32,), (16, 16)],
        "modelo__alpha": [0.0, 1e-4, 1e-3, 1e-2],
    }


def complexidade(p: dict) -> tuple:
    """Ordem de complexidade para a 1-SE — idêntica à `complexidade_mlp`."""
    hidden = p["modelo__hidden_layer_sizes"]
    return (len(hidden), sum(hidden), -p["modelo__alpha"])


def pipe(seed: int = config.SEED, early_stopping: bool = False, **kwargs):
    """O `MLPClassifier` no mesmo pipeline de todos os outros candidatos.

    `early_stopping=False` é o DEFAULT DESTE MÓDULO, ao contrário do reflexo
    comum — e a escolha é o objeto do experimento, não uma preferência: ligá-lo
    troca o critério de parada por `accuracy_score` sem oferecer alternativa.
    O §3 mede as duas configurações lado a lado.
    """
    return construir_pipeline(
        MLPClassifier(
            max_iter=MAX_ITER,
            early_stopping=early_stopping,
            random_state=seed,
            **kwargs,
        ),
        escalonar=True,
        encoding="onehot",
    )


def _cv_media(estimador, X, y, cv) -> tuple[float, float]:
    r = cross_validate(estimador, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    return float(r["test_score"].mean()), float(r["test_score"].std())


def fase_seeds(X, y, cv, params: dict, early_stopping: bool) -> pd.DataFrame:
    """As 5 seeds — a variância de inicialização, que a CV não enxerga.

    A CV repetida reamostra os DADOS; ela é cega ao fato de que um MLP muda de
    solução com os dados fixos. Comparar as duas dispersões é o que diz qual
    delas está falando (na Etapa 8 o desvio entre seeds foi 0,04x o desvio entre
    folds — a dimensão seed não precisava entrar no protocolo, e só se soube
    porque foi medida).
    """
    linhas = []
    for seed in SEEDS:
        media, dp = _cv_media(pipe(seed=seed, early_stopping=early_stopping, **params),
                              X, y, cv)
        linhas.append({"seed": seed, "pr_auc": media, "dp_folds": dp})
    return pd.DataFrame(linhas)


def fase_validacao(X, y, X_val, y_val, params: dict,
                   early_stopping: bool = False) -> pd.DataFrame:
    """As 5 seeds na validação — o número que vai para a tabela mestra.

    Treina no treino inteiro e mede uma vez na validação da Etapa 2. O teste
    permanece intocado: ele já foi lido uma única vez, na Etapa 11, sobre o
    artefato promovido — e reabri-lo para acomodar um candidato novo é
    exatamente o que a disciplina do toque único existe para impedir.
    """
    linhas, probabilidades = [], []
    for seed in SEEDS:
        p = pipe(seed=seed, early_stopping=early_stopping, **params).fit(X, y)
        est = p.named_steps["modelo"]
        prob = p.predict_proba(X_val)[:, 1]
        probabilidades.append(prob)
        custo = evaluate.curva_custo(y_val, prob)
        m = evaluate.avaliar(y_val, prob, limiar=custo["limiar_otimo"])
        linhas.append({
            "seed": seed,
            "pr_auc": m["pr_auc"], "brier": m["brier"],
            "limiar_otimo": custo["limiar_otimo"],
            "custo_brl": custo["custo_otimo_brl"],
            "recall_at_10": m["recall_at_10"], "recall_at_20": m["recall_at_20"],
            "n_iter": int(est.n_iter_),
            "parou_no_teto": bool(est.n_iter_ >= MAX_ITER),
            # ⚠️ `best_validation_score_` EXISTE mesmo sem early stopping — com
            # valor `None`, e não ausente. `getattr(..., default)` não cobre esse
            # caso (o atributo está lá), e `float(None)` levanta TypeError: é a
            # diferença entre "não tem" e "tem, vazio" cobrando um `or np.nan`.
            # Quando ele tem valor, o que guarda é ACURÁCIA — a evidência do §3.
            "score_de_parada": float(est.best_validation_score_ or np.nan),
        })
    df = pd.DataFrame(linhas)
    df.attrs["desvio_predicao"] = float(
        np.std(np.vstack(probabilidades), axis=0).mean()
    )
    return df


def main() -> None:  # noqa: PLR0915 — relatório linear; quebrar piora a leitura
    ap = argparse.ArgumentParser(
        description="Etapa 8-bis — a rede exigida pelo enunciado (MLPClassifier)",
    )
    ap.add_argument("--rapido", action="store_true", help="1 repetição da CV")
    args = ap.parse_args()

    np.random.seed(config.SEED)
    dados = data.dividir()
    X, y = dados.treino.X, dados.treino.y
    X_val, y_val = dados.validacao.X, dados.validacao.y

    reps = 1 if args.rapido else N_REPS
    cv = RepeatedStratifiedKFold(
        n_splits=N_FOLDS, n_repeats=reps, random_state=config.SEED,
    )
    n_dobras = cv.get_n_splits()

    print("Etapa 8-bis — MLPClassifier (scikit-learn) · CV repetida no TREINO "
          f"({n_dobras} dobras por configuração)")
    print(f"Protocolo idêntico ao das Etapas 6, 7 e 8 · n_treino={len(X)} · "
          "13 features · one-hot + padronização\n")

    # === 1. a grade ========================================================
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        busca = GridSearchCV(
            pipe(), grade(), scoring="average_precision", cv=cv, n_jobs=-1,
            return_train_score=True,
            refit=refit_1se(complexidade, n_dobras),
        )
        busca.fit(X, y)
    n_convergencia = sum(1 for a in avisos if issubclass(a.category, ConvergenceWarning))

    r = busca.cv_results_
    medias = np.asarray(r["mean_test_score"])
    dps = np.asarray(r["std_test_score"])
    i_pico, i_1se = int(np.argmax(medias)), int(busca.best_index_)
    env = envelopes(medias, dps, i_pico, n_dobras)

    print(f"=== GRADE — {len(r['params'])} configurações ===")
    print(f"{'hidden':<10} {'alpha':>8} {'PR-AUC':>8} {'±dp':>7} {'treino':>7} {'gap':>7}")
    print("-" * 52)
    for i in np.argsort(medias)[::-1]:
        p = r["params"][i]
        marca = " ←1-SE" if i == i_1se else (" ←pico" if i == i_pico else "")
        print(f"{p['modelo__hidden_layer_sizes']!s:<10} {p['modelo__alpha']:>8.4f} "
              f"{medias[i]:>8.4f} {dps[i]:>7.4f} {r['mean_train_score'][i]:>7.4f} "
              f"{r['mean_train_score'][i] - medias[i]:>7.4f}{marca}")

    params_1se = {k.replace("modelo__", ""): v for k, v in r["params"][i_1se].items()}
    print(f"\n  envelope 1-SE: dp/√{n_dobras} {env['se']:.4f} · "
          f"dp/√{N_FOLDS} (Nadeau-Bengio) {env['conservador']:.4f} · "
          f"dp cheio {env['desvio_cheio']:.4f}")
    print(f"  escolha 1-SE: {params_1se}")

    # === 2. o controle de profundidade zero ================================
    i_zero = next(i for i, p in enumerate(r["params"])
                  if p["modelo__hidden_layer_sizes"] == () and p["modelo__alpha"] == 0.0)
    pr_logreg, _ = _cv_media(
        construir_pipeline(
            LogisticRegression(max_iter=1000, random_state=config.SEED),
            escalonar=True, encoding="onehot",
        ), X, y, cv,
    )
    print("\n=== CONTROLE — o MLPClassifier de ZERO camadas ocultas é a LogReg? ===")
    print(f"  LogReg do sklearn (Etapa 6)          {pr_logreg:.4f}")
    print(f"  MLPClassifier hidden=(), alpha=0     {medias[i_zero]:.4f}   "
          f"[{medias[i_zero] - pr_logreg:+.4f}]")
    print("  Se estes dois divergirem muito, o que separa os candidatos não é")
    print("  só a profundidade — é também o otimizador (Adam x LBFGS).")

    # === 3. o early stopping que pontua por ACURÁCIA =======================
    # A afirmação está no fonte do sklearn (`_score_with_function(...,
    # score_function=accuracy_score)`, sem `scoring` exposto). Aqui ela é
    # MEDIDA: o número que decidiu a parada aparece ao lado do número que o
    # projeto usa para decidir qualquer coisa.
    melhor_rede = {k: v for k, v in params_1se.items()}
    if melhor_rede["hidden_layer_sizes"] == ():
        # A 1-SE elegeu a degenerada, como na Etapa 8. O entregável exige uma
        # REDE, então segue-se a melhor configuração COM camada oculta — e as
        # duas vão para a tabela, que é o que a Etapa 8 também fez.
        i_rede = max(
            (i for i, p in enumerate(r["params"])
             if p["modelo__hidden_layer_sizes"] != ()),
            key=lambda i: medias[i],
        )
        melhor_rede = {k.replace("modelo__", ""): v for k, v in r["params"][i_rede].items()}
    print(f"\n=== EARLY STOPPING — o custo do default, medido (rede {melhor_rede}) ===")
    for es in (False, True):
        df = fase_seeds(X, y, cv, melhor_rede, early_stopping=es)
        print(f"  early_stopping={es!s:<5} PR-AUC {df.pr_auc.mean():.4f} "
              f"± {df.pr_auc.std():.4f} (5 seeds)")

    # === 4. a validação — o número que entra na tabela mestra ==============
    print("\n=== VALIDAÇÃO — tocada uma vez, com a configuração já escolhida ===")
    df_val = fase_validacao(X, y, X_val, y_val, melhor_rede)
    for _, s in df_val.iterrows():
        print(f"  seed {int(s.seed):>6}  PR-AUC {s.pr_auc:.4f}  Brier {s.brier:.4f}  "
              f"limiar {s.limiar_otimo:.2f}  R$ {s.custo_brl:>8,.0f}  "
              f"{int(s.n_iter):>4} iterações"
              f"{'  ⚠️ parou no teto' if s.parou_no_teto else ''}")
    print(f"\n  MLPClassifier {melhor_rede['hidden_layer_sizes']}  PR-AUC "
          f"{df_val.pr_auc.mean():.4f} ± {df_val.pr_auc.std():.4f} · "
          f"Brier {df_val.brier.mean():.4f} · "
          f"custo R$ {df_val.custo_brl.mean():,.0f} · "
          f"R@10% {df_val.recall_at_10.mean():.3f}")

    df_es = fase_validacao(X, y, X_val, y_val, melhor_rede, early_stopping=True)
    print(f"  o mesmo com early_stopping=True      PR-AUC "
          f"{df_es.pr_auc.mean():.4f} ± {df_es.pr_auc.std():.4f} · "
          f"Brier {df_es.brier.mean():.4f} · "
          f"custo R$ {df_es.custo_brl.mean():,.0f} · "
          f"R@10% {df_es.recall_at_10.mean():.3f}")
    print(f"  ⇒ o score que DECIDIU a parada foi {df_es.score_de_parada.mean():.4f} — "
          "e é ACURÁCIA, não PR-AUC.")
    print("    Duas réguas no mesmo treino: uma escolhe a época, a outra avalia o")
    print("    resultado. É o defeito que a implementação em PyTorch corrige.")

    # === 5. convergência ===================================================
    print(f"\n=== CONVERGÊNCIA — {n_convergencia} avisos em "
          f"{len(r['params']) * n_dobras} ajustes da grade (max_iter={MAX_ITER}) ===")
    print("  Um ConvergenceWarning não interrompe nada e devolve os pesos onde")
    print("  parou: o modelo treina, prediz e nunca mais reclama. Contá-los é o")
    print("  que impede o número da tabela de descrever um treino interrompido.")

    # === 6. o registro =====================================================
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="mlp-sklearn-etapa8bis"):
        mlflow.log_params({
            "familia": "MLPClassifier",
            "hidden_layer_sizes": str(melhor_rede["hidden_layer_sizes"]),
            "alpha": melhor_rede["alpha"],
            "max_iter": MAX_ITER,
            "early_stopping": False,
            "n_folds": N_FOLDS, "n_repeticoes": reps,
            "seed": config.SEED,
            "dataset_sha256": data.sha256_dataset(),
        })
        mlflow.log_metrics({
            "cv_pr_auc_1se": float(medias[i_1se]),
            "cv_pr_auc_rede": float(medias[i_pico]),
            "val_pr_auc": float(df_val.pr_auc.mean()),
            "val_pr_auc_dp_seeds": float(df_val.pr_auc.std()),
            "val_brier": float(df_val.brier.mean()),
            "val_custo_brl": float(df_val.custo_brl.mean()),
            "val_recall_at_10": float(df_val.recall_at_10.mean()),
            "val_pr_auc_early_stopping": float(df_es.pr_auc.mean()),
            "controle_hidden_vazio": float(medias[i_zero]),
            "controle_logreg": pr_logreg,
            "avisos_convergencia": n_convergencia,
            "desvio_predicao_entre_seeds": df_val.attrs["desvio_predicao"],
        })
    print("\nRegistrado no MLflow. Nada foi promovido: a promoção continua sendo")
    print("`make promover`, e o campeão continua sendo decidido pelo gate.")


if __name__ == "__main__":
    main()

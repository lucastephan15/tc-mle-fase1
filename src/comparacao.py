"""
Comparação de algoritmos — Etapa 6.

    python -m src.comparacao                 # a comparação completa
    python -m src.comparacao --rapido        # 5 folds x 1, para iterar

Responde à pergunta "qual família de modelo serve a este problema?" com um
protocolo único para todos os candidatos, porque comparação em condições
diferentes não é comparação.

TRÊS DECISÕES METODOLÓGICAS, todas deliberadas:

1. **CV estratificada REPETIDA no treino** (5 folds x 3 repetições = 15 fits por
   candidato), não um split único. Na Etapa 4 um split único disse que as
   features novas ganhavam +0,0067 — a CV mostrou que isso cabia quatro vezes
   dentro do desvio entre folds. O mesmo ceticismo se aplica aqui, e é por isso
   que o desvio-padrão é reportado ao lado de toda média: sem ele não existe
   comparação, existe torcida.

2. **O encoding é eixo do experimento, não constante.** One-hot é o certo para a
   LogReg e cobra pedágio da árvore: espalha uma feature por k dummies, o que
   (a) gasta profundidade — o que seria um split vira até k-1 — e (b) dilui o
   sorteio do `max_features` da Random Forest, dando à feature dominante menos
   chance de ser candidata em cada nó. Comparar árvore e modelo linear com o
   mesmo encoding faz a árvore competir com uma mão nas costas.

3. **A árvore única não está aqui para ganhar.** Ela é o CONTROLE do argumento
   viés-variância: uma árvore sem poda tem viés baixíssimo e variância altíssima,
   e é o desvio entre folds dela (comparado ao da floresta, que é a média de
   várias) que justifica empiricamente o ensemble. Se ela perder feio e oscilar
   muito, o experimento funcionou.

O tuning é a Etapa 7. Aqui cada família roda numa configuração de referência
declarada, não ajustada aos dados.
"""

from __future__ import annotations

import argparse
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from src import config, data
from src.preprocess import construir_pipeline, mascara_categorica
from src.train import commit_hash

EXPERIMENTO = "churn-fase1"
N_FOLDS = 5
N_REPS = 3


def catalogo(encoding: str):
    """Os estimadores da Etapa 6, em configuração de REFERÊNCIA (não tunada).

    Devolve (estimador, escalonar). O escalonamento só existe para a LogReg:
    split por limiar é invariante a transformação monotônica, então padronizar
    antes de uma árvore não muda absolutamente nada — declarar `escalonar=False`
    é honestidade sobre o que o pipeline faz, não otimização.

    Sobre `n_estimators=300` na RF: NÃO é tuning. É a assimetria que a M02-A04
    estabelece — mais árvores nunca faz a Random Forest sobreajustar, só custa
    tempo. Usar 300 em vez do default 100 REMOVE uma fonte de variância do
    próprio estimador e torna a comparação mais estável. No boosting seria o
    oposto: lá `n_estimators` É um regularizador, e por isso o HGB fica no seu
    default com `early_stopping` interno.
    """
    return {
        "logreg": (
            LogisticRegression(max_iter=1000, random_state=config.SEED),
            True,
        ),
        "arvore": (
            # Sem poda, de propósito: é o CONTROLE do argumento viés-variância.
            # `max_depth=None` é o default e faz a árvore crescer até a pureza.
            DecisionTreeClassifier(random_state=config.SEED),
            False,
        ),
        "rf": (
            RandomForestClassifier(
                n_estimators=300, random_state=config.SEED, n_jobs=-1,
            ),
            False,
        ),
        # Mesma família, com a ÚNICA restrição que a Etapa 4 já usava. Não é
        # tuning: é a exigência de justiça do gate — declarar vencedor sobre um
        # adversário sabidamente mal configurado (árvores crescidas até a pureza
        # em 4.225 amostras) não é comparação, é armar o resultado. O ajuste fino
        # continua sendo a Etapa 7.
        "rf_reg": (
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5,
                random_state=config.SEED, n_jobs=-1,
            ),
            False,
        ),
        "hgb_reg": (
            HistGradientBoostingClassifier(
                random_state=config.SEED,
                learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=1.0,
                categorical_features=(
                    mascara_categorica() if encoding == "ordinal" else None
                ),
            ),
            False,
        ),
        "hgb": (
            HistGradientBoostingClassifier(
                random_state=config.SEED,
                # Categórica NATIVA: o HGB particiona o conjunto de níveis num
                # único split, em vez de tratar o código ordinal como número
                # ordenado. É o tratamento que XGBoost/LightGBM popularizaram e
                # o motivo de o encoding ordinal não ser um mal menor aqui.
                categorical_features=(
                    mascara_categorica() if encoding == "ordinal" else None
                ),
            ),
            False,
        ),
    }


def avaliar_candidato(X, y, nome: str, encoding: str, rapido: bool) -> dict:
    """CV repetida de um candidato. Devolve média, desvio e tempo."""
    estimador, escalonar = catalogo(encoding)[nome]
    pipe = construir_pipeline(estimador, escalonar=escalonar, encoding=encoding)
    cv = RepeatedStratifiedKFold(
        n_splits=N_FOLDS,
        n_repeats=1 if rapido else N_REPS,
        random_state=config.SEED,
    )
    t0 = time.perf_counter()
    # 'average_precision' é o PR-AUC — a métrica primária da Etapa 0. Deixar
    # 'accuracy' aqui por inércia mediria outra coisa (e o chute na majoritária
    # já daria 73%).
    r = cross_validate(
        pipe, X, y, cv=cv, scoring="average_precision",
        n_jobs=-1, return_train_score=True,
    )
    return {
        "modelo": nome,
        "encoding": encoding,
        "pr_auc": float(r["test_score"].mean()),
        "dp": float(r["test_score"].std()),
        "treino": float(r["train_score"].mean()),
        "segundos": time.perf_counter() - t0,
    }


def variancia_predicao(X, y, nome: str, encoding: str, n: int = 10) -> float:
    """Quanto a predição oscila quando o treino muda um pouco.

    Esta é a medida DIRETA da variância no sentido da decomposição viés-variância
    — e a razão de existir dos ensembles. Treina o mesmo estimador em `n`
    reamostragens bootstrap do treino, prediz sempre no MESMO conjunto (a
    validação, que nenhuma delas viu) e devolve o desvio-padrão médio da
    probabilidade prevista, cliente a cliente.

    É a versão medida da frase da aula: *"dependendo do conjunto de treino, a
    regra da raiz mudava"*. Uma árvore instável dá opiniões diferentes sobre o
    mesmo cliente; a floresta, que já é a média de 300 delas, quase não muda.
    """
    estimador, escalonar = catalogo(encoding)[nome]
    dados = data.dividir()
    X_val = dados.validacao.X
    rng = np.random.default_rng(config.SEED)
    predicoes = []
    for _ in range(n):
        idx = rng.choice(len(X), size=len(X), replace=True)
        pipe = construir_pipeline(estimador, escalonar=escalonar, encoding=encoding)
        pipe.fit(X.iloc[idx], y.iloc[idx])
        predicoes.append(pipe.predict_proba(X_val)[:, 1])
    # desvio por cliente, depois a média sobre os clientes
    return float(np.std(np.vstack(predicoes), axis=0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Etapa 6 — comparação de algoritmos")
    ap.add_argument("--rapido", action="store_true",
                    help="1 repetição em vez de 3, para iterar")
    args = ap.parse_args()

    np.random.seed(config.SEED)
    dados = data.dividir()
    X, y = dados.treino.X, dados.treino.y

    reps = 1 if args.rapido else N_REPS
    print(f"Etapa 6 — CV estratificada repetida no TREINO "
          f"({N_FOLDS} folds x {reps} = {N_FOLDS * reps} fits por candidato)")
    print(f"Métrica: PR-AUC · n_treino={len(X)} · prevalência={y.mean():.4f}\n")

    # A LogReg só entra com one-hot: ordinal num modelo linear seria erro
    # grosseiro, porque ele leria 'Contract=2' como o dobro de 'Contract=1' —
    # impondo ordem e escala a uma variável que não tem nenhuma das duas.
    plano = [("logreg", "onehot")]
    for m in ("arvore", "rf", "rf_reg", "hgb", "hgb_reg"):
        plano += [(m, "onehot"), (m, "ordinal")]

    linhas = [avaliar_candidato(X, y, m, e, args.rapido) for m, e in plano]
    df = pd.DataFrame(linhas).sort_values("pr_auc", ascending=False)

    base = df[df.modelo == "logreg"].iloc[0]
    df["delta"] = df.pr_auc - base.pr_auc
    # Em quantos desvios-padrão o ganho cabe. É a pergunta da Etapa 4 outra vez:
    # "isso é sinal ou é variância do sorteio?". Abaixo de ~1 dp, é sorteio.
    df["delta_em_dp"] = df.delta / base.dp
    df["gap"] = df.treino - df.pr_auc

    print(f"{'modelo':<8} {'encoding':<9} {'PR-AUC':>8} {'±dp':>7} "
          f"{'Δ base':>8} {'Δ/dp':>6} {'treino':>7} {'gap':>7} {'seg':>6}")
    print("-" * 76)
    for _, r in df.iterrows():
        print(f"{r.modelo:<8} {r.encoding:<9} {r.pr_auc:>8.4f} {r.dp:>7.4f} "
              f"{r.delta:>+8.4f} {r.delta_em_dp:>+6.2f} {r.treino:>7.4f} "
              f"{r.gap:>7.4f} {r.segundos:>6.1f}")

    print(f"\nRuído de referência: o desvio entre folds do baseline é {base.dp:.4f}.")
    print("Ganho menor que isso não é ganho — é o sorteio da partição.\n")

    # --- o experimento do encoding, isolado --------------------------------
    print("Efeito do encoding (mesma família, mesmo protocolo):")
    for m in ("arvore", "rf", "rf_reg", "hgb", "hgb_reg"):
        oh = df[(df.modelo == m) & (df.encoding == "onehot")].iloc[0]
        od = df[(df.modelo == m) & (df.encoding == "ordinal")].iloc[0]
        print(f"  {m:<8} onehot {oh.pr_auc:.4f} → ordinal {od.pr_auc:.4f} "
              f"({od.pr_auc - oh.pr_auc:+.4f}, {(od.pr_auc - oh.pr_auc) / base.dp:+.2f} dp)")

    # --- o argumento viés-variância, medido corretamente -------------------
    # ⚠️ O desvio entre folds NÃO serve para isto: ele mistura a variância do
    # modelo com o erro de estimativa de cada fold, e ainda é medido em escalas
    # de PR-AUC muito diferentes. As duas leituras honestas são o GAP (o quanto
    # o modelo memoriza sem generalizar) e a VARIÂNCIA DE PREDIÇÃO abaixo.
    print("\nViés-variância — o gap treino → validação cruzada:")
    for m in ("arvore", "rf", "rf_reg"):
        r = df[(df.modelo == m) & (df.encoding == "ordinal")].iloc[0]
        print(f"  {m:<8} treino {r.treino:.4f} → CV {r.pr_auc:.4f} "
              f"(gap {r.gap:.4f})")
    print("  Árvore e floresta memorizam o treino IGUALMENTE (ambas ~1,0). O que")
    print("  a agregação muda é só a generalização — e é isso a redução de variância.")

    print("\nVariância do modelo (desvio da predição em 10 reamostragens do treino):")
    for m in ("arvore", "rf_reg"):
        v = variancia_predicao(X, y, m, "ordinal")
        print(f"  {m:<8} desvio médio da probabilidade prevista: {v:.4f}")
    print("  É a medida direta do 'comitê de especialistas': treinado em dados")
    print("  ligeiramente diferentes, quanto a opinião do modelo oscila?\n")

    mlflow.set_tracking_uri(f"sqlite:///{config.RAIZ / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="etapa6-comparacao"):
        mlflow.set_tags({
            "dataset_sha256": dados.sha256,
            "commit": commit_hash(),
            "etapa": "6-comparacao",
            "avaliado_em": "cv-treino",  # validação e teste seguem intocados
        })
        mlflow.log_params({
            "n_folds": N_FOLDS, "n_repeticoes": reps,
            "seed": config.SEED, "n_features": len(config.FEATURES),
        })
        for _, r in df.iterrows():
            chave = f"{r.modelo}_{r.encoding}"
            mlflow.log_metrics({
                f"prauc_{chave}": r.pr_auc,
                f"dp_{chave}": r.dp,
                f"gap_{chave}": r.gap,
            })
        caminho = config.PROCESSED / "etapa6_comparacao.csv"
        caminho.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(caminho, index=False)
        mlflow.log_artifact(str(caminho))

    print(f"Resultado salvo em {caminho} e registrado no MLflow.")


if __name__ == "__main__":
    main()

"""
Etapa 6, fase 2 — os finalistas na validação.

    python -m src.finalistas

A fase 1 (`src.comparacao`) usou validação cruzada no TREINO para escolher os
candidatos. Esta fase leva os três finalistas à VALIDAÇÃO, uma única vez, para
responder o que a CV não responde:

1. **Qual é o limiar de operação DE CADA MODELO.** O 0,22 do baseline foi
   derivado da razão de custo 3:1 sobre a distribuição de probabilidades da
   LogReg, e não transfere: a Random Forest é uma média de votos e comprime a
   probabilidade para o centro; o boosting extremiza. Herdar o corte deslocaria
   silenciosamente o ponto de operação — o volume da fila de retenção mudaria
   sem ninguém ter decidido isso.

2. **Quanto se paga em calibração.** PR-AUC mede ORDENAÇÃO; o Brier mede se a
   probabilidade significa alguma coisa. Um modelo pode subir num e piorar no
   outro — não é bug, e importa porque a fila é ordenada por P(churn) x CLTV.

3. **Quais features cada um usa** — e aqui a medida é `permutation_importance`
   na validação, não `feature_importances_`. A importância por impureza do
   sklearn é calculada no treino e é enviesada para variáveis com muitos pontos
   de corte candidatos: contínuas e alta cardinalidade parecem mais importantes
   do que são. A EDA da Etapa 1 serve de gabarito independente.

⛔ O conjunto de TESTE continua intocado.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config, data, evaluate
from src.comparacao import EXPERIMENTO, catalogo
from src.preprocess import construir_pipeline
from src.train import commit_hash

# Os três que empataram na fase 1, cada um com o encoding que lhe serve.
FINALISTAS = [("logreg", "onehot"), ("rf_reg", "ordinal"), ("hgb_reg", "ordinal")]


def treinar(nome: str, encoding: str, X, y):
    estimador, escalonar = catalogo(encoding)[nome]
    pipe = construir_pipeline(estimador, escalonar=escalonar, encoding=encoding)
    return pipe.fit(X, y)


def main() -> None:
    np.random.seed(config.SEED)
    dados = data.dividir()
    X, y = dados.treino.X, dados.treino.y
    X_val, y_val = dados.validacao.X, dados.validacao.y

    print(f"Etapa 6 · fase 2 — finalistas na VALIDAÇÃO (n={len(X_val)}, "
          f"prevalência={y_val.mean():.4f})\n")

    modelos, linhas = {}, []
    for nome, enc in FINALISTAS:
        pipe = treinar(nome, enc, X, y)
        modelos[nome] = pipe
        p = pipe.predict_proba(X_val)[:, 1]

        custo = evaluate.curva_custo(y_val, p)
        m = evaluate.avaliar(y_val, p, limiar=custo["limiar_otimo"])
        linhas.append({
            "modelo": nome, "encoding": enc,
            "pr_auc": m["pr_auc"], "brier": m["brier"],
            "limiar_otimo": custo["limiar_otimo"],
            "custo_brl": custo["custo_otimo_brl"],
            "custo_com_limiar_022": None,  # preenchido abaixo
            "recall_at_10": m["recall_at_10"],
            "recall_at_20": m["recall_at_20"],
            "p_min": float(p.min()), "p_max": float(p.max()),
            "p_desvio": float(p.std()),
        })
        # O contrafactual que prova o ponto 1: quanto custaria HERDAR o 0,22.
        y_h = (p >= 0.22).astype(int)
        fn = int(((y_val == 1) & (y_h == 0)).sum())
        fp = int(((y_val == 0) & (y_h == 1)).sum())
        linhas[-1]["custo_com_limiar_022"] = float(fn * 194 + fp * 62)

    df = pd.DataFrame(linhas)

    print(f"{'modelo':<9} {'PR-AUC':>8} {'Brier':>7} {'limiar':>7} "
          f"{'custo R$':>10} {'se herdasse 0,22':>17} {'R@10%':>7} {'R@20%':>7}")
    print("-" * 82)
    for _, r in df.iterrows():
        extra = r.custo_com_limiar_022 - r.custo_brl
        print(f"{r.modelo:<9} {r.pr_auc:>8.4f} {r.brier:>7.4f} "
              f"{r.limiar_otimo:>7.2f} {r.custo_brl:>10,.0f} "
              f"{r.custo_com_limiar_022:>11,.0f} (+{extra:>4,.0f}) "
              f"{r.recall_at_10:>7.3f} {r.recall_at_20:>7.3f}")

    print("\nForma da distribuição de probabilidades (por que o limiar não transfere):")
    for _, r in df.iterrows():
        print(f"  {r.modelo:<9} min {r.p_min:.3f} · max {r.p_max:.3f} · "
              f"desvio {r.p_desvio:.3f}")

    # --- importância: permutação (validação) x impureza (treino) -----------
    print("\nImportância das features — permutação na VALIDAÇÃO, top 8:")
    for nome in ("logreg", "rf_reg"):
        pipe = modelos[nome]
        r = permutation_importance(
            pipe, X_val, y_val, n_repeats=10,
            random_state=config.SEED, scoring="average_precision", n_jobs=-1,
        )
        ordem = np.argsort(r.importances_mean)[::-1][:8]
        print(f"\n  [{nome}]")
        for i in ordem:
            print(f"    {X_val.columns[i]:<22} {r.importances_mean[i]:>+8.4f} "
                  f"± {r.importances_std[i]:.4f}")

    # O teste de sanidade: o MDI concorda com a permutação e com a EDA?
    rf = modelos["rf_reg"]
    mdi = rf.named_steps["modelo"].feature_importances_
    nomes = rf.named_steps["preproc"].get_feature_names_out()
    top_mdi = [nomes[i] for i in np.argsort(mdi)[::-1][:5]]
    print(f"\n  MDI (feature_importances_, medido no TREINO) top 5: {top_mdi}")
    print("  ⚠️ Comparar com o ranking acima e com a EDA (Contract dominou por 39,9 pp).")

    mlflow.set_tracking_uri(f"sqlite:///{config.RAIZ / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="etapa6-finalistas"):
        mlflow.set_tags({
            "dataset_sha256": dados.sha256, "commit": commit_hash(),
            "etapa": "6-finalistas", "avaliado_em": "validacao",
        })
        for _, r in df.iterrows():
            for c in ("pr_auc", "brier", "limiar_otimo", "custo_brl"):
                mlflow.log_metric(f"{r.modelo}_{c}", float(r[c]))
        caminho = config.PROCESSED / "etapa6_finalistas.csv"
        df.to_csv(caminho, index=False)
        mlflow.log_artifact(str(caminho))
    print(f"\nSalvo em {caminho} e registrado no MLflow.")


if __name__ == "__main__":
    main()

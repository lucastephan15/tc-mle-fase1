"""
Seleção de features — Etapa 5.

    python -m src.selecao

Quatro perguntas, nesta ordem:

  A. Quantas das 46 colunas pós one-hot são realmente necessárias? (filtro,
     wrapper e embedded — os três tipos da Aula 03)
  B. Alguma feature ORIGINAL pode sair inteira? — é esta que reduz custo de
     coleta, validação e monitoramento em produção, não a A.
  C. O que fazer com 'Total Charges', que correlaciona 0,9996 com
     Tenure x Monthly? (item 9 do backlog de revisita)
  D. Quanto o leakage do seletor infla a métrica, se rodado fora do Pipeline?

Toda medição é por CV estratificada no TREINO, mesma metodologia da Etapa 4 —
a validação segue reservada para a escolha final entre modelos.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config, data
from src.preprocess import construir_pipeline, construir_preprocessador

N_FOLDS = 5


def modelo():
    return LogisticRegression(max_iter=1000, random_state=config.SEED)


def cv(X, y, **kwargs) -> tuple[float, float]:
    pipe = construir_pipeline(modelo(), **kwargs)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
    notas = cross_val_score(pipe, X, y, cv=folds, scoring="average_precision", n_jobs=-1)
    return float(notas.mean()), float(notas.std())


def secao(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main() -> None:
    dados = data.dividir()
    X, y = dados.treino.X, dados.treino.y
    base_m, base_s = cv(X, y)
    print(f"Referência — 19 features / 46 colunas: PR-AUC {base_m:.4f} ± {base_s:.4f}")
    print(f"(CV estratificada, {N_FOLDS} folds, no treino)")

    # --- A. quantas colunas bastam ----------------------------------------
    secao("A · Os três tipos de seleção (Aula 03), sobre as 46 colunas")
    print(f"{'método':<38} {'k':>4} {'PR-AUC':>8} {'±dp':>7} {'Δ':>9}")
    print("-" * 72)
    for k in (10, 15, 20, 30, 46):
        m, s = cv(X, y, seletor=SelectKBest(f_classif, k=k))
        print(f"{'filtro · SelectKBest(f_classif)':<38} {k:>4} {m:>8.4f} {s:>7.4f} {m - base_m:>+9.4f}")
    for k in (10, 15, 20):
        m, s = cv(X, y, seletor=RFE(modelo(), n_features_to_select=k))
        print(f"{'wrapper · RFE':<38} {k:>4} {m:>8.4f} {s:>7.4f} {m - base_m:>+9.4f}")
    for c in (0.1, 0.05):
        sel = SelectFromModel(LogisticRegression(
            penalty="l1", C=c, solver="liblinear", max_iter=1000,
            random_state=config.SEED))
        m, s = cv(X, y, seletor=sel)
        print(f"{f'embedded · L1 (C={c})':<38} {'—':>4} {m:>8.4f} {s:>7.4f} {m - base_m:>+9.4f}")

    # --- B. ablação por feature ORIGINAL ----------------------------------
    secao("B · Remoção de feature ORIGINAL inteira (o que reduz custo real)")
    print(f"{'sem qual feature':<38} {'PR-AUC':>8} {'±dp':>7} {'perda':>9}")
    print("-" * 72)
    perdas = []
    for f in config.FEATURES:
        m, s = cv(X, y, excluir=[f])
        perdas.append((f, base_m - m))
        print(f"{'sem ' + f:<38} {m:>8.4f} {s:>7.4f} {base_m - m:>+9.4f}")

    print("\nOrdenado por perda (as que mais importam primeiro):")
    for f, p in sorted(perdas, key=lambda t: -t[1])[:6]:
        print(f"  {p:>+8.4f}  {f}")
    descartaveis = [f for f, p in perdas if p <= 0]
    print(f"\nRemover MELHORA ou não muda ({len(descartaveis)}): {descartaveis}")

    # Remover as 9 juntas: o efeito individual pode ser ruído e o acumulado não.
    m_9, s_9 = cv(X, y, excluir=descartaveis)
    print(f"\nremovendo as {len(descartaveis)} de uma vez  -> {m_9:.4f} ± {s_9:.4f} "
          f"(Δ {m_9 - base_m:+.4f}) com {len(config.FEATURES) - len(descartaveis)} features")

    # --- C. o caso Total Charges ------------------------------------------
    secao("C · 'Total Charges' — multicolinearidade (item 9 do revisita)")
    m_sem, s_sem = cv(X, y, excluir=["Total Charges"])
    print(f"com Total Charges : {base_m:.4f} ± {base_s:.4f}")
    print(f"sem Total Charges : {m_sem:.4f} ± {s_sem:.4f}   (Δ {m_sem - base_m:+.4f})")

    # --- D. o gate: leakage do seletor ------------------------------------
    secao("D · GATE — SelectKBest ajustado FORA do Pipeline (leakage)")
    pre = construir_preprocessador().fit(X)           # ⚠️ fit em TUDO, de propósito
    Xt = pre.transform(X)
    sel = SelectKBest(f_classif, k=15).fit(Xt, y)      # ⚠️ vê o alvo de todos os folds
    Xsel = sel.transform(Xt)
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
    vazado = cross_val_score(modelo(), Xsel, y, cv=folds,
                             scoring="average_precision", n_jobs=-1)
    correto_m, _ = cv(X, y, seletor=SelectKBest(f_classif, k=15))
    print(f"seleção DENTRO do pipeline (correto) : {correto_m:.4f}")
    print(f"seleção FORA do pipeline (vazado)    : {vazado.mean():.4f}")
    print(f"inflação artificial                  : {vazado.mean() - correto_m:+.4f}")
    print("\n⚠️ O gap treino-teste NÃO detectaria isto: os dois lados foram")
    print("   contaminados igualmente. Não é overfitting, é o instrumento quebrado.")

    # --- D-bis. o mesmo erro no regime em que ele MORDE -------------------
    secao("D-bis · O mesmo erro com poucas amostras e muitas features (p >> n)")
    print("Se a inflação acima deu ~zero, é porque com 4.225 amostras a seleção é")
    print("estável: qualquer 80% dos dados elege quase as mesmas colunas. O")
    print("mecanismo só morde quando a seleção é INSTÁVEL. Demonstração:")
    print("300 amostras + 500 colunas de RUÍDO PURO, sem relação com o alvo.\n")

    rng = np.random.default_rng(config.SEED)
    n = 300
    idx = rng.choice(len(y), size=n, replace=False)
    y_p = y.to_numpy()[idx]
    ruido = rng.normal(size=(n, 500))  # nenhuma coluna tem sinal, por construção

    folds_p = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
    # errado: escolhe as 20 "melhores" olhando o alvo de TODAS as amostras
    sel_v = SelectKBest(f_classif, k=20).fit(ruido, y_p)
    vaz = cross_val_score(modelo(), sel_v.transform(ruido), y_p, cv=folds_p,
                          scoring="average_precision", n_jobs=-1)
    # certo: a seleção refaz dentro de cada fold, sem ver o fold de validação
    from sklearn.pipeline import Pipeline as P
    ok = cross_val_score(P([("s", SelectKBest(f_classif, k=20)), ("m", modelo())]),
                         ruido, y_p, cv=folds_p, scoring="average_precision", n_jobs=-1)
    print(f"prevalência (o piso honesto)      : {y_p.mean():.4f}")
    print(f"seleção DENTRO do pipeline        : {ok.mean():.4f}")
    print(f"seleção FORA do pipeline (vazado) : {vaz.mean():.4f}")
    print(f"inflação sobre puro ruído         : {vaz.mean() - ok.mean():+.4f}")
    print("\n→ O modelo 'aprendeu' a prever a partir de colunas ALEATÓRIAS. Todo")
    print("  esse desempenho é artefato da seleção ter visto o alvo inteiro.")


if __name__ == "__main__":
    np.random.seed(config.SEED)
    main()

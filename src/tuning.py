"""
Tuning de hiperparâmetros — Etapa 7.

    python -m src.tuning              # a busca completa (grid + random)
    python -m src.tuning --rapido     # 1 repetição da CV, para iterar

Os dois finalistas da Etapa 6 (LogReg e HistGradientBoosting regularizado) sob
o **mesmo protocolo** da comparação: CV estratificada repetida 5x3 no TREINO,
métrica PR-AUC, mesma seed, mesmas 13 features. Mudar o protocolo aqui tornaria
os números incomparáveis com a tabela da Etapa 6 — e a pergunta da etapa é
justamente "o tuning move alguma coisa?".

TRÊS DECISÕES METODOLÓGICAS, todas pré-registradas no decision log §5c:

1. **A estratégia de busca é escolhida pelo TAMANHO da grade, não por gosto.**
   LogReg tem 12 configurações e dois eixos sabidamente relevantes -> grid
   exaustivo, que é barato e garante o ótimo dentro da grade. O HGB tem cinco
   eixos e suspeita de eixo inerte -> busca aleatória (Bergstra & Bengio, 2012):
   num grid, um eixo que não importa gasta k vezes o orçamento medindo a mesma
   coisa; na busca aleatória cada amostra testa um valor novo de TODOS.

2. **O `refit` é a regra 1-SE, não o `argmax`.** O pico da grade é, em boa
   medida, ruído favorável — o `max` sobre medições ruidosas captura sorte, e o
   viés cresce com o tamanho da grade. A 1-SE (Hastie et al., 2009) escolhe o
   candidato MAIS REGULARIZADO cujo score está dentro de 1 erro padrão do melhor.

3. **O melhor score da grade NÃO é reportado como generalização.** Ele foi usado
   para escolher, logo é otimista por construção. O número que vai para a
   documentação sai da VALIDAÇÃO, e o teste segue intocado.
"""

from __future__ import annotations

import argparse
import json
import time

import mlflow
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
)

from src import config, data, evaluate
from src.comparacao import EXPERIMENTO, N_FOLDS, N_REPS
from src.preprocess import construir_pipeline, mascara_categorica
from src.train import commit_hash

N_ITER_RANDOM = 40


# --- as grades -------------------------------------------------------------


def grade_logreg() -> dict:
    """`C` em escala LOGARÍTMICA x tipo de penalização — os dois eixos da M02-A02.

    A escala é logarítmica porque o efeito da regularização é multiplicativo:
    uma grade linear (0,5 · 1,0 · 1,5 …) gastaria quase todo o orçamento numa
    região onde nada muda e não visitaria as ordens de grandeza que importam.

    ⚠️ `C = 1/lambda` — escala INVERTIDA em relação ao `alpha` do Ridge/Lasso.
    Menor `C` = MAIS penalização. Montar a grade no piloto automático com a
    intuição de `alpha` inverteria o experimento inteiro.

    ⚠️ O segundo eixo NÃO é `penalty`, apesar de a aula e a literatura o
    chamarem assim: o scikit-learn 1.8 DEPRECIOU o parâmetro (removido em 1.10)
    e unificou tudo em `l1_ratio` — `0.0` é a L2 pura, `1.0` a L1 pura, e
    qualquer valor no meio é elastic net. Ganho colateral: o eixo deixou de ser
    binário, e `l1_ratio=0.5` entra de graça como terceiro ponto.

    Solver `saga` nos três pontos, de propósito. É o único que aceita a faixa
    inteira, e usar o mesmo em todos mantém a comparação controlada: `liblinear`
    penalizaria também o intercepto, mudando a função objetivo justamente no
    eixo que estamos medindo. Como o problema é convexo, `saga` com
    `l1_ratio=0, C=1` converge para a MESMA solução do LBFGS do baseline — o que
    torna o baseline um ponto de dentro da grade, e não um número de fora.
    """
    return {
        "modelo__C": np.logspace(-3, 2, 6).tolist(),
        "modelo__l1_ratio": [0.0, 0.5, 1.0],
        "modelo__solver": ["saga"],
    }


def grade_hgb() -> dict:
    """Cinco eixos, amostrados aleatoriamente.

    ⚠️ `n_estimators` (aqui `max_iter`) NÃO entra na grade — e essa ausência é a
    assimetria central da M02-A04. Na Random Forest mais árvores nunca
    sobreajusta (só custa tempo); no boosting mais árvores É overfitting, porque
    cada uma corrige o resíduo da anterior. A receita da aula é fixar um
    `learning_rate` pequeno e deixar a VALIDAÇÃO escolher onde parar — que é
    exatamente o `early_stopping` interno configurado no estimador.

    `max_leaf_nodes` baixo é o análogo do `max_depth` 3-8 da aula: o modelo-base
    do boosting tem de ser FRACO de propósito.

    ⚠️ `max_leaf_nodes` começa em **2** (o toco de decisão — literalmente a
    árvore mais fraca que existe) e `min_samples_leaf` vai até 320 porque a
    primeira execução bateu na borda dos dois eixos, e nos dois casos na direção
    de MENOS capacidade. Estender a grade para onde o aviso apontou é o que
    diferencia um aviso de borda útil de um alerta decorativo.
    """
    return {
        "modelo__learning_rate": stats.loguniform(0.01, 0.2),
        "modelo__max_leaf_nodes": [2, 3, 4, 7, 15, 31],
        "modelo__min_samples_leaf": [10, 20, 40, 80, 160, 320],
        "modelo__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "modelo__max_features": [0.5, 0.7, 1.0],
    }


# --- ordem de complexidade: o que "mais simples" significa em cada família ---
#
# A regra 1-SE foi formulada para um eixo de complexidade DENTRO de uma família,
# então "mais simples" precisa de uma ordenação declarada — sem ela a regra não
# tem como decidir. Menor é mais simples nas duas funções abaixo.


def complexidade_logreg(p: dict) -> tuple:
    """Menor `C` = mais penalizado = mais simples. Mais L1 desempata.

    O desempate por L1 não é estético: L1 zera coeficientes, então dois modelos
    com o mesmo `C` não têm a mesma complexidade efetiva — o mais esparso usa
    menos parâmetros para dizer a mesma coisa.
    """
    return (p["modelo__C"], -p["modelo__l1_ratio"])


def complexidade_hgb(p: dict) -> tuple:
    """Menos capacidade primeiro: folhas, taxa, tamanho mínimo, penalização.

    A ordem dos critérios reflete o quanto cada eixo restringe a família de
    funções: o número de folhas limita a forma que uma árvore pode ter, o
    `learning_rate` limita o quanto cada uma pode contribuir, e os dois últimos
    são regularizadores secundários (por isso entram com o sinal invertido —
    MAIS restrição é MENOS complexidade).
    """
    return (
        p["modelo__max_leaf_nodes"],
        round(p["modelo__learning_rate"], 4),
        -p["modelo__min_samples_leaf"],
        -p["modelo__l2_regularization"],
    )


# Eixos cujo extremo é fronteira do DOMÍNIO, não da grade: não adianta procurar
# fora deles. `l1_ratio=1.0` já é a L1 pura e `max_features=1.0` já são todas as
# colunas — avisar "o ótimo pode estar fora" nesses casos é ruído, e alerta que
# dispara sem ação possível é o começo da fadiga de alerta.
BORDAS_NATURAIS = {"modelo__l1_ratio", "modelo__max_features",
                   "modelo__l2_regularization"}


def avisos_de_borda(grade, params: dict) -> list[str]:
    """O ótimo caiu no extremo de algum eixo? (provocação obrigatória da etapa)

    Confere contra a GRADE DECLARADA, não contra os valores que a busca
    aleatória por acaso sorteou — num `RandomizedSearchCV` de 40 amostras, um
    valor da lista pode simplesmente nunca ter sido visitado, e comparar com o
    sorteado inventaria uma borda que não existe.

    Eixos definidos por distribuição contínua (`loguniform`) não têm lista para
    comparar; o critério vira "caiu nos 10% extremos da faixa amostrada".
    """
    if isinstance(grade, list):  # grade em sub-dicionários
        grade = {k: v for sub in grade for k, v in sub.items()}
    avisos = []
    for eixo, valores in grade.items():
        v = params.get(eixo)
        if v is None or eixo in BORDAS_NATURAIS:
            continue
        if isinstance(valores, list):
            if len(valores) > 1 and v in (min(valores), max(valores)):
                avisos.append(f"{eixo}={v} está na BORDA da grade "
                              f"({min(valores)} … {max(valores)}) — "
                              f"o ótimo pode estar fora dela.")
        else:  # distribuição: usa o suporte declarado
            lo, hi = valores.support()
            faixa = np.log10(hi) - np.log10(lo)
            if np.log10(v) - np.log10(lo) < 0.1 * faixa or np.log10(hi) - np.log10(v) < 0.1 * faixa:
                avisos.append(f"{eixo}={v:.4f} caiu nos 10% extremos da faixa "
                              f"amostrada ({lo} … {hi}).")
    return avisos


def refit_1se(ordem, n_dobras: int):
    """Devolve o `refit` callable que aplica a regra 1-SE.

    *"Escolher o modelo mais simples cujo desempenho esteja a no máximo 1
    desvio-padrão do desempenho do melhor modelo"* — Hastie et al. (2009), citado
    na M02-A07. Trocar o `argmax` por isto custa uma função e compra robustez
    medida: no hands-on da aula (300 seeds), a 1-SE recupera a complexidade
    verdadeira em 98% das amostras contra 66% do `argmin`.

    ⚠️ O erro padrão usado aqui é `dp/sqrt(K)` com K = 15 dobras, que é a fórmula
    do material — e ela SUPERESTIMA a precisão, porque as dobras de uma CV
    repetida não são independentes (as repetições reusam os mesmos dados). É a
    correção de Nadeau-Bengio, ausente da aula. Consequência: este é o envelope
    MAIS ESTREITO dos três, ou seja, a versão mais conservadora da regra — ela
    troca de modelo menos vezes do que trocaria com o envelope correto. O
    relatório imprime as três leituras para que a conclusão não dependa desta.
    """

    def escolher(cv_results) -> int:
        medias = np.asarray(cv_results["mean_test_score"])
        dps = np.asarray(cv_results["std_test_score"])
        i_melhor = int(np.argmax(medias))
        piso = medias[i_melhor] - dps[i_melhor] / np.sqrt(n_dobras)
        dentro = np.flatnonzero(medias >= piso)
        # `min` do Python, não `np.argmin`: a chave de complexidade é uma TUPLA
        # (comparação lexicográfica), e o numpy achataria o array 2D e devolveria
        # um índice do array plano — um bug silencioso quando a grade é pequena
        # o bastante para o índice ainda existir.
        return int(min(dentro, key=lambda i: ordem(cv_results["params"][i])))

    return escolher


# --- a busca ---------------------------------------------------------------


def estimador_base(nome: str):
    """O ponto de partida de cada busca, em configuração declarada.

    O HGB entra com `scoring='average_precision'` no early stopping interno —
    NÃO no default. O default de `HistGradientBoostingClassifier` é parar pela
    *loss*, o que significa escolher o número de árvores por um critério que não
    é a métrica primária do projeto, dentro do laço de treino, sem log nem
    warning. Todo critério de parada é uma escolha de métrica: se você não a
    escolheu, o default escolheu por você.
    """
    if nome == "logreg":
        return LogisticRegression(max_iter=5000, random_state=config.SEED), True, "onehot"
    if nome == "hgb":
        return (
            HistGradientBoostingClassifier(
                random_state=config.SEED,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                scoring="average_precision",
                categorical_features=mascara_categorica(),
            ),
            False,
            "ordinal",
        )
    raise ValueError(nome)


def buscar(nome: str, X, y, cv, rapido: bool) -> dict:
    estimador, escalonar, encoding = estimador_base(nome)
    pipe = construir_pipeline(estimador, escalonar=escalonar, encoding=encoding)
    n_dobras = cv.get_n_splits()

    comum = dict(
        scoring="average_precision",  # a métrica da Etapa 0, não o default
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    if nome == "logreg":
        grade = grade_logreg()
        busca = GridSearchCV(
            pipe, grade,
            refit=refit_1se(complexidade_logreg, n_dobras), **comum,
        )
    else:
        grade = grade_hgb()
        busca = RandomizedSearchCV(
            pipe, grade,
            n_iter=10 if rapido else N_ITER_RANDOM,
            random_state=config.SEED,
            refit=refit_1se(complexidade_hgb, n_dobras), **comum,
        )

    t0 = time.perf_counter()
    busca.fit(X, y)
    segundos = time.perf_counter() - t0

    r = busca.cv_results_
    i_pico = int(np.argmax(r["mean_test_score"]))
    i_1se = int(busca.best_index_)
    # O ponto da grade que corresponde ao DEFAULT da biblioteca (C=1.0, L2 pura)
    # — ou seja, o modelo que já estava no repositório desde a Etapa 3. Sem ele,
    # "o tuning ganhou" não teria contra o quê ser medido.
    i_default = next(
        (i for i, p in enumerate(r["params"])
         if p.get("modelo__C") == 1.0 and p.get("modelo__l1_ratio") == 0.0),
        None,
    )
    return {
        "nome": nome, "busca": busca, "encoding": encoding, "grade": grade,
        "i_pico": i_pico, "i_1se": i_1se, "i_default": i_default,
        "segundos": segundos, "n_configs": len(r["params"]),
        # os scores por dobra dos dois candidatos — é o que habilita o teste
        # pareado adiante, e vem de graça porque a CV tem seed fixa
        "scores_1se": np.array([r[f"split{i}_test_score"][i_1se] for i in range(n_dobras)]),
        "scores_pico": np.array([r[f"split{i}_test_score"][i_pico] for i in range(n_dobras)]),
    }


def envelopes(medias: np.ndarray, dps: np.ndarray, i: int, n_dobras: int) -> dict:
    """As três leituras da dispersão — a robustez é a concordância entre elas.

    - `se`: `dp/sqrt(15)`, a fórmula do material. A mais estreita, e otimista
      sobre a precisão porque as dobras de CV repetida não são independentes.
    - `conservador`: `dp/sqrt(5)`, tratando apenas as dobras de UMA repetição
      como independentes. É a aproximação prática da correção de Nadeau-Bengio.
    - `desvio_cheio`: o desvio-padrão inteiro, sem dividir por nada. O envelope
      mais largo, e o que corresponde à leitura literal de "1 desvio-padrão".
    """
    dp = float(dps[i])
    m = float(medias[i])
    return {
        "se": m - dp / np.sqrt(n_dobras),
        "conservador": m - dp / np.sqrt(N_FOLDS),
        "desvio_cheio": m - dp,
    }


def comparar_pareado(a: np.ndarray, b: np.ndarray) -> dict:
    """t pareado e Wilcoxon sobre as MESMAS dobras.

    O pareamento não custou nada: o `random_state` fixo no
    `RepeatedStratifiedKFold` faz todos os candidatos verem exatamente as mesmas
    partições, então a diferença por dobra isola o efeito do modelo do efeito do
    sorteio. Wilcoxon é o mais defensável com K=15 — não supõe normalidade.

    ⚠️ Os dois p-valores são ANTICONSERVADORES pelo mesmo motivo do envelope
    `se`: dobras de CV repetida não são independentes (Nadeau-Bengio). Servem
    como evidência corroborante da 1-SE, nunca como decisor isolado.
    """
    d = a - b
    t = stats.ttest_rel(a, b)
    # `zero_method='zsplit'` evita que dobras empatadas sumam da amostra
    w = stats.wilcoxon(a, b, zero_method="zsplit")
    return {
        "delta_medio": float(d.mean()),
        "vitorias": int((d > 0).sum()),
        "p_t": float(t.pvalue),
        "p_wilcoxon": float(w.pvalue),
    }


def main() -> None:  # noqa: PLR0915 — é um relatório linear, quebrar piora a leitura
    ap = argparse.ArgumentParser(description="Etapa 7 — tuning de hiperparâmetros")
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

    print(f"Etapa 7 — tuning · CV repetida no TREINO ({n_dobras} dobras por config)")
    print(f"Métrica: PR-AUC · n_treino={len(X)} · protocolo idêntico ao da Etapa 6\n")

    resultados = [buscar(n, X, y, cv, args.rapido) for n in ("logreg", "hgb")]

    linhas = []
    for res in resultados:
        r = res["busca"].cv_results_
        medias, dps = np.asarray(r["mean_test_score"]), np.asarray(r["std_test_score"])
        i_pico, i_1se = res["i_pico"], res["i_1se"]
        env = envelopes(medias, dps, i_pico, n_dobras)

        print(f"=== {res['nome'].upper()} — {res['n_configs']} configurações "
              f"em {res['segundos']:.0f}s ===")
        print(f"  pico da grade    {medias[i_pico]:.4f} ± {dps[i_pico]:.4f}  "
              f"{res['busca'].cv_results_['params'][i_pico]}")
        print(f"  escolha 1-SE     {medias[i_1se]:.4f} ± {dps[i_1se]:.4f}  "
              f"{res['busca'].cv_results_['params'][i_1se]}")
        print(f"  envelope 1-SE    dp/√{n_dobras} {env['se']:.4f} · "
              f"dp/√{N_FOLDS} (Nadeau-Bengio) {env['conservador']:.4f} · "
              f"dp cheio {env['desvio_cheio']:.4f}")
        dentro = {k: int((medias >= v).sum()) for k, v in env.items()}
        print(f"  candidatos dentro do envelope: {dentro}  (de {res['n_configs']})")
        if i_pico == i_1se:
            print("  → a 1-SE e o argmax coincidem nesta grade.")
        else:
            print(f"  → a 1-SE trocou o pico por um modelo mais simples "
                  f"(custo: {medias[i_pico] - medias[i_1se]:+.4f}, "
                  f"{(medias[i_1se] - medias[i_pico]) / dps[i_pico]:+.2f} dp).")

        for aviso in avisos_de_borda(res["grade"], r["params"][i_1se]):
            print(f"  ⚠️ {aviso}")
        # a pergunta que dá sentido à etapa: o tuning ganhou do default?
        if res["i_default"] is not None:
            i_d = res["i_default"]
            ganho = medias[i_1se] - medias[i_d]
            print(f"  vs. DEFAULT ({r['params'][i_d]['modelo__C']} / l1_ratio "
                  f"{r['params'][i_d]['modelo__l1_ratio']}): {medias[i_d]:.4f} → "
                  f"{medias[i_1se]:.4f} = {ganho:+.4f} "
                  f"({ganho / dps[i_d]:+.2f} dp do próprio default)")
        print()

        linhas.append({
            "modelo": res["nome"], "encoding": res["encoding"],
            "cv_pico": float(medias[i_pico]), "cv_1se": float(medias[i_1se]),
            "dp_1se": float(dps[i_1se]),
            "treino_1se": float(r["mean_train_score"][i_1se]),
            "params": json.dumps(r["params"][i_1se], default=str),
            "n_configs": res["n_configs"], "segundos": res["segundos"],
        })

    # --- o teste pareado, agora que os scores por dobra existem -------------
    a, b = resultados[0]["scores_1se"], resultados[1]["scores_1se"]
    par = comparar_pareado(b, a)  # HGB - LogReg
    print("Teste pareado nas MESMAS 15 dobras (HGB − LogReg):")
    print(f"  Δ médio {par['delta_medio']:+.4f} · HGB vence em "
          f"{par['vitorias']}/{n_dobras} dobras")
    print(f"  t pareado p={par['p_t']:.3f} · Wilcoxon p={par['p_wilcoxon']:.3f}")
    print("  ⚠️ Anticonservadores: dobras de CV repetida não são independentes")
    print("     (Nadeau-Bengio). Corroboram a 1-SE, não decidem sozinhos.\n")

    # --- a validação: o número honesto -------------------------------------
    # ⛔ O que foi impresso acima escolheu configurações, logo é otimista por
    # construção. O número que vai para a documentação é este.
    print("VALIDAÇÃO — tocada uma vez, com o modelo já escolhido:")

    # O CAMPEÃO ATUAL entra na mesma tela: é o modelo que já está no repositório
    # desde a Etapa 3 (LogReg nos defaults) e o que o gate do CI protege. Sem
    # esta linha, "o tuning melhorou" não teria referência — e a resposta a essa
    # pergunta é o produto da etapa, não a tabela de hiperparâmetros.
    campeao = construir_pipeline(
        LogisticRegression(max_iter=1000, random_state=config.SEED),
        escalonar=True, encoding="onehot",
    ).fit(X, y)
    p_c = campeao.predict_proba(X_val)[:, 1]
    custo_c = evaluate.curva_custo(y_val, p_c)
    m_c = evaluate.avaliar(y_val, p_c, limiar=custo_c["limiar_otimo"])
    print(f"  {'CAMPEÃO':<7} PR-AUC {m_c['pr_auc']:.4f} (Etapa 3, sem tuning)"
          f"       · Brier {m_c['brier']:.4f} · "
          f"limiar {custo_c['limiar_otimo']:.2f} · "
          f"custo R$ {custo_c['custo_otimo_brl']:,.0f} · "
          f"R@10% {m_c['recall_at_10']:.3f}")

    for res, linha in zip(resultados, linhas, strict=True):
        pipe = res["busca"].best_estimator_
        p = pipe.predict_proba(X_val)[:, 1]
        custo = evaluate.curva_custo(y_val, p)
        m = evaluate.avaliar(y_val, p, limiar=custo["limiar_otimo"])
        linha |= {
            "val_pr_auc": m["pr_auc"], "val_brier": m["brier"],
            "limiar_otimo": custo["limiar_otimo"],
            "custo_brl": custo["custo_otimo_brl"],
            "recall_at_10": m["recall_at_10"], "recall_at_20": m["recall_at_20"],
        }
        print(f"  {res['nome']:<7} PR-AUC {m['pr_auc']:.4f} (CV disse "
              f"{linha['cv_1se']:.4f}) · Brier {m['brier']:.4f} · "
              f"limiar {custo['limiar_otimo']:.2f} · "
              f"custo R$ {custo['custo_otimo_brl']:,.0f} · "
              f"R@10% {m['recall_at_10']:.3f}")

    # --- quanto da queda CV → validação é VIÉS DE SELEÇÃO, e quanto é partição -
    # O campeão é o controle: ele não foi escolhido por grade nenhuma, então a
    # queda dele mede o que a troca de conjunto custa por si só. O EXCEDENTE
    # sobre essa queda é o que se pode atribuir à seleção — e a previsão da aula
    # é que ele cresça com o tamanho da grade.
    queda_controle = 0.6904 - m_c["pr_auc"]  # 0,6904 = CV do campeão na Etapa 6
    print(f"\n  Queda CV → validação (o campeão como CONTROLE, 0 candidatos): "
          f"{queda_controle:+.4f}")
    for res, linha in zip(resultados, linhas, strict=True):
        q = linha["val_pr_auc"] - linha["cv_1se"]
        print(f"    {res['nome']:<7} {res['n_configs']:>2} candidatos: "
              f"{q:+.4f}  (excedente sobre o controle: {q + queda_controle:+.4f})")

    est_logreg = resultados[0]["busca"].best_estimator_.named_steps["modelo"]
    print(f"\n  Convergência do SAGA: n_iter={est_logreg.n_iter_} "
          f"(max_iter={est_logreg.max_iter}) — sem coeficientes 'onde parou'.")
    # Quanto a L1 efetivamente encolheu o modelo. Importa para a Etapa 11: uma
    # coluna com coeficiente zerado sai da tabela de odds ratios, e a seleção
    # embutida se soma à poda explícita que a Etapa 5 já fez.
    coefs = est_logreg.coef_.ravel()
    nomes = resultados[0]["busca"].best_estimator_.named_steps[
        "preproc"].get_feature_names_out()
    zerados = int((np.abs(coefs) < 1e-8).sum())
    print(f"  Coeficientes zerados pela L1: {zerados} de {coefs.size} "
          f"({zerados / coefs.size:.0%} das colunas pós-encoding).")
    # ⛔ E agora a pergunta que decide se essa esparsidade VALE alguma coisa:
    # zerar dummy não é zerar feature. A coluna original só sai do custo de
    # coleta, validação e monitoramento se TODAS as suas dummies zerarem — é a
    # mesma distinção que estruturou a Etapa 5, aplicada agora à seleção
    # embutida em vez da explícita.
    mortas = [
        f for f in config.FEATURES
        if all(abs(c) < 1e-8
               for c, n in zip(coefs, nomes, strict=True) if n.startswith(f))
    ]
    print(f"  → colunas ORIGINAIS eliminadas inteiras: "
          f"{mortas if mortas else 'nenhuma'} "
          f"({len(mortas)} de {len(config.FEATURES)}).")
    est_hgb = resultados[1]["busca"].best_estimator_.named_steps["modelo"]
    print(f"  Árvores efetivamente usadas pelo HGB: {est_hgb.n_iter_} de "
          f"{est_hgb.max_iter} (early stopping por PR-AUC).")

    df = pd.DataFrame(linhas)

    mlflow.set_tracking_uri(f"sqlite:///{config.RAIZ / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="etapa7-tuning"):
        mlflow.set_tags({
            "dataset_sha256": dados.sha256, "commit": commit_hash(),
            "etapa": "7-tuning", "avaliado_em": "cv-treino+validacao",
            "regra_selecao": "1-SE (Hastie et al., 2009)",
        })
        mlflow.log_params({
            "n_folds": N_FOLDS, "n_repeticoes": reps, "seed": config.SEED,
            "n_iter_random": N_ITER_RANDOM,
        })
        for _, r in df.iterrows():
            for c in ("cv_pico", "cv_1se", "val_pr_auc", "val_brier",
                      "limiar_otimo", "custo_brl"):
                mlflow.log_metric(f"{r.modelo}_{c}", float(r[c]))
        for k, v in par.items():
            mlflow.log_metric(f"pareado_{k}", float(v))
        caminho = config.PROCESSED / "etapa7_tuning.csv"
        caminho.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(caminho, index=False)
        mlflow.log_artifact(str(caminho))

    print(f"\nSalvo em {caminho} e registrado no MLflow.")


if __name__ == "__main__":
    main()

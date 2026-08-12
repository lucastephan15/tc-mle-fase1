"""
MLP em PyTorch — Etapa 8.

    python -m src.mlp                 # o experimento completo
    python -m src.mlp --rapido        # 1 repetição da CV, para iterar

A rede é exigência da Fase 1, mas a ordem importa: ela entra **depois** do
baseline, da comparação e do tuning, para que sua adoção (ou rejeição) seja
justificada por evidência em vez de por enunciado.

A DECISÃO QUE ESTRUTURA O ARQUIVO: o MLP é embrulhado como um estimador do
scikit-learn (`MLPTorch`) e entra no MESMO `construir_pipeline` dos demais
candidatos, sob a MESMA CV estratificada repetida 5x3, com a MESMA seed e as
MESMAS 13 features. Não é conveniência de código — é a condição para o número
conversar com a tabela das Etapas 6 e 7, e é o que faz o teste pareado nas
mesmas 15 dobras sair de graça.

🎯 E é isso que torna a etapa um EXPERIMENTO CONTROLADO. Entre a LogReg e o MLP
não muda a perda (a log-verossimilhança da logística, com sinal trocado, é a
entropia cruzada — o `BCEWithLogitsLoss` daqui), nem a forma da saída, nem o
encoding, nem o pré-processamento. Muda **uma variável só: a profundidade**.
Logo o delta LogReg -> MLP mede, isolada, quanta não-linearidade existe no
Telco. RF e HGB trocam a família inteira de uma vez, e é por isso que o empate
da Etapa 6 é ambíguo sobre *o que* empatou.

O `hidden=()` na grade é o controle desse controle: um MLP de zero camadas
ocultas **é** uma regressão logística (M02-A05, pág. 12). Se ele não reproduzir
o 0,6904 da Etapa 6, o problema é esta implementação — e não um achado sobre os
dados. É o primeiro número a olhar.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")  # sem display: o script roda em CI e por linha de comando

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from torch import nn

from src import config, data, evaluate, features
from src.comparacao import EXPERIMENTO, N_FOLDS, N_REPS
from src.preprocess import construir_pipeline, mascara_categorica
from src.train import commit_hash
from src.tuning import comparar_pareado, envelopes, refit_1se

# 5 seeds, porque `random_state` é FATOR do experimento e não detalhe de
# execução. LogReg é convexa (mesma partição => mesma solução) e as árvores têm
# `random_state`; o MLP não tem essa propriedade — a superfície de erro é
# não-convexa e os pesos partem de inicialização aleatória, então o mesmo dado
# produz modelos diferentes. Comparar uma média com um sorteio é comparar
# réguas diferentes, e o vencedor pode ter sido decidido pelo sorteio.
SEEDS = [42, 7, 123, 2024, 31337]

# A configuração que a regra 1-SE elegeu na Etapa 7, transcrita de
# data/processed/etapa7_tuning.csv. É ela o adversário — não a configuração de
# referência da Etapa 6. Declarar vencedor sobre um boosting mal ajustado
# repetiria o erro que o gate de justiça da Etapa 6 existe para impedir.
HGB_TUNADO = {
    "l2_regularization": 1.0,
    "learning_rate": 0.11973258472758717,
    "max_features": 0.7,
    "max_leaf_nodes": 2,
    "min_samples_leaf": 80,
}


# --- o estimador -----------------------------------------------------------


class MLPTorch(ClassifierMixin, BaseEstimator):
    """Rede densa em PyTorch com a interface do scikit-learn.

    Existe para que o MLP possa ser avaliado pelo protocolo que já está no
    repositório em vez de por um laço de treino paralelo — `cross_validate`,
    `GridSearchCV`, `Pipeline` e o scorer `average_precision` funcionam sem
    adaptação, e o resultado entra na mesma tabela dos outros candidatos.

    Três escolhas que não são default e precisam de justificativa:

    1. **`BCEWithLogitsLoss`, nunca `Sigmoid` + `BCELoss`.** A versão fundida
       aplica o truque do log-sum-exp internamente; a versão separada satura em
       float32 e produz gradiente nulo (ou `inf` na loss) quando o logit fica
       grande, que é exatamente o regime de um classificador confiante.

    2. **`Adam(lr=1e-3)`.** O `SGD(lr=1.0)` do hands-on da aula é um
       hiperparâmetro calibrado para um problema de **quatro pontos** (XOR);
       importado para 4.225 linhas com 28 colunas padronizadas, ele diverge.

    3. **Early stopping por PR-AUC numa validação interna** extraída do treino,
       com restauração dos pesos da MELHOR época. O `early_stopping=True` do
       `MLPClassifier` do sklearn pontua com `accuracy_score` — está no fonte e
       **não é configurável**, a classe não expõe `scoring`. Num problema de
       prevalência 26,5% com custo 3:1, isso é otimizar a métrica errada dentro
       do laço de treino, onde não há log nem warning para denunciar.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (16,),
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_epocas: int = 300,
        batch_size: int = 256,
        paciencia: int = 25,
        frac_val_interna: float = 0.15,
        seed: int = config.SEED,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        # `weight_decay` do Adam é a L2 sobre os PESOS — o terceiro alvo da mesma
        # penalidade que já apareceu nos coeficientes (Etapa 7) e nas folhas do
        # boosting. Com capacidade sobrando, o eixo que decide é este, não o
        # tamanho da rede: "aproximador universal" garante que a capacidade
        # existe, não que ela seja encontrável nem necessária.
        self.weight_decay = weight_decay
        self.max_epocas = max_epocas
        self.batch_size = batch_size
        self.paciencia = paciencia
        self.frac_val_interna = frac_val_interna
        self.seed = seed

    def _rede(self, n_entrada: int) -> nn.Sequential:
        """Camadas densas com ReLU; a última devolve LOGIT, não probabilidade.

        Com `hidden=()` sobra um único `Linear(n, 1)`: literalmente a regressão
        logística, e é assim que o controle de profundidade zero é construído.
        """
        camadas: list[nn.Module] = []
        entrada = n_entrada
        for h in self.hidden:
            camadas += [nn.Linear(entrada, h), nn.ReLU()]
            entrada = h
        camadas.append(nn.Linear(entrada, 1))
        return nn.Sequential(*camadas)

    def fit(self, X, y):
        # Single-thread de propósito: a redução paralela do BLAS muda a ordem
        # das somas e com ela o último bit do resultado, o que faz o mesmo
        # código com a mesma seed dar números ligeiramente diferentes entre
        # execuções. Com 4.225x28 não há nada a ganhar em paralelizar dentro do
        # fit — o paralelismo que importa é o da CV, entre processos.
        torch.set_num_threads(1)

        X = np.asarray(X, dtype=np.float32)
        y_int = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_int)
        self.n_features_in_ = X.shape[1]

        # A semente cobre inicialização dos pesos E ordem dos minibatches: as
        # duas fontes de aleatoriedade do treino.
        torch.manual_seed(self.seed)
        gerador = torch.Generator().manual_seed(self.seed)

        # ⛔ A validação interna sai do TREINO, não da partição da Etapa 2.
        # Parar o treino olhando a validação faria do número de épocas mais um
        # parâmetro escolhido nela — o mesmo viés de seleção que a Etapa 7 mediu
        # com grupo de controle, um andar abaixo e sem sintoma visível.
        # Estratificada porque 26,5% de prevalência num recorte de 15% pode
        # facilmente sair com proporção diferente e mover o PR-AUC de parada.
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y_int,
            test_size=self.frac_val_interna,
            stratify=y_int,
            random_state=self.seed,
        )

        rede = self._rede(X.shape[1])
        otimizador = torch.optim.Adam(
            rede.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        criterio = nn.BCEWithLogitsLoss()

        Xt = torch.from_numpy(X_tr)
        yt = torch.from_numpy(y_tr.astype(np.float32))
        Xv = torch.from_numpy(X_va)
        yv = torch.from_numpy(y_va.astype(np.float32))

        melhor_score, melhor_epoca, espera = -np.inf, 0, 0
        melhores_pesos = None
        historico = []

        for epoca in range(1, self.max_epocas + 1):
            rede.train()
            ordem = torch.randperm(len(Xt), generator=gerador)
            perda_acumulada = 0.0
            for i in range(0, len(ordem), self.batch_size):
                idx = ordem[i : i + self.batch_size]
                otimizador.zero_grad()
                logits = rede(Xt[idx]).squeeze(1)
                perda = criterio(logits, yt[idx])
                perda.backward()
                otimizador.step()
                perda_acumulada += float(perda.detach()) * len(idx)

            rede.eval()
            with torch.no_grad():
                logits_va = rede(Xv).squeeze(1)
                perda_va = float(criterio(logits_va, yv))
                p_va = torch.sigmoid(logits_va).numpy()
            score = float(average_precision_score(y_va, p_va))
            historico.append({
                "epoca": epoca,
                "loss_treino": perda_acumulada / len(Xt),
                "loss_val": perda_va,
                "pr_auc_val": score,
            })

            # A tolerância evita que ruído na 12ª casa zere a paciência e o
            # treino nunca pare.
            if score > melhor_score + 1e-6:
                melhor_score, melhor_epoca, espera = score, epoca, 0
                # Os pesos da MELHOR época, não os da última. Parar por
                # paciência e ficar com o estado final entregaria um modelo pior
                # que aquele que o próprio critério de parada elegeu.
                melhores_pesos = {k: v.detach().clone()
                                  for k, v in rede.state_dict().items()}
            else:
                espera += 1
                if espera >= self.paciencia:
                    break

        rede.load_state_dict(melhores_pesos)
        rede.eval()
        self.rede_ = rede
        self.historico_ = pd.DataFrame(historico)
        self.melhor_epoca_ = melhor_epoca
        self.melhor_score_interno_ = melhor_score
        self.n_epocas_ = len(historico)
        self.n_parametros_ = sum(p.numel() for p in rede.parameters())
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            p = torch.sigmoid(self.rede_(torch.from_numpy(X)).squeeze(1)).numpy()
        p = p.astype(np.float64)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        # O 0,5 aqui é referência, não o limiar de operação — quem define o
        # corte é a economia da campanha (Etapa 0), e ele é re-derivado por
        # modelo (Etapa 6). Nada neste projeto decide por `predict`.
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --- a grade ---------------------------------------------------------------


def grade_mlp() -> dict:
    """Arquitetura x penalização L2 — os dois eixos, ambos de REGULARIZAÇÃO.

    Com 4.225 amostras de treino e 28 colunas pós-encoding, uma camada de 32
    neurônios já são ~950 pesos: **4 amostras por parâmetro**. Nesse regime o
    tamanho da rede não é "capacidade a explorar", é "capacidade a conter" — por
    isso `hidden` entra ao lado de `weight_decay` e não como eixo de ganho.

    `()` está na grade como CONTROLE: zero camadas ocultas é a regressão
    logística, e o valor dela aqui tem de bater com o 0,6904 da Etapa 6.
    """
    return {
        "modelo__hidden": [(), (8,), (16,), (32,), (16, 16)],
        "modelo__weight_decay": [0.0, 1e-3, 1e-2],
    }


def complexidade_mlp(p: dict) -> tuple:
    """Ordem de complexidade para a regra 1-SE: menor é mais simples.

    Profundidade primeiro, largura depois, penalização por último (invertida —
    mais L2 é menos complexidade). A ordem reflete o quanto cada eixo amplia a
    família de funções representáveis: uma camada a mais muda o que a rede PODE
    expressar; mais neurônios refinam a mesma classe de funções; o
    `weight_decay` só restringe a região do espaço de pesos.
    """
    hidden = p["modelo__hidden"]
    return (len(hidden), sum(hidden), -p["modelo__weight_decay"])


class LogRegMesmoOrcamento(ClassifierMixin, BaseEstimator):
    """A LogReg do sklearn treinada nos MESMOS 85% do treino que o MLP vê.

    Existe por causa de um confundimento descoberto na primeira execução: o MLP
    de profundidade zero ficou abaixo da LogReg, e **duas** coisas diferem entre
    os dois ao mesmo tempo — o otimizador (Adam + early stopping × LBFGS + L2) e
    o orçamento de dados, porque o early stopping do MLP consome 15% do treino
    numa validação interna que a LogReg não precisa reservar.

    Este estimador fixa o segundo fator: mesmo split, mesma seed, mesmos 85%.
    Com ele a diferença se decompõe em duas leituras separadas, em vez de uma
    diferença única atribuída à explicação que mais agradar.
    """

    def __init__(self, seed: int = config.SEED, frac_val_interna: float = 0.15) -> None:
        self.seed = seed
        self.frac_val_interna = frac_val_interna

    def fit(self, X, y):
        X = np.asarray(X)
        y_int = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_int)
        X_tr, _, y_tr, _ = train_test_split(
            X, y_int, test_size=self.frac_val_interna,
            stratify=y_int, random_state=self.seed,
        )
        self.modelo_ = LogisticRegression(
            max_iter=1000, random_state=self.seed,
        ).fit(X_tr, y_tr)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.modelo_.predict_proba(np.asarray(X))

    def predict(self, X) -> np.ndarray:
        return self.modelo_.predict(np.asarray(X))


def pipe_mlp(seed: int = config.SEED, novas: list[str] | None = None, **kwargs):
    """O MLP no mesmo pipeline dos demais: one-hot + padronização.

    `escalonar=True` não é opcional aqui por dois motivos independentes: o
    gradiente não converge bem com colunas em escalas díspares, e é a igualdade
    de pré-processamento com a LogReg que sustenta o experimento controlado.
    Encoding ordinal seria o mesmo erro grosseiro que num modelo linear — a rede
    leria `Contract=2` como o dobro de `Contract=1`.
    """
    return construir_pipeline(
        MLPTorch(seed=seed, **kwargs),
        escalonar=True, encoding="onehot", novas=novas or [],
    )


# --- as fases do experimento ------------------------------------------------


def fase_grade(X, y, cv, rapido: bool) -> dict:
    """15 configurações sob o protocolo das Etapas 6 e 7, com `refit` 1-SE."""
    busca = GridSearchCV(
        pipe_mlp(),
        grade_mlp() if not rapido else {"modelo__hidden": [(), (16,)],
                                        "modelo__weight_decay": [0.0, 1e-2]},
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        # Mesma regra da Etapa 7: o pico de uma grade é em boa parte ruído
        # favorável, e o viés cresce com o tamanho dela.
        refit=refit_1se(complexidade_mlp, cv.get_n_splits()),
    )
    t0 = time.perf_counter()
    busca.fit(X, y)
    return {"busca": busca, "segundos": time.perf_counter() - t0}


def fase_seeds(X, y, cv, params: dict) -> pd.DataFrame:
    """O mesmo modelo, cinco inicializações — a variância que a CV não vê.

    A CV repetida 5x3 cobre variância **de partição**: ela reamostra os dados e
    é cega ao fato de que o MLP muda de solução sem que nada nos dados mude. As
    duas dispersões medem coisas diferentes e o critério registrado no item 36
    do revisita é comparar uma com a outra.
    """
    linhas = []
    for seed in SEEDS:
        pipe = pipe_mlp(seed=seed, **params)
        r = cross_validate(pipe, X, y, cv=cv, scoring="average_precision",
                           n_jobs=-1, return_train_score=True)
        linhas.append({
            "seed": seed,
            "pr_auc": float(r["test_score"].mean()),
            "dp_folds": float(r["test_score"].std()),
            "treino": float(r["train_score"].mean()),
            "scores": r["test_score"].tolist(),
        })
    return pd.DataFrame(linhas)


def fase_validacao(X, y, X_val, y_val, params: dict) -> pd.DataFrame:
    """As 5 seeds na validação — o número que vai para a documentação.

    Treina no treino inteiro (a validação interna do early stopping continua
    saindo de dentro dele) e mede uma vez na validação da Etapa 2. O teste segue
    intocado.
    """
    linhas, probabilidades = [], []
    for seed in SEEDS:
        pipe = pipe_mlp(seed=seed, **params).fit(X, y)
        p = pipe.predict_proba(X_val)[:, 1]
        probabilidades.append(p)
        custo = evaluate.curva_custo(y_val, p)
        m = evaluate.avaliar(y_val, p, limiar=custo["limiar_otimo"])
        est = pipe.named_steps["modelo"]
        linhas.append({
            "seed": seed,
            "pr_auc": m["pr_auc"], "brier": m["brier"],
            "limiar_otimo": custo["limiar_otimo"],
            "custo_brl": custo["custo_otimo_brl"],
            "recall_at_10": m["recall_at_10"], "recall_at_20": m["recall_at_20"],
            "epocas": est.n_epocas_, "melhor_epoca": est.melhor_epoca_,
            "n_parametros": est.n_parametros_,
        })
    df = pd.DataFrame(linhas)
    # Variância de INICIALIZAÇÃO na mesma unidade da variância de reamostragem
    # medida na Etapa 6 (árvore ±0,2462 × floresta ±0,0438): desvio da
    # probabilidade prevista por cliente, com os dados fixos e só a seed mudando.
    df.attrs["desvio_predicao"] = float(np.std(np.vstack(probabilidades), axis=0).mean())
    return df


def fase_features(X, y, cv, params: dict) -> dict:
    """Item 16 do revisita: as 4 features da Etapa 4 ajudam o MLP?

    Elas foram descartadas por ablação contra LogReg e RF. A hipótese que
    sobrava era "a rede aprende interação sozinha" — que corta nos dois sentidos:
    se a interação valesse algo, o MLP a acharia SEM a feature pronta.
    """
    novas = features.NOVAS_NUM + features.NOVAS_CAT
    pipe = pipe_mlp(novas=novas, **params)
    r = cross_validate(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    return {"pr_auc": float(r["test_score"].mean()), "dp": float(r["test_score"].std())}


def fase_curva(X, y, params: dict, caminho) -> dict:
    """Curva de treino x validação interna — o overfitting em forma de figura.

    É o diagnóstico que a M02-A04 pede para o boosting e que vale igual aqui: a
    loss de treino cai sempre; onde a de validação vira para cima é onde o
    modelo passou a decorar. O early stopping é a linha vertical.
    """
    pipe = pipe_mlp(**params).fit(X, y)
    h = pipe.named_steps["modelo"].historico_
    melhor = pipe.named_steps["modelo"].melhor_epoca_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(h.epoca, h.loss_treino, label="treino")
    ax1.plot(h.epoca, h.loss_val, label="validação interna")
    ax1.axvline(melhor, ls="--", c="gray", label=f"early stopping (época {melhor})")
    ax1.set_xlabel("época")
    ax1.set_ylabel("entropia cruzada")
    ax1.legend()
    ax1.set_title("Perda — a mesma que a LogReg otimiza")
    ax2.plot(h.epoca, h.pr_auc_val, c="tab:green")
    ax2.axvline(melhor, ls="--", c="gray")
    ax2.set_xlabel("época")
    ax2.set_ylabel("PR-AUC (validação interna)")
    ax2.set_title("O critério de parada é a métrica primária, não a acurácia")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return {"melhor_epoca": melhor, "n_epocas": len(h)}


# --- relatório --------------------------------------------------------------


def main() -> None:  # noqa: PLR0915 — relatório linear; quebrar piora a leitura
    ap = argparse.ArgumentParser(description="Etapa 8 — MLP em PyTorch")
    ap.add_argument("--rapido", action="store_true", help="1 repetição da CV")
    args = ap.parse_args()

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    dados = data.dividir()
    X, y = dados.treino.X, dados.treino.y
    X_val, y_val = dados.validacao.X, dados.validacao.y

    reps = 1 if args.rapido else N_REPS
    cv = RepeatedStratifiedKFold(
        n_splits=N_FOLDS, n_repeats=reps, random_state=config.SEED,
    )
    n_dobras = cv.get_n_splits()

    print(f"Etapa 8 — MLP em PyTorch · CV repetida no TREINO "
          f"({n_dobras} dobras por configuração)")
    print(f"Protocolo idêntico ao das Etapas 6 e 7 · n_treino={len(X)} · "
          f"13 features · one-hot + padronização\n")

    # === 1. a grade ========================================================
    res = fase_grade(X, y, cv, args.rapido)
    busca = res["busca"]
    r = busca.cv_results_
    medias = np.asarray(r["mean_test_score"])
    dps = np.asarray(r["std_test_score"])
    i_pico, i_1se = int(np.argmax(medias)), int(busca.best_index_)
    env = envelopes(medias, dps, i_pico, n_dobras)

    print(f"=== GRADE — {len(r['params'])} configurações em {res['segundos']:.0f}s ===")
    print(f"{'hidden':<10} {'w_decay':>8} {'PR-AUC':>8} {'±dp':>7} {'treino':>7} {'gap':>7}")
    print("-" * 52)
    ordem = np.argsort(medias)[::-1]
    for i in ordem:
        p = r["params"][i]
        marca = " ←1-SE" if i == i_1se else (" ←pico" if i == i_pico else "")
        print(f"{str(p['modelo__hidden']):<10} {p['modelo__weight_decay']:>8.3f} "
              f"{medias[i]:>8.4f} {dps[i]:>7.4f} {r['mean_train_score'][i]:>7.4f} "
              f"{r['mean_train_score'][i] - medias[i]:>7.4f}{marca}")

    print(f"\n  envelope 1-SE: dp/√{n_dobras} {env['se']:.4f} · "
          f"dp/√{N_FOLDS} (Nadeau-Bengio) {env['conservador']:.4f} · "
          f"dp cheio {env['desvio_cheio']:.4f}")
    dentro = {k: int((medias >= v).sum()) for k, v in env.items()}
    print(f"  candidatos dentro do envelope: {dentro} (de {len(medias)})")
    params_1se = {k.replace("modelo__", ""): v for k, v in r["params"][i_1se].items()}
    print(f"  escolha 1-SE: {params_1se}")

    # === 2. o controle de profundidade zero ================================
    # Se este número não bater com o da LogReg, nada abaixo significa coisa
    # alguma: o experimento inteiro assume que só a profundidade mudou.
    i_zero = next(i for i, p in enumerate(r["params"])
                  if p["modelo__hidden"] == () and p["modelo__weight_decay"] == 0.0)
    logreg_cv = cross_validate(
        construir_pipeline(LogisticRegression(max_iter=1000, random_state=config.SEED),
                           escalonar=True, encoding="onehot"),
        X, y, cv=cv, scoring="average_precision", n_jobs=-1,
    )
    pr_logreg = float(logreg_cv["test_score"].mean())
    # O terceiro número separa dois efeitos que a comparação de dois números
    # confunde: a LogReg com o MESMO orçamento de dados do MLP (85% do fold).
    orcamento_cv = cross_validate(
        construir_pipeline(LogRegMesmoOrcamento(), escalonar=True, encoding="onehot"),
        X, y, cv=cv, scoring="average_precision", n_jobs=-1,
    )
    pr_orcamento = float(orcamento_cv["test_score"].mean())
    print("\n=== CONTROLE — o MLP de ZERO camadas ocultas é a LogReg? ===")
    print(f"  (a) LogReg sklearn, 100% do fold      {pr_logreg:.4f}")
    print(f"  (b) LogReg sklearn, mesmos 85%        {pr_orcamento:.4f}   "
          f"[custo do early stopping: {pr_orcamento - pr_logreg:+.4f}]")
    print(f"  (c) MLPTorch hidden=(), mesmos 85%    {medias[i_zero]:.4f}   "
          f"[efeito do otimizador: {medias[i_zero] - pr_orcamento:+.4f}]")
    print(f"  diferença bruta (c)−(a) {medias[i_zero] - pr_logreg:+.4f} "
          f"({(medias[i_zero] - pr_logreg) / dps[i_zero]:+.2f} dp), decomposta acima.")
    print("  Mesma perda e mesma família de funções nos três; muda o otimizador")
    print("  (Adam + parada por PR-AUC × LBFGS + L2) e o orçamento de dados.")

    # === 3. os DOIS candidatos que a etapa precisa avaliar ==================
    # A 1-SE escolheu a configuração mais simples dentro do envelope, e neste
    # dataset isso pode significar profundidade ZERO — o que é o achado da
    # etapa, não um erro. Mas a fase exige uma REDE, e um MLP sem camada oculta
    # é a LogReg que já está no repositório. Então os dois seguem lado a lado:
    #   · `params_1se`  — o que a regra elegeu (o resultado metodológico)
    #   · `params_rede` — a melhor configuração COM camada oculta (o entregável)
    params_1se = {k.replace("modelo__", ""): v for k, v in r["params"][i_1se].items()}
    i_rede = max((i for i in range(len(medias)) if r["params"][i]["modelo__hidden"]),
                 key=lambda i: medias[i])
    params_rede = {k.replace("modelo__", ""): v for k, v in r["params"][i_rede].items()}
    degenerada = not params_1se["hidden"]

    print("\n=== OS DOIS CANDIDATOS ===")
    print(f"  escolha 1-SE : {params_1se}  →  CV {medias[i_1se]:.4f} ± {dps[i_1se]:.4f}")
    print(f"  melhor rede  : {params_rede}  →  CV {medias[i_rede]:.4f} ± {dps[i_rede]:.4f}")
    if degenerada:
        print("  ⚠️ A regra 1-SE, aplicada DENTRO da família das redes neurais, desceu")
        print("     até a profundidade zero — que é a regressão logística. A rede com")
        print("     camada oculta segue sendo avaliada, porque é o entregável da fase.")

    # === 4. a variância que a CV não enxerga ===============================
    print(f"\n=== SEEDS — {len(SEEDS)} inicializações, nos dois regimes ===")
    df_seeds = fase_seeds(X, y, cv, params_rede)
    for _, s in df_seeds.iterrows():
        print(f"  seed {int(s.seed):>6}  PR-AUC {s.pr_auc:.4f}  "
              f"(dp entre folds {s.dp_folds:.4f})")
    dp_seeds = float(df_seeds.pr_auc.std())
    dp_folds_medio = float(df_seeds.dp_folds.mean())
    print(f"  rede {str(params_rede['hidden']):<8} (NÃO-CONVEXO): média "
          f"{df_seeds.pr_auc.mean():.4f} ± {dp_seeds:.4f}")
    print(f"  desvio ENTRE SEEDS {dp_seeds:.4f} × desvio ENTRE FOLDS "
          f"{dp_folds_medio:.4f} = {dp_seeds / dp_folds_medio:.2f}×")
    print("  (item 36: se fossem da mesma ordem, nenhum número de seed única")
    print("   sustentaria conclusão — e o protocolo da Etapa 6 precisaria da")
    print("   dimensão seed. A medida é o que responde, não a intuição.)")

    df_seeds_1se = None
    if degenerada:
        # O contraste que separa as duas fontes de aleatoriedade: sem camada
        # oculta o problema volta a ser CONVEXO, então o que sobra de variação
        # entre seeds é o split interno + a ordem dos minibatches — não mínimo
        # local. Medir os dois é o que impede de chamar tudo de "variância de
        # inicialização".
        df_seeds_1se = fase_seeds(X, y, cv, params_1se)
        dp_1se = float(df_seeds_1se.pr_auc.std())
        print(f"\n  escolha 1-SE {str(params_1se['hidden']):<6} (CONVEXO): média "
              f"{df_seeds_1se.pr_auc.mean():.4f} ± {dp_1se:.4f}")
        print(f"    A rede varia {dp_seeds / dp_1se:.2f}× o convexo. Sem camada oculta")
        print("    não há mínimo local: o que resta é o split interno e a ordem dos lotes.")

    # === 5. teste pareado nas mesmas dobras ================================
    # A pergunta do experimento controlado: acrescentar profundidade acrescenta
    # alguma coisa? Mesmas dobras, mesma perda, mesmo pré-processamento.
    scores_rede = np.array([r[f"split{i}_test_score"][i_rede] for i in range(n_dobras)])
    par = comparar_pareado(scores_rede, logreg_cv["test_score"])
    print(f"\n=== PAREADO nas mesmas {n_dobras} dobras (rede − LogReg) ===")
    print(f"  Δ médio {par['delta_medio']:+.4f} · rede vence em "
          f"{par['vitorias']}/{n_dobras} · t p={par['p_t']:.3f} · "
          f"Wilcoxon p={par['p_wilcoxon']:.3f}")
    print("  ⚠️ Anticonservador (Nadeau-Bengio) e responde 'a diferença é")
    print("     consistente?', nunca 'a diferença importa?'.")

    # === 6. item 16 — as features da Etapa 4 com o MLP =====================
    feat = fase_features(X, y, cv, params_rede)
    base_cv = float(df_seeds[df_seeds.seed == config.SEED].pr_auc.iloc[0])
    print("\n=== ITEM 16 — as 4 features da Etapa 4, agora contra a rede ===")
    print(f"  sem features {base_cv:.4f}  →  com features {feat['pr_auc']:.4f} "
          f"({feat['pr_auc'] - base_cv:+.4f}, "
          f"{(feat['pr_auc'] - base_cv) / dp_folds_medio:+.2f} dp)")
    print("  'a rede aprende interação sozinha' corta nos dois sentidos: se a")
    print("  interação valesse algo, ela a acharia SEM a feature pronta.")

    # === 7. a validação — o número honesto =================================
    print("\n=== VALIDAÇÃO — tocada uma vez, com as arquiteturas já escolhidas ===")
    df_val = fase_validacao(X, y, X_val, y_val, params_rede)
    for _, s in df_val.iterrows():
        print(f"  seed {int(s.seed):>6}  PR-AUC {s.pr_auc:.4f}  Brier {s.brier:.4f}  "
              f"limiar {s.limiar_otimo:.2f}  R$ {s.custo_brl:>8,.0f}  "
              f"{int(s.epocas):>3} épocas (melhor: {int(s.melhor_epoca)})")

    n_par = int(df_val.n_parametros.iloc[0])
    print(f"\n  {'MLP ' + str(params_rede['hidden']):<12} PR-AUC "
          f"{df_val.pr_auc.mean():.4f} ± {df_val.pr_auc.std():.4f} · "
          f"Brier {df_val.brier.mean():.4f} · "
          f"custo R$ {df_val.custo_brl.mean():,.0f} ± {df_val.custo_brl.std():,.0f} · "
          f"R@10% {df_val.recall_at_10.mean():.3f}")
    print(f"  {n_par} parâmetros para {len(X)} amostras de treino "
          f"({len(X) / n_par:.1f} amostras por parâmetro)")

    df_val_1se = None
    if degenerada:
        df_val_1se = fase_validacao(X, y, X_val, y_val, params_1se)
        print(f"  {'MLP 1-SE ()':<12} PR-AUC "
              f"{df_val_1se.pr_auc.mean():.4f} ± {df_val_1se.pr_auc.std():.4f} · "
              f"Brier {df_val_1se.brier.mean():.4f} · "
              f"custo R$ {df_val_1se.custo_brl.mean():,.0f} ± "
              f"{df_val_1se.custo_brl.std():,.0f} · "
              f"R@10% {df_val_1se.recall_at_10.mean():.3f}")

    # Os dois adversários, sob o mesmo tratamento e na mesma tela.
    campeao = construir_pipeline(
        LogisticRegression(max_iter=1000, random_state=config.SEED),
        escalonar=True, encoding="onehot",
    ).fit(X, y)
    hgb = construir_pipeline(
        HistGradientBoostingClassifier(
            random_state=config.SEED, max_iter=1000, early_stopping=True,
            n_iter_no_change=20, validation_fraction=0.1,
            scoring="average_precision",
            categorical_features=mascara_categorica(), **HGB_TUNADO,
        ),
        escalonar=False, encoding="ordinal",
    ).fit(X, y)

    adversarios = {}
    for nome, pipe in (("CAMPEÃO", campeao), ("HGB tunado", hgb)):
        p = pipe.predict_proba(X_val)[:, 1]
        custo = evaluate.curva_custo(y_val, p)
        m = evaluate.avaliar(y_val, p, limiar=custo["limiar_otimo"])
        adversarios[nome] = {"pr_auc": m["pr_auc"], "brier": m["brier"],
                             "custo": custo["custo_otimo_brl"],
                             "recall_at_10": m["recall_at_10"]}
        print(f"  {nome:<12} PR-AUC {m['pr_auc']:.4f} · Brier {m['brier']:.4f} · "
              f"limiar {custo['limiar_otimo']:.2f} · "
              f"custo R$ {custo['custo_otimo_brl']:,.0f} · "
              f"R@10% {m['recall_at_10']:.3f}")

    # ⚠️ A leitura que a tabela acima esconde se ninguém a escrever: PR-AUC e
    # custo em reais podem apontar para lados DIFERENTES. A métrica agregada
    # integra sobre todos os limiares; o custo mede um ponto só — o de operação.
    delta_prauc = df_val.pr_auc.mean() - adversarios["CAMPEÃO"]["pr_auc"]
    delta_custo = df_val.custo_brl.mean() - adversarios["CAMPEÃO"]["custo"]
    print(f"\n  Rede − campeão: PR-AUC {delta_prauc:+.4f} · "
          f"custo R$ {delta_custo:+,.0f} por ciclo")
    print(f"  A dispersão do custo ENTRE AS SEEDS da própria rede é "
          f"R$ {df_val.custo_brl.std():,.0f} "
          f"({'maior' if df_val.custo_brl.std() > abs(delta_custo) else 'menor'} "
          f"que essa diferença).")

    print("\n  Variância entre seeds na predição (dados fixos, só a seed muda):")
    print(f"    desvio da probabilidade prevista por cliente: "
          f"{df_val.attrs['desvio_predicao']:.4f}")
    print("    referência da Etapa 6 (reamostragem do treino): "
          "árvore ±0,2462 · floresta ±0,0438")

    # === 8. a figura =======================================================
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    fig_caminho = config.PROCESSED / "etapa8_curva_treino.png"
    curva = fase_curva(X, y, params_rede, fig_caminho)
    print(f"\n  Curva de treino salva em {fig_caminho.name} "
          f"({curva['n_epocas']} épocas, melhor na {curva['melhor_epoca']})")

    # === 9. registro =======================================================
    df = pd.DataFrame([{
        "modelo": "mlp_torch",
        "params_rede": json.dumps(params_rede, default=str),
        "params_1se": json.dumps(params_1se, default=str),
        "cv_pico": float(medias[i_pico]), "cv_1se": float(medias[i_1se]),
        "cv_rede": float(medias[i_rede]),
        "cv_media_seeds": float(df_seeds.pr_auc.mean()),
        "dp_entre_seeds": dp_seeds, "dp_entre_folds": dp_folds_medio,
        "cv_hidden_vazio": float(medias[i_zero]), "cv_logreg": pr_logreg,
        "cv_logreg_85pct": pr_orcamento,
        "val_pr_auc": float(df_val.pr_auc.mean()),
        "val_pr_auc_dp": float(df_val.pr_auc.std()),
        "val_brier": float(df_val.brier.mean()),
        "val_custo_brl": float(df_val.custo_brl.mean()),
        "val_custo_dp": float(df_val.custo_brl.std()),
        "val_recall_at_10": float(df_val.recall_at_10.mean()),
        "desvio_predicao_seeds": df_val.attrs["desvio_predicao"],
        "com_features_etapa4": feat["pr_auc"],
        "n_parametros": n_par,
        "campeao_val_pr_auc": adversarios["CAMPEÃO"]["pr_auc"],
        "hgb_val_pr_auc": adversarios["HGB tunado"]["pr_auc"],
        "segundos": res["segundos"],
    }])

    mlflow.set_tracking_uri(f"sqlite:///{config.RAIZ / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="etapa8-mlp"):
        mlflow.set_tags({
            "dataset_sha256": dados.sha256, "commit": commit_hash(),
            "etapa": "8-mlp", "avaliado_em": "cv-treino+validacao",
            "framework": f"pytorch {torch.__version__}",
            "regra_selecao": "1-SE (Hastie et al., 2009)",
        })
        mlflow.log_params({
            "n_folds": N_FOLDS, "n_repeticoes": reps, "seeds": str(SEEDS),
            "arquitetura": str(params_rede["hidden"]),
            "weight_decay": params_rede["weight_decay"],
            "otimizador": "Adam(lr=1e-3)", "perda": "BCEWithLogitsLoss",
            "early_stopping": "PR-AUC em validação interna (15% do treino)",
        })
        for coluna in ("cv_1se", "cv_rede", "cv_media_seeds", "dp_entre_seeds",
                       "dp_entre_folds", "cv_hidden_vazio", "cv_logreg",
                       "cv_logreg_85pct", "val_pr_auc", "val_brier", "val_custo_brl",
                       "desvio_predicao_seeds", "com_features_etapa4"):
            mlflow.log_metric(coluna, float(df[coluna].iloc[0]))
        for k, v in par.items():
            mlflow.log_metric(f"pareado_{k}", float(v))
        caminho = config.PROCESSED / "etapa8_mlp.csv"
        df.to_csv(caminho, index=False)
        df_seeds.drop(columns=["scores"]).to_csv(
            config.PROCESSED / "etapa8_seeds.csv", index=False)
        mlflow.log_artifact(str(caminho))
        mlflow.log_artifact(str(fig_caminho))

    print(f"\nSalvo em {caminho} e registrado no MLflow.")


if __name__ == "__main__":
    main()

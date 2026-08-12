"""
Testes da Etapa 8 — MLP em PyTorch.

O que precisa de teste aqui não é a rede aprender (isso a CV mede): é o
**contrato do experimento**. Três afirmações do decision log só valem se o
código as cumprir, e nenhuma delas falha com exceção quando quebra:

1. o MLP de zero camadas ocultas é literalmente a regressão logística — se a
   arquitetura ganhar uma não-linearidade escondida, o controle deixa de
   controlar e ninguém percebe;
2. o treino para por **PR-AUC**, não por acurácia nem pela loss — foi assim que
   o `MLPClassifier` do sklearn virou armadilha de catálogo;
3. o modelo devolvido é o da **melhor** época, não o da última — parar por
   paciência e ficar com o estado final entrega um modelo pior do que o próprio
   critério de parada elegeu, e a métrica cai sem explicação visível.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn

from src import config, gate
from src.mlp import LogRegMesmoOrcamento, MLPTorch, complexidade_mlp, grade_mlp


@pytest.fixture(scope="module")
def dados_sinteticos():
    """Dado tabular pequeno com sinal linear — rápido e suficiente.

    Não usa o Telco de propósito: estes testes travam o comportamento do
    estimador, e carregar o Excel a cada um trocaria segundos por nada.
    """
    rng = np.random.default_rng(config.SEED)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    return X.astype(np.float64), y


def test_hidden_vazio_e_uma_unica_camada_linear(dados_sinteticos):
    """Zero camadas ocultas = um `Linear(n, 1)` = a regressão logística.

    É a base do experimento controlado da etapa: entre a LogReg e o MLP muda
    **só a profundidade**. Se alguém acrescentar uma ativação "inofensiva" no
    fim, o controle passa a comparar duas coisas diferentes e a conclusão sobre
    a não-linearidade do Telco perde o chão.
    """
    X, y = dados_sinteticos
    m = MLPTorch(hidden=(), max_epocas=5).fit(X, y)
    camadas = list(m.rede_)
    assert len(camadas) == 1
    assert isinstance(camadas[0], nn.Linear)
    # sem sigmoide no fim: a saída é LOGIT, porque a perda é BCEWithLogitsLoss.
    # Fundir sigmoide e log dentro da perda é o que evita a saturação em float32;
    # Sigmoid + BCELoss é numericamente instável e é o erro que o material sugere.
    assert not any(isinstance(c, nn.Sigmoid) for c in camadas)


def test_arquitetura_com_camada_oculta_tem_relu(dados_sinteticos):
    """Com camada oculta, tem de haver não-linearidade — senão a rede colapsa.

    Empilhar dois `Linear` sem ativação no meio é matematicamente equivalente a
    um único `Linear`: a "rede profunda" seria a mesma regressão logística com
    mais parâmetros, e a etapa mediria zero por construção em vez de por achado.
    """
    X, y = dados_sinteticos
    m = MLPTorch(hidden=(4,), max_epocas=5).fit(X, y)
    assert any(isinstance(c, nn.ReLU) for c in m.rede_)


def test_mesma_seed_reproduz_a_mesma_predicao(dados_sinteticos):
    """Sem isto, nenhum número desta etapa é reproduzível.

    A seed cobre as duas fontes de aleatoriedade do treino — inicialização dos
    pesos e ordem dos minibatches — mais o split da validação interna.
    """
    X, y = dados_sinteticos
    a = MLPTorch(hidden=(4,), max_epocas=20, seed=42).fit(X, y).predict_proba(X)
    b = MLPTorch(hidden=(4,), max_epocas=20, seed=42).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)


def test_seeds_diferentes_produzem_modelos_diferentes(dados_sinteticos):
    """A outra metade: se a seed não mudasse nada, medir 5 delas seria teatro.

    É esta variação que a Etapa 8 quantifica (item 36 do revisita) e que a CV
    repetida, por reamostrar apenas os dados, é incapaz de enxergar.
    """
    X, y = dados_sinteticos
    a = MLPTorch(hidden=(4,), max_epocas=20, seed=42).fit(X, y).predict_proba(X)
    b = MLPTorch(hidden=(4,), max_epocas=20, seed=7).fit(X, y).predict_proba(X)
    assert not np.allclose(a, b)


def test_early_stopping_para_por_pr_auc_e_nao_por_acuracia(dados_sinteticos):
    """A época escolhida é o argmax do PR-AUC no histórico, e nada mais.

    O `MLPClassifier(early_stopping=True)` do sklearn pontua com
    `accuracy_score` — está no fonte e não é configurável. Num problema de
    prevalência 26,5% com custo 3:1, isso é otimizar a métrica errada DENTRO do
    laço de treino, onde não há log nem warning para denunciar. Este teste é o
    que impede a armadilha de voltar por descuido.
    """
    X, y = dados_sinteticos
    m = MLPTorch(hidden=(4,), max_epocas=40, paciencia=40).fit(X, y)
    assert "pr_auc_val" in m.historico_.columns
    melhor_por_prauc = int(m.historico_.loc[m.historico_.pr_auc_val.idxmax(), "epoca"])
    assert m.melhor_epoca_ == melhor_por_prauc


def test_pesos_restaurados_sao_os_da_melhor_epoca(dados_sinteticos):
    """O modelo devolvido tem de pontuar como a melhor época, não como a última.

    Reconstrói o mesmo split interno (determinístico dada a seed) e recalcula o
    PR-AUC do modelo já treinado: tem de bater com o melhor score do histórico.
    Sem a restauração, este número seria o da última época — pior por definição,
    e sem nenhum sintoma que aponte para a causa.
    """
    X, y = dados_sinteticos
    m = MLPTorch(hidden=(4,), max_epocas=60, paciencia=10, seed=42).fit(X, y)
    _, X_va, _, y_va = train_test_split(
        X, y, test_size=m.frac_val_interna, stratify=y, random_state=42,
    )
    recalculado = average_precision_score(y_va, m.predict_proba(X_va)[:, 1])
    assert np.isclose(recalculado, m.melhor_score_interno_, atol=1e-6)
    assert m.melhor_score_interno_ >= m.historico_.pr_auc_val.iloc[-1]


def test_predict_proba_e_probabilidade_valida(dados_sinteticos):
    X, y = dados_sinteticos
    p = MLPTorch(hidden=(4,), max_epocas=10).fit(X, y).predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((p >= 0) & (p <= 1)).all()


def test_compativel_com_clone_do_sklearn():
    """`clone` é o que `GridSearchCV` e `cross_validate` chamam a cada fit.

    Se o `__init__` fizesse qualquer coisa além de atribuir parâmetros, o clone
    perderia configuração silenciosamente — e cada dobra treinaria um modelo
    ligeiramente diferente do declarado.
    """
    m = MLPTorch(hidden=(16, 8), weight_decay=0.01, seed=7)
    c = clone(m)
    assert c.get_params() == m.get_params()


def test_controle_de_orcamento_treina_nos_mesmos_85_por_cento(dados_sinteticos):
    """O controle tem de reproduzir EXATAMENTE a LogReg no mesmo subconjunto.

    Ele existe para separar dois efeitos que a comparação direta confunde: o
    otimizador (Adam + parada por PR-AUC × LBFGS) e o orçamento de dados, já que
    o early stopping do MLP consome 15% do treino numa validação interna. Se ele
    não replicar o split, a decomposição do decision log fica errada e ninguém
    tem como perceber.
    """
    X, y = dados_sinteticos
    controle = LogRegMesmoOrcamento(seed=42).fit(X, y)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42,
    )
    esperado = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
    assert np.allclose(controle.predict_proba(X), esperado.predict_proba(X))
    # e tem de diferir da LogReg treinada em 100% — senão não controla nada
    cheia = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
    assert not np.allclose(controle.predict_proba(X), cheia.predict_proba(X))


def test_ordem_de_complexidade_do_mlp():
    """Profundidade > largura > penalização, e menor é mais simples.

    É a ordenação que a regra 1-SE usa para decidir. Sem ela declarada, a regra
    não tem como escolher "o mais simples dentro do envelope" — e foi essa ordem
    que fez a busca descer até profundidade zero neste dataset.
    """
    p = lambda h, w: {"modelo__hidden": h, "modelo__weight_decay": w}  # noqa: E731
    assert complexidade_mlp(p((), 0.0)) < complexidade_mlp(p((8,), 0.0))
    assert complexidade_mlp(p((8,), 0.0)) < complexidade_mlp(p((32,), 0.0))
    assert complexidade_mlp(p((32,), 0.0)) < complexidade_mlp(p((16, 16), 0.0))
    # mesma arquitetura: mais penalização é MENOS complexidade
    assert complexidade_mlp(p((8,), 0.01)) < complexidade_mlp(p((8,), 0.0))


def test_grade_contem_o_controle_de_profundidade_zero():
    """`hidden=()` tem de estar na grade — é o controle do experimento.

    Sem ele, a comparação MLP × LogReg misturaria mudança de profundidade com
    mudança de biblioteca e de otimizador, e o delta deixaria de medir
    não-linearidade.
    """
    assert () in grade_mlp()["modelo__hidden"]


def test_gate_exige_os_dois_eixos():
    """Desempenho E calibração, não um dos dois (item 41 do revisita).

    Um gate de eixo único aprova o modelo que sobe a PR-AUC e piora o Brier — e
    a Etapa 6 mediu que isso é possível. O estrago é silencioso: probabilidade
    deslocada => limiar corta a fila no lugar errado => mais custo em reais, com
    o CI verde.
    """
    bom_prauc = config.GATE_PR_AUC_MIN + 0.01
    bom_brier = config.GATE_BRIER_MAX - 0.01
    assert gate.aprovado(bom_prauc, bom_brier)[0]
    assert not gate.aprovado(config.GATE_PR_AUC_MIN - 0.01, bom_brier)[0]
    # o caso que só o segundo eixo pega: ordena bem, calibra mal
    passou, motivo = gate.aprovado(bom_prauc, config.GATE_BRIER_MAX + 0.01)
    assert not passou
    assert "Brier" in motivo

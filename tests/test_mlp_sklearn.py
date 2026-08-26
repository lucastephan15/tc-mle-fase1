"""
Testes da Etapa 8-bis — o `MLPClassifier` exigido pelo enunciado.

O que precisa de teste aqui não é a rede aprender — a CV mede isso. É o
**contrato do experimento**, que sustenta duas afirmações da documentação e
nenhuma das duas falha com exceção quando quebra:

1. o `MLPClassifier` roda no MESMO pipeline dos demais candidatos (one-hot +
   padronização). Se alguém trocar o encoding "para a rede ficar melhor", a
   comparação com a LogReg deixa de ser controlada e a tabela passa a comparar
   pré-processamento em vez de profundidade — sem nada quebrar;
2. a classe **não expõe `scoring`**, que é a justificativa inteira de a Etapa 8
   ter sido escrita em PyTorch. É uma afirmação sobre uma biblioteca de
   terceiro, e afirmações sobre terceiros envelhecem: se uma versão futura do
   sklearn passar a aceitar `scoring`, este teste falha — e falhar é o
   comportamento certo, porque a documentação teria de mudar junto.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from src import config
from src.mlp_sklearn import complexidade, grade, pipe


def test_mesmo_preprocessamento_da_logreg():
    """One-hot + padronização — a condição de validade da comparação.

    Encoding ordinal aqui seria o mesmo erro grosseiro que num modelo linear: a
    rede leria `Contract=2` como o dobro de `Contract=1`. E sem padronização o
    gradiente não converge bem com colunas em escalas díspares (`Total Charges`
    até 8.684 ao lado de dummies 0/1).
    """
    from sklearn.linear_model import LogisticRegression

    from src.preprocess import construir_pipeline

    def assinatura(p):
        ct = p.named_steps["preproc"]
        return [(nome, type(t).__name__, list(cols)) for nome, t, cols in ct.transformers]

    logreg = construir_pipeline(
        LogisticRegression(max_iter=1000, random_state=config.SEED),
        escalonar=True, encoding="onehot",
    )
    assert assinatura(pipe()) == assinatura(logreg)


def test_early_stopping_desligado_por_padrao():
    """O default DESTE módulo é o contrário do reflexo comum, de propósito.

    Ligar `early_stopping` troca o critério de parada por `accuracy_score` sem
    oferecer alternativa. Medido: PR-AUC 0,6846 -> 0,6542 na CV e R$ 30.793 ->
    R$ 32.631 por ciclo na validação. O default do módulo é a configuração
    correta; a errada só aparece no experimento que a mede.
    """
    assert pipe().named_steps["modelo"].early_stopping is False


def test_mlpclassifier_nao_expoe_scoring():
    """A justificativa do PyTorch, escrita como teste.

    `early_stopping=True` pontua a validação interna com `accuracy_score` — está
    no fonte do sklearn e a classe não aceita `scoring`. Num problema de
    prevalência 26,5% com custo 3:1, isso é otimizar a métrica errada dentro do
    laço de treino, onde não há log nem warning que denuncie.

    Se este teste ficar vermelho porque o parâmetro passou a existir, a notícia
    é boa e a documentação está desatualizada: a Etapa 8 poderia ter sido escrita
    em vinte linhas.
    """
    assert "scoring" not in MLPClassifier().get_params()


def test_grade_espelha_a_da_etapa_8():
    """As duas grades são a MESMA lista, ou as tabelas não se comparam.

    O `alpha` do sklearn e o `weight_decay` do Adam são a mesma penalização L2
    sobre os pesos; o que precisa coincidir é o eixo de arquitetura, porque é
    ele que define o que cada rede PODE representar. 1e-4 é o extra: o default
    do sklearn, isto é, a configuração que este experimento teria se ninguém
    tivesse escolhido nada.
    """
    pytest.importorskip("torch")
    from src.mlp import grade_mlp

    assert grade()["modelo__hidden_layer_sizes"] == grade_mlp()["modelo__hidden"]
    assert set(grade()["modelo__alpha"]) == {
        *grade_mlp()["modelo__weight_decay"], 1e-4,
    }


def test_complexidade_ordena_profundidade_antes_de_largura():
    """A ordem da 1-SE: uma camada a mais muda o que a rede pode expressar;
    mais neurônios só refinam a mesma classe de funções."""
    def p(hidden, alpha=0.0):
        return {"modelo__hidden_layer_sizes": hidden, "modelo__alpha": alpha}

    assert complexidade(p(())) < complexidade(p((8,)))
    assert complexidade(p((32,))) < complexidade(p((16, 16)))
    # Mais L2 é MENOS complexidade — por isso o sinal invertido.
    assert complexidade(p((8,), 1e-2)) < complexidade(p((8,), 0.0))


def test_pontua_em_probabilidade(subset):
    """Sanidade: o objeto devolve probabilidade em [0,1], não classe."""
    X, y = subset
    p = pipe(seed=config.SEED, hidden_layer_sizes=(8,)).fit(X, y)
    prob = p.predict_proba(X)[:, 1]
    assert prob.shape == (len(X),)
    assert np.all((prob >= 0) & (prob <= 1))


@pytest.fixture(scope="module")
def subset():
    """300 linhas estratificadas do treino — o bruto é ORDENADO PELO ALVO.

    `head()` traria uma classe só e o erro do sklearn não apontaria a causa.
    """
    from sklearn.model_selection import train_test_split

    from src import data

    d = data.dividir()
    X, _, y, _ = train_test_split(
        d.treino.X, d.treino.y, train_size=300,
        stratify=d.treino.y, random_state=config.SEED,
    )
    return X, y

"""
Testes das features derivadas — Etapa 4.

As features foram **medidas e descartadas** do modelo v1 (ver decision log §3),
mas o código permanece testado e ligável por parâmetro: a Etapa 8 (MLP) vai
reavaliá-las, e um caminho de código sem teste é um caminho quebrado esperando
a vez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config, data, features
from src.preprocess import construir_pipeline


@pytest.fixture(scope="module")
def bruto() -> pd.DataFrame:
    return data.carregar_bruto()


@pytest.fixture(scope="module")
def treino() -> data.Particao:
    """⚠️ Amostras para treinar SEMPRE saem daqui, nunca de `bruto.head(n)`.

    O arquivo está ordenado pelo alvo (ver test_arquivo_bruto_esta_ordenado_pelo_alvo):
    as 1.869 primeiras linhas são todas churners, então head() devolve uma classe
    só e o sklearn levanta 'needs samples of at least 2 classes'. Esta fixture
    existiu porque o erro foi cometido aqui, depois de documentado.
    """
    return data.dividir().treino


TODAS = features.NOVAS_NUM + features.NOVAS_CAT


def test_funcao_e_pura(bruto):
    """Não pode alterar o DataFrame recebido: dentro de um Pipeline a mesma
    entrada é reutilizada por outros passos."""
    entrada = bruto[config.FEATURES].head(50)
    antes = entrada.copy()
    features.adicionar_features(entrada)
    pd.testing.assert_frame_equal(entrada, antes)


def test_contagem_de_servicos_no_intervalo_valido(bruto):
    s = features.adicionar_features(bruto[config.FEATURES])["n_servicos_adicionais"]
    assert s.between(0, len(features.SERVICOS_ADICIONAIS)).all()
    assert (s == 0).any(), "clientes sem nenhum adicional existem e são os de maior risco"


def test_charge_por_servico_nunca_e_infinito(bruto):
    """O +1 no denominador existe para isto: clientes com zero serviços
    adicionais existem, e sem o +1 produziriam divisão por zero — a mesma
    armadilha do inf que a Etapa 1 documentou com 'Tenure Months'."""
    X = features.adicionar_features(bruto[config.FEATURES])
    assert np.isfinite(X["charge_por_servico"]).all()


def test_nenhuma_feature_derivada_gera_nan(bruto):
    X = features.adicionar_features(bruto[config.FEATURES])
    assert not X[TODAS].isna().any().any()


def test_binning_cobre_todos_os_clientes(bruto):
    """Um cliente fora de todas as faixas viraria NaN e seria imputado pela moda,
    silenciosamente."""
    faixa = features.adicionar_features(bruto[config.FEATURES])["tenure_faixa"]
    assert set(faixa.unique()) <= set(features.ROTULOS_TENURE)
    assert faixa.nunique() == len(features.ROTULOS_TENURE)


def test_pipeline_deriva_as_features_a_partir_do_dado_CRU(treino):
    """🔑 O teste que justifica o FunctionTransformer.

    O pipeline recebe apenas as 19 colunas originais — exatamente o que a API vai
    receber como JSON — e deriva as 4 novas por dentro. Se a feature engineering
    vivesse numa célula de notebook, este teste falharia com KeyError, que é o
    aviso que produção não dá: lá o erro sai como predição errada com 200 OK.
    """
    X = treino.X
    assert not set(TODAS) & set(X.columns), "o cru não tem as derivadas"

    pipe = construir_pipeline(
        LogisticRegression(max_iter=1000, random_state=config.SEED), novas=TODAS
    )
    pipe.fit(X.head(500), treino.y.head(500))
    p = pipe.predict_proba(X.head(10))[:, 1]
    assert p.shape == (10,)
    assert np.isfinite(p).all()


def test_desligar_features_nao_muda_o_contrato_de_entrada(treino):
    """Com ou sem as derivadas, a API recebe o MESMO conjunto de colunas. É o que
    permite ligá-las na Etapa 8 sem quebrar nada a jusante."""
    X = treino.X.head(200)
    y = treino.y.head(200)
    for novas in ([], TODAS):
        pipe = construir_pipeline(
            LogisticRegression(max_iter=1000, random_state=config.SEED), novas=novas
        )
        pipe.fit(X, y)
        assert pipe.predict_proba(X)[:, 1].shape == (200,)

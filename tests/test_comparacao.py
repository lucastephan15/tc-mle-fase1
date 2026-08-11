"""
Testes da Etapa 6 — comparação de algoritmos.

Como os demais, cada teste trava uma DECISÃO do decision log. Aqui as decisões
são sobre o encoding alternativo introduzido para as árvores e sobre o gate de
comparabilidade: se o protocolo não for idêntico entre candidatos, a tabela da
Etapa 6 deixa de significar o que ela diz significar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data
from src.comparacao import catalogo
from src.preprocess import (
    construir_pipeline,
    construir_preprocessador,
    mascara_categorica,
)


@pytest.fixture(scope="module")
def bruto() -> pd.DataFrame:
    return data.carregar_bruto()


def test_ordinal_isola_categoria_inedita_em_valor_proprio(bruto):
    """A mesma exigência do teste do one-hot, por outro mecanismo.

    A Etapa 2 decidiu que categoria inédita NÃO pode ser confundida com uma
    categoria conhecida (o erro do `drop_first`, que responde 200 OK com predição
    errada). O encoding ordinal preserva essa garantia via `unknown_value=-1`:
    como as conhecidas recebem 0..k-1, o -1 fica fora do intervalo e a árvore
    consegue isolá-lo num único split.
    """
    pre = construir_preprocessador(escalonar=False, encoding="ordinal")
    pre.fit(bruto[config.FEATURES])
    linha = bruto[config.FEATURES].head(1).copy()
    linha["Gender"] = "Nao-binario"

    cols = list(pre.get_feature_names_out())
    i = cols.index("Gender")
    valor = pre.transform(linha)[0][i]
    assert valor == -1, "categoria inédita deveria virar -1, distinto de toda conhecida"
    conhecidas = pre.transform(bruto[config.FEATURES].head(50))[:, i]
    assert valor not in set(conhecidas)


def test_mascara_categorica_bate_com_a_saida_do_preprocessador(bruto):
    """A máscara passada ao `categorical_features` do boosting é construída a
    partir da ORDEM dos grupos do ColumnTransformer. Se alguém reordenar os
    transformers, o boosting passaria a tratar coluna numérica como categórica
    — silenciosamente, sem erro. Este teste é a trava dessa suposição."""
    pre = construir_preprocessador(escalonar=False, encoding="ordinal")
    pre.fit(bruto[config.FEATURES])
    cols = list(pre.get_feature_names_out())
    mascara = mascara_categorica()

    assert len(mascara) == len(cols)
    marcadas = {c for c, m in zip(cols, mascara, strict=True) if m}
    assert marcadas == set(config.CAT)


def test_ordinal_produz_uma_coluna_por_categorica(bruto):
    """É o que distingue os dois encodings — e a razão de a máscara acima só
    valer para o ordinal."""
    ordinal = construir_preprocessador(escalonar=False, encoding="ordinal")
    onehot = construir_preprocessador(escalonar=False, encoding="onehot")
    ordinal.fit(bruto[config.FEATURES])
    onehot.fit(bruto[config.FEATURES])

    n_ord = len(ordinal.get_feature_names_out())
    n_oh = len(onehot.get_feature_names_out())
    assert n_ord == len(config.FEATURES)
    assert n_oh > n_ord, "one-hot expande cada categórica em k colunas"


def test_encoding_invalido_falha_alto():
    """Erro de digitação em `--encoding` não pode cair silenciosamente no
    default: o resultado da Etapa 6 dependeria de um encoding que ninguém
    escolheu."""
    with pytest.raises(ValueError, match="encoding desconhecido"):
        construir_preprocessador(encoding="onehotordinal")


@pytest.mark.parametrize("nome", ["logreg", "arvore", "rf", "rf_reg", "hgb", "hgb_reg"])
def test_todo_candidato_treina_e_produz_probabilidade_valida(nome):
    """Gate de comparabilidade: um candidato que não devolve `predict_proba`
    contínuo não pode ser comparado por PR-AUC, que é métrica de ORDENAÇÃO."""
    dados = data.dividir()
    X = dados.treino.X.head(300)
    y = dados.treino.y.head(300)
    encoding = "ordinal" if nome != "logreg" else "onehot"
    estimador, escalonar = catalogo(encoding)[nome]
    pipe = construir_pipeline(estimador, escalonar=escalonar, encoding=encoding)
    p = pipe.fit(X, y).predict_proba(dados.validacao.X.head(100))[:, 1]

    assert p.shape == (100,)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_seed_unica_em_todos_os_candidatos():
    """O gate da Etapa 6 exige mesma partição, mesmas features e mesma seed.
    Um `random_state` divergente faria a comparação medir o sorteio."""
    for encoding in ("onehot", "ordinal"):
        for nome, (estimador, _) in catalogo(encoding).items():
            assert estimador.get_params()["random_state"] == config.SEED, nome

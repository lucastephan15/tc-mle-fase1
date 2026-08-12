"""
Testes da Etapa 7 — tuning de hiperparâmetros.

A lógica nova desta etapa não está no `GridSearchCV` (esse é do sklearn e já é
testado): está na REGRA DE DECISÃO que substitui o `argmax`. É ela que precisa
de teste, porque é ela que escolhe o modelo — e um erro aqui não produz exceção,
produz o modelo errado escolhido em silêncio.

Foi exatamente o que aconteceu na primeira execução: `np.argmin` sobre uma lista
de TUPLAS achata o array e devolve um índice do array plano. Naquele caso o
índice estourou e o erro apareceu; numa grade maior ele teria simplesmente
apontado para outro candidato, sem nenhum sintoma. O teste
`test_1se_escolhe_o_mais_simples_dentro_do_envelope` trava esse caso.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.tuning import (
    avisos_de_borda,
    comparar_pareado,
    complexidade_hgb,
    complexidade_logreg,
    envelopes,
    grade_logreg,
    refit_1se,
)


def _cv_results(medias, dps, params):
    return {"mean_test_score": np.array(medias),
            "std_test_score": np.array(dps),
            "params": params}


def test_1se_escolhe_o_mais_simples_dentro_do_envelope():
    """A regra tem de trocar o pico por um modelo mais regularizado.

    Cenário montado à mão: o melhor score (0,70) tem `C=100`, mas há um `C=0,01`
    a 0,695 — dentro de 1 erro padrão. A resposta certa é o `C` MENOR, porque em
    LogReg menor `C` é mais penalização (`C = 1/lambda`, escala invertida). Um
    teste que aceitasse o pico passaria com o `argmax`, que é justamente o que
    esta etapa decidiu não usar.
    """
    params = [
        {"modelo__C": 100.0, "modelo__l1_ratio": 0.0},
        {"modelo__C": 0.01, "modelo__l1_ratio": 0.0},
        {"modelo__C": 1.0, "modelo__l1_ratio": 0.0},
    ]
    r = _cv_results([0.700, 0.695, 0.698], [0.02, 0.02, 0.02], params)
    i = refit_1se(complexidade_logreg, n_dobras=15)(r)
    assert params[i]["modelo__C"] == 0.01


def test_1se_nao_aceita_candidato_fora_do_envelope():
    """Simples demais e ruim demais não passa: o envelope é o limite.

    Sem esta metade, a regra degeneraria em "escolha sempre o mais regularizado",
    que é um modelo constante — e um modelo constante tem PR-AUC igual à
    prevalência, o piso que a Etapa 0 registrou.
    """
    params = [{"modelo__C": 100.0, "modelo__l1_ratio": 0.0},
              {"modelo__C": 0.001, "modelo__l1_ratio": 0.0}]
    # o segundo está 0,10 abaixo — muito além de 1 erro padrão (0,02/√15)
    r = _cv_results([0.700, 0.600], [0.02, 0.02], params)
    i = refit_1se(complexidade_logreg, n_dobras=15)(r)
    assert params[i]["modelo__C"] == 100.0


def test_1se_desempata_pela_esparsidade():
    """Mesmo `C`, a L1 ganha: ela zera coeficientes, logo usa menos parâmetros."""
    params = [{"modelo__C": 0.1, "modelo__l1_ratio": 0.0},
              {"modelo__C": 0.1, "modelo__l1_ratio": 1.0}]
    r = _cv_results([0.700, 0.699], [0.02, 0.02], params)
    i = refit_1se(complexidade_logreg, n_dobras=15)(r)
    assert params[i]["modelo__l1_ratio"] == 1.0


def test_complexidade_hgb_ordena_por_capacidade():
    """Menos folhas é mais simples; com folhas iguais, taxa menor é mais simples.

    A ordem é lexicográfica de propósito: `max_leaf_nodes` restringe a FORMA que
    cada árvore pode ter, e o `learning_rate` só limita o quanto ela contribui.
    Inverter os dois faria a regra preferir uma árvore grande e lenta a um toco.
    """
    toco = {"modelo__max_leaf_nodes": 2, "modelo__learning_rate": 0.2,
            "modelo__min_samples_leaf": 10, "modelo__l2_regularization": 0.0}
    grande = {"modelo__max_leaf_nodes": 31, "modelo__learning_rate": 0.01,
              "modelo__min_samples_leaf": 320, "modelo__l2_regularization": 10.0}
    assert complexidade_hgb(toco) < complexidade_hgb(grande)


def test_envelope_conservador_e_mais_largo_que_o_do_material():
    """Nadeau-Bengio: `dp/√15` superestima a precisão, `dp/√5` é mais honesto.

    As três leituras têm de ficar em ordem — se o envelope "conservador" fosse
    mais estreito que o do material, o argumento de robustez da etapa estaria
    invertido e ninguém perceberia olhando só o número final.
    """
    e = envelopes(np.array([0.70]), np.array([0.03]), 0, n_dobras=15)
    assert e["se"] > e["conservador"] > e["desvio_cheio"]


def test_aviso_de_borda_ignora_fronteira_natural_e_usa_a_grade_declarada():
    """Duas armadilhas de uma vez, ambas cometidas na primeira versão.

    (a) `l1_ratio=1.0` é a L1 pura — não existe "mais L1", então avisar que o
    ótimo pode estar fora da grade é ruído, e alerta sem ação possível é o começo
    da fadiga de alerta.
    (b) A conferência é contra a GRADE, não contra os valores sorteados: num
    `RandomizedSearchCV`, um valor da lista pode nunca ter sido visitado, e
    comparar com o sorteado inventaria uma borda que não existe.
    """
    grade = {"modelo__max_leaf_nodes": [2, 4, 8, 16],
             "modelo__l1_ratio": [0.0, 0.5, 1.0]}
    assert avisos_de_borda(grade, {"modelo__max_leaf_nodes": 8,
                                   "modelo__l1_ratio": 1.0}) == []
    avisos = avisos_de_borda(grade, {"modelo__max_leaf_nodes": 2,
                                     "modelo__l1_ratio": 0.5})
    assert len(avisos) == 1 and "max_leaf_nodes" in avisos[0]


def test_aviso_de_borda_entende_distribuicao_continua():
    """Eixo amostrado por `loguniform` não tem lista: o critério vira a faixa."""
    grade = {"modelo__learning_rate": stats.loguniform(0.01, 0.2)}
    assert avisos_de_borda(grade, {"modelo__learning_rate": 0.05}) == []
    assert avisos_de_borda(grade, {"modelo__learning_rate": 0.011}) != []


def test_grade_logreg_contem_o_default_da_biblioteca():
    """`C=1.0` + L2 pura têm de estar na grade — é o modelo que já está no repo.

    Sem esse ponto, a pergunta "o tuning ganhou alguma coisa?" não teria contra o
    quê ser medida, e a etapa reportaria um número absoluto em vez de um delta.
    """
    g = grade_logreg()
    assert 1.0 in g["modelo__C"]
    assert 0.0 in g["modelo__l1_ratio"]


def test_comparacao_pareada_usa_as_dobras_na_ordem():
    """O pareamento é a razão de guardar os 15 scores (item 45 do revisita).

    Um modelo uniformemente melhor por uma margem pequena tem de aparecer como
    vitória em todas as dobras — é isso que o teste pareado enxerga e o teste
    independente não enxergaria, porque a variação ENTRE dobras (0,06 de
    amplitude) é várias vezes a diferença entre modelos (~0,01).

    A margem varia um pouco de dobra para dobra de propósito: diferença
    exatamente constante tem variância zero e faz o `ttest_rel` avisar sobre
    perda de precisão — situação que não existe em dado real e que produziria um
    teste verde por um caminho que os dados nunca percorrem.
    """
    a = np.array([0.70, 0.66, 0.72, 0.68, 0.69])
    b = a - np.array([0.010, 0.008, 0.013, 0.009, 0.011])
    r = comparar_pareado(a, b)
    assert r["vitorias"] == 5
    assert np.isclose(r["delta_medio"], 0.0102)
    assert r["p_wilcoxon"] < 0.10

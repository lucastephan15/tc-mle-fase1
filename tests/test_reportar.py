"""
Testes da leitura única do conjunto de teste — Etapa 11.

O teste que importa aqui não é sobre aritmética: é o que amarra **o número publicado**
ao **artefato servido**. Um relatório que descreve um modelo diferente do que está em
produção é a versão documental do oitavo servidor da Knight Capital — ninguém percebe,
porque nada falha.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import artefato, config, reportar


@pytest.fixture(scope="module")
def registro() -> dict:
    if not reportar.REGISTRO.exists():
        pytest.skip("o teste final ainda não foi tocado (rode: make reportar)")
    return json.loads(reportar.REGISTRO.read_text())


def test_registro_descreve_o_artefato_QUE_ESTA_PROMOVIDO(registro):
    """O número da documentação tem de ser o número do modelo servido.

    Se alguém promover outro artefato e não re-reportar, o relatório continua verde e
    passa a descrever um objeto que não existe mais. É o mesmo argumento do teste de
    caracterização, um andar acima: lá se protege o modelo, aqui se protege a afirmação
    pública sobre ele.
    """
    sha_atual = artefato.sha256_arquivo(config.ARTEFATO)
    assert registro["artefato_sha256"] == sha_atual, (
        "o resultado publicado foi medido sobre OUTRO artefato. Promover um modelo novo "
        "exige re-tocar o teste (`make reportar --reexecutar` primeiro, para saber o que "
        "mudou) e substituir o registro NUM COMMIT PRÓPRIO."
    )


def test_registro_usa_o_limiar_de_OPERACAO_e_nao_0_5(registro):
    """A mesma armadilha da auditoria de fairness: 0,5 descreve uma fila que ninguém corta."""
    a = artefato.carregar()
    assert registro["limiar_operacao"] == pytest.approx(a.limiar)
    assert registro["limiar_operacao"] != 0.5


def test_metrica_vem_com_o_piso_ao_lado(registro):
    """PR-AUC sem prevalência é métrica sem piso — não é medida, é opinião."""
    assert registro["prevalencia_teste"] == pytest.approx(0.2654, abs=1e-3)
    assert registro["teste"]["pr_auc"] > registro["prevalencia_teste"]


def test_custos_triviais_sao_as_duas_estrategias_sem_modelo():
    """Aritmética verificável: 3 churners × R$194 e 2 não-churners × R$62."""
    y = np.array([1, 1, 1, 0, 0])
    c = reportar.custos_triviais(y)
    assert c["nao_fazer_nada_brl"] == 3 * 194
    assert c["abordar_todos_brl"] == 2 * 62


def test_conferir_acusa_artefato_diferente():
    """A re-execução compara identidade, não só números — modelo trocado é notícia."""
    base = {"teste": {"pr_auc": 0.6496, "brier": 0.1352,
                      "recall_at_10": 0.286, "custo_erro_brl": 32882.0},
            "artefato_sha256": "a" * 64}
    igual = {**base, "artefato_sha256": "a" * 64}
    outro = {**base, "artefato_sha256": "b" * 64}
    assert reportar.conferir(igual, base) == []
    assert any("OUTRO modelo" in p for p in reportar.conferir(outro, base))


def test_conferir_acusa_metrica_deslocada():
    base = {"teste": {"pr_auc": 0.6496, "brier": 0.1352,
                      "recall_at_10": 0.286, "custo_erro_brl": 32882.0},
            "artefato_sha256": "a" * 64}
    novo = {**base, "teste": {**base["teste"], "pr_auc": 0.6500}}
    assert any("pr_auc" in p for p in reportar.conferir(novo, base))

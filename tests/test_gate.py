"""
Testes do gate de promoção — Etapa 9.5.

DOIS controles com perguntas diferentes, e é a diferença que importa:

- `aprovado()` protege contra **degradação deliberada** — um modelo novo que é
  pior que o compromisso assumido com o negócio. É um PISO.
- `test_caracterizacao_do_campeao` protege contra **mudança acidental** — uma
  refatoração que não devia mexer em nada e mexeu. É um CONTRATO.

Um piso mede se o modelo é bom o bastante; um contrato mede se ele é o mesmo.
Nenhum dos dois substitui o outro: medido em 13/08/2026, com o piso sozinho é
possível remover 5 das 13 features (Total Charges, Monthly Charges, Gender,
Senior Citizen, Multiple Lines) e ficar com o CI inteiro VERDE — a PR-AUC vai
a 0,6660, acima do piso de 0,66. Nove das treze somem uma a uma sem acusar,
incluindo 'Contract'. A folga do piso (0,0046) é da ordem da distância entre o
campeão e os seis candidatos que ele derrotou na Etapa 6 (0,0057), ou seja: da
zona inteira onde a escolha do campeão foi decidida.
"""

from __future__ import annotations

import pytest

from src import config, data, gate

# --- O contrato do campeão -------------------------------------------------
# Medidos no run que elegeu o campeão (LogReg, 13 features, validação) e
# confirmados bit a bit: três execuções do zero dão o mesmo sha256 das 1.409
# probabilidades e o mesmo repr 0.6646020518504109.
#
# ⛔ MUDAR ESTES NÚMEROS É UM COMMIT PRÓPRIO, com justificativa — mesma regra do
# piso do gate. Se um deles mudou sem que a intenção fosse mudar o modelo, o
# teste está fazendo exatamente o que deveria: pergunte o que mudou antes de
# atualizar a referência.
PR_AUC_REF = 0.6646
BRIER_REF = 0.1339

# Frouxa o bastante para absorver troca de versão de biblioteca, apertada o
# bastante para reprovar qualquer mudança real: a menor alteração que testei
# (remover 'Contract') desloca a PR-AUC em 0,0019, dezenove vezes a tolerância.
TOL = 1e-4


@pytest.fixture(scope="module")
def dados() -> data.Dados:
    return data.dividir()


def test_caracterizacao_do_campeao(dados):
    """O campeão continua sendo o MESMO modelo, não apenas um modelo bom.

    Verificado reprovando antes de entrar (disciplina do item 79): com 5
    features removidas dá 0,6660, com 'Contract' fora dá 0,6627 e sem
    escalonamento dá 0,6663 — as três passam no gate e as três são pegas aqui.

    Note que DUAS delas AUMENTAM a PR-AUC. Um controle que só olha para baixo
    não é contrato, é piso com outro nome: a pergunta aqui não é "ficou melhor?",
    é "mudou?".
    """
    m, _ = gate.medir(dados)
    assert m["pr_auc"] == pytest.approx(PR_AUC_REF, abs=TOL)
    assert m["brier"] == pytest.approx(BRIER_REF, abs=TOL)


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


def test_campeao_passa_no_proprio_gate(dados):
    """Amarra os dois controles: o valor caracterizado é aprovado pela regra.

    Sem isto, PR_AUC_REF e GATE_PR_AUC_MIN poderiam derivar em direções opostas
    sem nada acusar — o contrato afirmando um número que o piso rejeita.
    """
    m, _ = gate.medir(dados)
    passou, motivo = gate.aprovado(m["pr_auc"], m["brier"])
    assert passou, motivo

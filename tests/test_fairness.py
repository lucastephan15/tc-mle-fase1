"""
Testes da auditoria de fairness — Etapa 10.5.

🔑 **Por que estes testes caracterizam em vez de barrar um alvo.** A política
pré-registrada (disparidade <= 10 pp) **não é atingida** por 3 dos 4 atributos, e
a decisão registrada no decision log §5g é *aceitar com declaração*. Um teste que
exigisse os 10 pp reprovaria o modelo em produção a cada push — e a saída óbvia
seria afrouxar o número até o verde, que é a negociação *post hoc* que o
pré-registro existe para impedir.

⇒ A divisão é a mesma que o repo já usa para desempenho: o **piso** pergunta
*"é bom o bastante?"* e o **contrato** pergunta *"é o mesmo?"*. Aqui o "bom o
bastante" foi conscientemente não atingido, e está escrito no Model Card com o
motivo. O que o CI protege é que esses números não mudem sem alguém ver — para
melhor **ou** para pior, porque uma disparidade que cai também é notícia, e
porque um modelo pior tende a parecer mais justo (no limite, um modelo aleatório
tem disparidade zero).

⚠️ Isto é uma limitação assumida, não uma vitória: enquanto a decisão for
"aceitar", nenhum teste aqui impede o modelo de continuar a 58,89 pp. É o Model
Card que carrega o compromisso, e é por isso que ele traz o número na primeira
tela em vez de num apêndice.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, data, evaluate, fairness, gate

# Medidos em 19/08/2026, campeão LogReg, validação, limiar de operação 0,29.
# Tolerância de 1e-4 pelo mesmo motivo do teste de caracterização do gate: o
# pipeline é determinístico (mesma seed, mesma partição), então a folga existe
# para troca de biblioteca, não para ruído.
DISPARIDADE_REF = {
    "Gender": 0.0816,
    "Senior Citizen": 0.1272,
    "Partner": 0.1003,
    "Dependents": 0.5889,
}
TOL = 1e-4


@pytest.fixture(scope="module")
def auditoria():
    return fairness.medir_do_campeao()


def test_caracterizacao_da_disparidade(auditoria):
    """Os números do Model Card são estes, e mudá-los exige um diff.

    Verificado reprovando antes de entrar: alterar o limiar da auditoria de 0,29
    para 0,5 move `Dependents` de 0,5889 para 0,7681, e os quatro caem aqui.
    """
    aud, _ = auditoria
    for attr, esperado in DISPARIDADE_REF.items():
        assert aud[attr]["disparidade_recall"] == pytest.approx(esperado, abs=TOL), attr


def test_auditoria_usa_o_limiar_de_OPERACAO(auditoria):
    """Auditar em 0,5 mede um modelo que este projeto não usa.

    É o mesmo defeito de `.predict()` na API, um andar acima: o número sai
    plausível, não falha, e descreve uma fila que ninguém corta.
    """
    _, limiar = auditoria
    assert limiar == pytest.approx(0.29, abs=1e-9)
    assert limiar != 0.5


def test_grupo_pequeno_e_marcado_como_incerto(auditoria):
    """`Dependents=Yes` tem 23 churners: o pior achado é o de menor amostra.

    O teste fixa que o relatório **diz isso**. Um recall de 0,2174 sobre 23
    casos tem a mesma aparência de um sobre 2.300 se ninguém escrever o n — e só
    um dos dois sustenta uma decisão.
    """
    aud, _ = auditoria
    yes = aud["Dependents"]["grupos"]["Yes"]
    assert yes["churners"] == 23
    assert not yes["confiavel"]
    lo, hi = yes["ic95_recall"]
    assert hi - lo > 0.3, "IC estreito demais para 23 casos — a conta está errada"
    # E o que impede o IC de virar desculpa: mesmo no limite superior, o grupo
    # continua muito abaixo do outro. A incerteza não salva o resultado.
    assert hi < aud["Dependents"]["grupos"]["No"]["recall"]


def test_o_grupo_pior_e_identificado_e_nao_e_o_protegido_em_senior(auditoria):
    """Disparidade a FAVOR do grupo protegido é disparidade, e não é o mesmo dano.

    Em `Senior Citizen` quem tem recall menor é o grupo **não** idoso. Reportar
    só o módulo da diferença apagaria isso, e a leitura de negócio inverteria.
    """
    aud, _ = auditoria
    assert aud["Senior Citizen"]["grupo_pior"] == "No"
    assert aud["Dependents"]["grupo_pior"] == "Yes"


def test_unawareness_NAO_resolve_e_ainda_reprova_o_gate():
    """Remover a coluna sensível: medido, não afirmado.

    O catálogo diz *"remover a coluna não remove o viés"*. Medido aqui, o
    resultado é mais matizado e mais útil: **atenua** (58,89 -> 13,20 pp) porque
    o viés sobrevive nos proxies, continua acima do limite, e derruba a PR-AUC
    para 0,6427 — **abaixo do piso de 0,66**, ou seja, o modelo nem seria
    promovível. A saída "mais justa" é reprovada pelo outro eixo do gate.
    """
    np.random.seed(config.SEED)
    d = data.dividir()
    from src.preprocess import construir_pipeline
    from src.train import construir_modelo

    modelo, escalonar = construir_modelo(config.GATE_MODELO_REFERENCIA, None)
    pipe = construir_pipeline(modelo, escalonar=escalonar, excluir=["Dependents"])
    pipe.fit(d.treino.X, d.treino.y)
    p = pipe.predict_proba(d.validacao.X)[:, 1]
    limiar = evaluate.curva_custo(d.validacao.y, p)["limiar_otimo"]
    aud = fairness.auditar(d.validacao.y, p, d.validacao.X[fairness.SENSIVEIS], limiar)

    disp = aud["Dependents"]["disparidade_recall"]
    assert disp < 0.20, "esperava atenuação substancial ao remover a coluna"
    assert disp > fairness.LIMITE_DISPARIDADE, "unawareness não deveria resolver"

    pr_auc = evaluate.avaliar(d.validacao.y, p, limiar=limiar)["pr_auc"]
    assert pr_auc < config.GATE_PR_AUC_MIN, "o achado é que esta saída REPROVA o gate"


def test_atributos_sensiveis_continuam_nas_features():
    """A decisão da Etapa 5 é o que torna esta auditoria possível.

    Se alguém remover as demográficas das features "para não discriminar", este
    teste cai — e a mensagem é o argumento. É a única forma de impedir que o
    reflexo mais comum da área desfaça a decisão registrada.
    """
    for attr in fairness.SENSIVEIS:
        assert attr in config.FEATURES, (
            f"{attr} saiu das features: isso é fairness through unawareness. "
            "Medido nesta etapa: atenua a disparidade, não a elimina, e reprova "
            "o gate de PR-AUC. Ver decision log §5g."
        )


def test_campeao_auditado_e_o_mesmo_do_gate(auditoria):
    """Amarra a auditoria ao objeto que o CI mede e a promoção grava.

    Sem isto, `fairness` poderia caracterizar um terceiro modelo construído com
    o mesmo código — a divergência não quebraria nada e os dois ficariam verdes.
    """
    np.random.seed(config.SEED)
    d = data.dividir()
    m, c = gate.medir(d)
    _, limiar = auditoria
    assert limiar == pytest.approx(c["limiar_otimo"], abs=1e-9)
    assert m["recall"] == pytest.approx(0.7701, abs=1e-4)

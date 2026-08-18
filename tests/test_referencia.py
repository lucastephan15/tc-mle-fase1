"""
Testes do baseline de drift — Etapa 10a-2.

Um baseline não tem métrica de qualidade própria: ele está certo quando o
detector construído sobre ele **dispara no drift e fica quieto no ruído**. Por
isso os testes daqui são pares — controle negativo (o que não pode acusar) e
controle positivo (o que tem de acusar) —, na mesma disciplina do item 79: um
controle que só passa não prova que existe.

🚨 O deslocamento fabricado nos testes é o mesmo do `simulate_drift.py`
(10c-bis). Aqui ele roda em milissegundos e sem rede; lá ele atravessa a API e
os logs. Se o detector quebrar, esta suíte diz **onde**, e não só que o número
final mudou.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import artefato, config, referencia


@pytest.fixture(scope="module")
def ref(dados) -> dict:
    return referencia.calcular(
        dados.treino.X, config.NUM_ZERO + config.NUM, config.CAT,
    )


def test_cobre_exatamente_as_features_servidas(ref):
    """Baseline parcial é pior que baseline ausente: parece completo.

    A coluna sem referência não daria erro na análise — ela simplesmente não
    apareceria na tabela de PSI, e ninguém audita a ausência de uma linha.
    """
    cobertas = set(ref["numericas"]) | set(ref["categoricas"])
    assert cobertas == set(config.FEATURES)
    assert set(ref["numericas"]) == set(config.NUM_ZERO + config.NUM)


def test_bins_sao_validos_e_a_massa_fecha(ref):
    """Bordas estritamente crescentes e proporções somando 1.

    Borda repetida gera bin de largura zero — nunca recebe ninguém, e a
    proporção esperada zerada envenena a razão do PSI. `np.unique` no cálculo é
    o que garante isso; este teste é o que garante o `np.unique`.
    """
    for coluna, r in ref["numericas"].items():
        bordas = r["bordas_psi"]
        assert bordas == sorted(set(bordas)), f"{coluna}: bordas repetidas"
        assert len(r["proporcoes_psi"]) == len(bordas) + 1
        assert sum(r["proporcoes_psi"]) == pytest.approx(1.0, abs=1e-4)

    for coluna, r in ref["categoricas"].items():
        soma = sum(r["frequencias"].values())
        assert soma == pytest.approx(1.0, abs=1e-4), coluna


def test_coluna_com_massa_concentrada_nao_gera_bin_de_largura_zero():
    """O caso que o `np.unique` das bordas protege — e que o dado real NÃO cobre.

    Medido: as 3 numéricas do repo têm 9 decis distintos cada, então remover o
    `np.unique` do cálculo **não quebra nenhum outro teste desta suíte**. Uma
    proteção que o dataset não exercita é uma proteção que ninguém verifica: a
    coluna com 70% de zeros abaixo faz os decis empatarem em 0, e sem a
    deduplicação sairiam bins de largura zero: faixas que nunca recebem ninguém
    nem no treino nem na janela, mas que entram na soma do PSI como termos
    artificiais — ruído com cara de sinal, numa métrica cuja leitura inteira é
    "passou de 0,25?".
    """
    import pandas as pd

    X = pd.DataFrame({"C": [0.0] * 70 + list(range(1, 31))})
    r = referencia.calcular(X, num=["C"], cat=[])["numericas"]["C"]

    assert r["bordas_psi"] == sorted(set(r["bordas_psi"]))
    # Nenhum bin INTERNO vazio: entre duas bordas distintas há massa, e um bin
    # interno com esperado 0 seria o buraco no meio da distribuição.
    assert all(p > 0 for p in r["proporcoes_psi"][1:]), r["proporcoes_psi"]
    assert sum(r["proporcoes_psi"]) == pytest.approx(1.0, abs=1e-4)

    # 🔑 O bin da PONTA ESQUERDA fica em 0 por construção — o mínimo do treino
    # virou borda, e nada no treino é menor que o mínimo. Isso é o comportamento
    # certo, não um furo: o bin existe para receber o que o treino nunca viu, e
    # se a janela o preencher, é drift dos graves. Verificado que o `EPSILON`
    # transforma esse `ln(a/0)` num número **grande e finito** em vez de `inf` —
    # sem ele a feature deixaria de ter número em vez de ter um número alarmante.
    abaixo_do_minimo = referencia._proporcoes(
        np.array([-5.0] * 50 + [0.0] * 50), r["bordas_psi"],
    )
    valor = referencia.psi(r["proporcoes_psi"], abaixo_do_minimo)
    assert np.isfinite(valor) and valor > referencia.PSI_AGIR, valor


def test_valor_fora_do_intervalo_do_treino_nao_some(ref):
    """As pontas dos bins são ABERTAS, e é isso que impede o pior falso negativo.

    Com binning fechado no `min`/`max` do treino, uma janela inteiramente fora
    do intervalo visto seria descartada da contagem — a distribuição mais
    deslocada possível daria o PSI mais tranquilo. Aqui: 500 clientes com
    `Tenure Months` = 500 (impossível na base) têm de cair **todos** no último
    bin, não sumir.
    """
    r = ref["numericas"]["Tenure Months"]
    props = referencia._proporcoes(np.full(500, 500.0), r["bordas_psi"])
    assert sum(props) == pytest.approx(1.0)
    assert props[-1] == pytest.approx(1.0)


def test_controles_do_psi(ref, dados):
    """Negativo: treino × treino = 0. Positivo: mesma população continua estável.

    O segundo é o que dá sentido ao corte de 0,10 — a validação é outra amostra
    da MESMA população, então tudo que o PSI vê ali é ruído de partição. Medido
    em 18/08/2026: máximo 0,0128 (`Tenure Months`), uma ordem de grandeza abaixo
    do limiar. Um detector que acusasse aqui estaria calibrado para alertar toda
    semana e ser ignorado na terceira.
    """
    igual = referencia.comparar(ref, dados.treino.X)
    assert max(v["psi"] for v in igual.values()) == 0.0

    outra_amostra = referencia.comparar(ref, dados.validacao.X)
    pior = max(outra_amostra.values(), key=lambda v: v["psi"])
    assert pior["psi"] < referencia.PSI_ESTAVEL, pior
    assert all(v["classificacao"] == "estavel" for v in outra_amostra.values())


def test_detecta_a_base_envelhecendo(ref, dados):
    """Controle positivo nº 1 — `Tenure Months` +12 meses (covariate shift).

    O cenário do enunciado: a carteira envelhece e ninguém avisa o modelo. Tem
    de cruzar 0,25 ("agir") na coluna deslocada **e** deixar as outras quietas —
    detector que acende tudo junto não localiza nada.
    """
    X = dados.validacao.X.copy()
    X["Tenure Months"] = X["Tenure Months"] + 12

    saida = referencia.comparar(ref, X)
    assert saida["Tenure Months"]["classificacao"] == "agir", saida["Tenure Months"]
    assert saida["Monthly Charges"]["classificacao"] == "estavel"
    assert saida["Contract"]["classificacao"] == "estavel"


def test_detecta_reajuste_de_preco(ref, dados):
    """Controle positivo nº 2 — `Monthly Charges` ×1,15."""
    X = dados.validacao.X.copy()
    X["Monthly Charges"] = X["Monthly Charges"] * 1.15

    saida = referencia.comparar(ref, X)
    assert saida["Monthly Charges"]["psi"] > referencia.PSI_ESTAVEL
    assert saida["Tenure Months"]["classificacao"] == "estavel"


def test_categoria_inedita_e_nomeada_e_faz_o_psi_saltar(ref, dados):
    """Controle positivo nº 3 — um plano novo entrou no catálogo.

    É o drift que as estatísticas contínuas não pegam e o mais fácil de
    acontecer de verdade: o `OneHotEncoder` trata a categoria desconhecida como
    tudo-zero e **prediz mesmo assim**, com HTTP 200. O baseline tem de dizer o
    nome da categoria, não só que algo mudou — "PSI 0,4 em Contract" manda
    alguém investigar; "apareceu `Two year prepaid`" já é a investigação.
    """
    X = dados.validacao.X.copy()
    X.iloc[: len(X) // 3, X.columns.get_loc("Contract")] = "Two year prepaid"

    saida = referencia.comparar(ref, X)
    assert saida["Contract"]["categorias_ineditas"] == ["Two year prepaid"]
    assert saida["Contract"]["classificacao"] == "agir"


def test_janela_sem_a_coluna_e_omitida_nao_zerada(ref, dados):
    """PSI 0 sobre coluna ausente diria "estável" sobre o que não foi medido."""
    X = dados.validacao.X.drop(columns=["Contract"])
    saida = referencia.comparar(ref, X)
    assert "Contract" not in saida
    assert "Tenure Months" in saida


def test_carga_estrita_recusa_artefato_sem_baseline(promovido, tmp_path):
    """Sem a chave, o artefato CARREGA — todos os acessos usam `.get`.

    Ou seja: sem esta checagem, um artefato promovido antes da 10a-2 subiria,
    responderia 200, e a falha só apareceria semanas depois como ausência de
    alerta — que é indistinguível de ausência de problema.

    Verificado reprovando, e verificado que o modo de inspeção (`estrito=False`)
    continua abrindo o artefato antigo: arqueologia de número já reportado não
    pode depender de o artefato ser servível hoje.
    """
    import joblib

    _, caminho = promovido
    payload = joblib.load(caminho)
    del payload["metadados"]["referencia"]
    forjado = tmp_path / "sem_referencia.joblib"
    joblib.dump(payload, forjado)

    with pytest.raises(artefato.ArtefatoIncompativel, match="referência"):
        artefato.carregar(forjado)
    assert artefato.carregar(forjado, estrito=False).versao == config.VERSAO_MODELO


def test_carga_estrita_recusa_baseline_de_outro_conjunto_de_features(promovido, tmp_path):
    """Baseline e contrato de colunas saídos de promoções diferentes.

    O modo de falha que isto barra é mudo: mede-se drift de uma coluna que o
    modelo não usa mais, e não se mede o da coluna que entrou.
    """
    import joblib

    _, caminho = promovido
    payload = joblib.load(caminho)
    payload["metadados"]["referencia"]["categoricas"].pop("Contract")
    forjado = tmp_path / "baseline_incompleto.joblib"
    joblib.dump(payload, forjado)

    with pytest.raises(artefato.ArtefatoIncompativel, match="não cobre"):
        artefato.carregar(forjado)


def test_o_artefato_promovido_expoe_a_referencia(art):
    """A ponte com a API: o que o `/health` e o `simulate_drift` vão ler."""
    ref = art.referencia
    assert ref["particao"] == "treino"
    assert ref["n"] > 0
    assert set(ref["numericas"]) | set(ref["categoricas"]) == set(art.features)

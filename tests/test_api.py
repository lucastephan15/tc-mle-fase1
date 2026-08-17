"""
Testes da API — Etapa 9d / 9d-sexies.

Dois níveis, com dublês em um e o artefato real no outro:

- **contrato** (`cliente_real`): o app montado sobre o artefato de fixture. É o
  que prova que os quatro payloads que corrompiam a predição agora dão 422.
- **lógica** (`dependency_overrides`): o app com um dublê de três linhas no
  lugar do pontuador. Prova limiar, formato de resposta e o 500 sem depender do
  modelo — e sem pagar a carga.

⚠️ O dublê nunca substitui o artefato real no teste de contrato: um teste de
integração que mocka a serialização e o carregamento fica verde testando os
próprios mocks.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_mod
from src.api import schema

# Um cliente válido, montado a partir do contrato real: sem constante à mão que
# possa divergir do artefato.
BASE = {
    "Total Charges": 108.15, "Tenure Months": 2.0, "Monthly Charges": 53.85,
    "Gender": "Male", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Multiple Lines": "No", "Internet Service": "DSL", "Online Security": "Yes",
    "Tech Support": "No", "Contract": "Month-to-month", "Paperless Billing": "Yes",
}


@pytest.fixture(scope="module")
def cliente_real(art):
    return TestClient(app_mod.criar_app(art), raise_server_exceptions=False)


class PontuadorFalso:
    """O dublê. Três linhas de estado e nenhum disco tocado."""

    def __init__(self, score: float = 0.99, limiar: float = 0.5, explode: bool = False):
        self._score, self._limiar, self._explode = score, limiar, explode

    versao = "dublê-0"
    sha256 = "0" * 64

    @property
    def limiar(self) -> float:
        return self._limiar

    @property
    def features(self) -> list[str]:
        return list(BASE)

    @property
    def versoes(self) -> dict[str, str]:
        return {}

    def pontuar(self, linhas: list[dict]) -> list[float]:
        if self._explode:
            # A mensagem que o sklearn produziria, com o dado do cliente dentro.
            raise ValueError("could not convert string to float: 'setenta reais'")
        return [self._score] * len(linhas)

    def pronto(self) -> bool:
        return True


def com_dublê(art, **kwargs) -> TestClient:
    app = app_mod.criar_app(art)
    app.dependency_overrides[app_mod.obter_pontuador] = lambda: PontuadorFalso(**kwargs)
    return TestClient(app, raise_server_exceptions=False)


# --- Contrato de domínio ---------------------------------------------------

@pytest.mark.parametrize("campo,valor", [
    ("Contract", "Vitalicio"),        # categoria inexistente
    ("Tenure Months", -999),          # sentinela de nulo de sistema legado
    ("Monthly Charges", -999),
    ("Total Charges", 1e9),
])
def test_payload_de_dominio_invalido_da_422(cliente_real, campo, valor):
    """Os quatro casos que a validação de ESQUEMA aprovava com 200.

    Medido em 17/08/2026 contra o campeão, sem validação de domínio:
    `Tenure = -999` dava P(churn) = 1,0000 nas 1.409 linhas e movia 878 clientes
    (62,3%) para o outro lado do limiar; `Contract` inexistente movia 153
    (10,9%), porque `handle_unknown="ignore"` faz o valor desconhecido virar
    linha de zeros — que para o modelo é *a categoria de referência*, não
    *não sei*.
    """
    r = cliente_real.post("/v1/predict", json={**BASE, campo: valor})
    assert r.status_code == 422, r.text


def test_campo_extra_da_422_e_nao_devolve_o_valor(cliente_real):
    """`CustomerID` é o campo que um integrador manda sem pensar — e é LGPD.

    Duas coisas num teste só, porque são duas correções distintas: `extra="forbid"`
    faz o campo ser rejeitado em vez de descartado em silêncio, e o handler de
    `RequestValidationError` impede que o **valor** volte na resposta (o default
    do Pydantic devolve `loc` E `input`).
    """
    r = cliente_real.post("/v1/predict", json={**BASE, "CustomerID": "3668-QPYBK"})
    assert r.status_code == 422
    assert "3668-QPYBK" not in r.text
    assert "CustomerID" in r.text  # o NOME do campo pode (e deve) aparecer


def test_campo_faltando_nao_ecoa_o_objeto_inteiro(cliente_real):
    """O erro `missing` do FastAPI devolve o payload INTEIRO em `detail[0].input`.

    É o caso que torna o vazamento invisível a teste manual: quem experimenta
    mandar lixo num campo vê só o valor daquele campo e conclui que está tudo
    bem.
    """
    payload = {k: v for k, v in BASE.items() if k != "Contract"}
    r = cliente_real.post("/v1/predict", json={**payload, "CustomerID": "3668-QPYBK"})
    assert r.status_code == 422
    assert "3668-QPYBK" not in r.text
    assert "53.85" not in r.text  # nenhum valor do payload volta


def test_ordem_das_chaves_nao_muda_a_predicao(cliente_real):
    """Seleção por NOME, não por posição — e agora com teste.

    A proteção veio de graça do `ColumnTransformer`, adotado na Etapa 2 por
    causa de *leakage*, não por contrato de API. Benefício que ninguém escolheu
    é benefício que alguém remove sem perceber: trocar o DataFrame por
    `np.array([[...]])` fixaria a ordem no corpo da função, onde nada a
    verifica, e a API responderia 200 com número errado.
    """
    direto = cliente_real.post("/v1/predict", json=BASE).json()
    invertido = cliente_real.post("/v1/predict", json=dict(reversed(list(BASE.items())))).json()
    assert direto["predicoes"][0]["probabilidade"] == invertido["predicoes"][0]["probabilidade"]


# --- Contrato de saída -----------------------------------------------------

def test_resposta_traz_probabilidade_limiar_e_versao(cliente_real, art):
    """Probabilidade, não classe; e o limiar que produziu a decisão.

    `.predict()` aplicaria 0,5 implícito — R$ 7.546 por ciclo e 83 churners de
    diferença, medidos na validação. O limiar acompanha a resposta porque sem
    ele a decisão não é auditável seis meses depois.
    """
    r = cliente_real.post("/v1/predict", json=BASE)
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["versao_modelo"] == art.versao
    p = corpo["predicoes"][0]
    assert 0.0 <= p["probabilidade"] <= 1.0
    assert p["limiar"] == art.limiar
    assert p["decisao"] == (p["probabilidade"] >= art.limiar)
    # response_model: nada além do declarado sai, mesmo que o handler inclua
    assert set(corpo) == {"request_id", "versao_modelo", "predicoes"}
    assert set(p) == {"probabilidade", "decisao", "limiar"}


def test_resposta_nao_ecoa_a_entrada(cliente_real):
    """Correlação por `request_id` gerado no servidor, nunca pelo payload de volta."""
    r = cliente_real.post("/v1/predict", json=BASE)
    assert "Month-to-month" not in r.text
    assert len(r.json()["request_id"]) == 36
    assert r.headers["X-Request-ID"] == r.json()["request_id"]
    assert float(r.headers["X-Response-Time-ms"]) > 0


def test_lote_e_unitario_concordam(cliente_real):
    """`/predict` é o caso particular de lote 1 — e concorda com o lote.

    🚨 **Concorda, mas NÃO bit a bit** — e o teste registra isso de propósito.
    Medido nas 1.409 linhas da validação: pontuar em lote e uma a uma dá
    resultados diferentes em **495 linhas (35%)**, com diferença máxima de
    **2,2e-16** (um ulp). É o BLAS: a multiplicação matriz-vetor toma caminhos
    de vetorização diferentes conforme o número de linhas.

    O que importa é o tamanho do efeito na decisão: **zero** clientes mudam de
    lado no limiar de 0,29. Mas a consequência de projeto fica registrada — a
    resposta da API para o mesmo cliente não é reproduzível **byte a byte**
    entre os dois endpoints, então nada rio abaixo pode comparar predições por
    igualdade exata (cache por hash de resposta, deduplicação, reconciliação do
    log da Etapa 10). O teste de caracterização do artefato não é afetado
    porque ele pontua sempre o mesmo lote.
    """
    um = cliente_real.post("/v1/predict", json=BASE).json()["predicoes"][0]
    lote = cliente_real.post("/v1/predict-batch", json={"clientes": [BASE] * 3}).json()
    assert len(lote["predicoes"]) == 3
    for p in lote["predicoes"]:
        assert p["probabilidade"] == pytest.approx(um["probabilidade"], abs=1e-12)
        assert p["decisao"] == um["decisao"]


def test_lote_tem_teto_declarado(cliente_real):
    """Lista sem teto é o vetor de negação de serviço: o trabalho é síncrono."""
    r = cliente_real.post("/v1/predict-batch",
                          json={"clientes": [BASE] * (schema.MAX_LOTE + 1)})
    assert r.status_code == 422


# --- Prontidão -------------------------------------------------------------

def test_health_declara_a_identidade_do_que_esta_carregado(cliente_real, art):
    """Um 200 que não diz QUAL modelo está servindo é o oitavo servidor.

    E o estado é lido do objeto que serve: um `/health` que consulta uma global
    setada no import responde `healthy` com o artefato apagado do disco —
    medido, com `/predict` levantando `FileNotFoundError` em seguida.
    """
    corpo = cliente_real.get("/health").json()
    assert corpo["status"] == "pronto"
    assert corpo["artefato_sha256"] == art.sha256
    assert corpo["versao_modelo"] == art.versao
    assert corpo["n_features"] == len(art.features)
    assert corpo["limiar_operacao"] == art.limiar
    assert corpo["versoes"]["scikit-learn"]


# --- Lógica isolada, com dublê ---------------------------------------------

def test_dublê_vence_a_injecao_real(art):
    """`dependency_overrides` substitui o pontuador — sem tocar o disco.

    É o que permite testar limiar e formato de resposta sem carregar artefato, e
    é também a verificação de que a rota depende da ABSTRAÇÃO: se ela
    instanciasse o modelo por dentro, este override não teria efeito nenhum e o
    teste passaria medindo o modelo real por engano.
    """
    c = com_dublê(art, score=0.99, limiar=0.5)
    p = c.post("/v1/predict", json=BASE).json()["predicoes"][0]
    assert p["probabilidade"] == 0.99
    assert p["limiar"] == 0.5
    assert p["decisao"] is True


@pytest.mark.parametrize("score,esperado", [(0.49, False), (0.50, True), (0.51, True)])
def test_decisao_e_maior_ou_igual_ao_limiar(art, score, esperado):
    """A borda exata do limiar — `>=`, não `>`. Um teste que o dublê torna trivial."""
    c = com_dublê(art, score=score, limiar=0.5)
    assert c.post("/v1/predict", json=BASE).json()["predicoes"][0]["decisao"] is esperado


def test_erro_interno_nao_vaza_a_mensagem_do_sklearn(art):
    """O 500 diz QUE falhou; o log diz POR QUÊ.

    `str(e)` do sklearn cita o dado do cliente — `could not convert string to
    float: 'setenta reais'` — e o material didático manda devolvê-lo quatro
    páginas depois de escrever *"em produção é melhor não expor detalhes"*.
    """
    c = com_dublê(art, explode=True)
    r = c.post("/v1/predict", json=BASE)
    assert r.status_code == 500
    assert "setenta reais" not in r.text
    assert r.json()["request_id"]


# --- O contrato contra o artefato ------------------------------------------

def test_schema_e_derivado_do_artefato_nao_do_config(art):
    """Os aliases do schema são exatamente as colunas que o pipeline espera.

    Duas listas mantidas iguais pela memória de quem escreveu divergem em
    silêncio — e esta divergência não apareceria como 422: apareceria como 500
    no `ColumnTransformer`, com dado real.
    """
    Cliente = schema.construir_modelo_entrada(art.pipeline)
    aliases = [c.alias for c in Cliente.model_fields.values()]
    assert aliases == list(art.pipeline.feature_names_in_)
    assert len(aliases) == 13


def test_toda_numerica_tem_faixa_declarada(art):
    """Feature numérica sem faixa entraria sem limite nenhum, e em silêncio.

    As faixas são DECISÃO de negócio (0–120 meses, 0–200, 0–12.000), folgadas de
    propósito: colar o limite no máximo do treino (72 meses) rejeitaria um
    cliente legítimo de 73. O alvo é `-999` e `1e9`, não a cauda.
    """
    _, categorias = schema.contrato_do_pipeline(art.pipeline)
    numericas = [c for c in art.pipeline.feature_names_in_ if c not in categorias]
    assert set(numericas) <= set(schema.FAIXAS), (
        f"sem faixa declarada: {set(numericas) - set(schema.FAIXAS)}"
    )


def test_openapi_publica_o_contrato(cliente_real):
    """A OpenAPI é a documentação de interface da entrega — não há segundo doc.

    ⚠️ Ela publica 13 nomes de feature, 25 valores de categoria e 3 faixas sem
    autenticação. Não é dado pessoal: é a descrição do modelo. Mantida por ser
    API interna, com a limitação declarada na documentação.
    """
    spec = cliente_real.get("/openapi.json").json()
    assert "/v1/predict" in spec["paths"]
    assert "/v1/predict-batch" in spec["paths"]
    assert "/health" in spec["paths"]

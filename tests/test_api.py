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

import os
import subprocess
import sys

import pytest
import sklearn
from fastapi.testclient import TestClient

from src import config
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
    def versoes_treino(self) -> dict[str, str]:
        return {}

    @property
    def versoes_runtime(self) -> dict[str, str]:
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
    # Duas perguntas diferentes, dois campos. Um campo `versoes` só, alimentado
    # pelos metadados do artefato, respondia "com que ambiente o modelo foi
    # treinado?" num endpoint cuja pergunta é "o que este serviço tem?". Medido
    # no container: o campo dizia Python 3.12.5 e o processo rodava 3.12.14.
    assert corpo["versoes_treino"]["scikit-learn"]
    assert corpo["versoes_runtime"]["scikit-learn"] == sklearn.__version__
    # O que NÃO pode divergir é o scikit-learn: é onde a serialização mora, e
    # `artefato.carregar()` mata o processo no boot se as duas discordarem.
    assert corpo["versoes_treino"]["scikit-learn"] == corpo["versoes_runtime"]["scikit-learn"]


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
    _, categorias, _ = schema.contrato_do_pipeline(art.pipeline)
    numericas = [c for c in art.pipeline.feature_names_in_ if c not in categorias]
    assert set(numericas) <= set(schema.FAIXAS), (
        f"sem faixa declarada: {set(numericas) - set(schema.FAIXAS)}"
    )


def test_importar_a_api_nao_exige_o_artefato():
    """Importar o módulo NÃO pode carregar nada do disco. Regressão real.

    Erro cometido, empurrado e pego pelo CI em 17/08/2026: um `app = criar_app()`
    no nível do módulo transformava a carga do artefato em **efeito colateral do
    import**. `make ci` passava na máquina de quem escreveu (o artefato está lá) e
    a suíte inteira falhava **na coleta** no runner limpo, onde `models/` está
    vazio por decisão — 81 testes derrubados por uma linha.

    🔑 A lição é sobre o instrumento, não sobre a linha: **este defeito só existe
    onde o arquivo não existe**, então nenhuma execução local podia encontrá-lo.
    É o "funciona na minha máquina" com a raiz correta — e o teste abaixo o traz
    para dentro do alcance do desenvolvimento, simulando a máquina limpa com um
    caminho de artefato que não existe.

    A propriedade que a factory preserva: quem chama `criar_app()` carrega, e
    falha alto se o artefato não estiver lá.
    """
    codigo = (
        "import src.api.app as m\n"
        "print('import ok')\n"
        "try:\n"
        "    m.criar_app()\n"
        "except Exception as e:\n"
        "    print(type(e).__name__)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True, text=True, cwd=config.RAIZ,
        env={**os.environ, "TC_ARTEFATO": "/caminho/que/nao/existe.joblib"},
    )
    assert r.returncode == 0, r.stderr
    assert "import ok" in r.stdout
    assert "ArtefatoIncompativel" in r.stdout


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


# --- O vazio declarado: o defeito que só o container revelou ----------------

def test_vazio_de_total_charges_e_aceito_e_pontuado(cliente_real, art):
    """O cliente do PRIMEIRO MÊS tem de poder ser pontuado.

    🚨 Regressão real, encontrada em 18/08/2026 pelo teste de integração da Etapa
    9e — e só porque ele mandou ao container as **1.409 linhas reais da
    validação** em vez de um payload sintético escolhido a dedo. Uma delas
    (índice 487) tem `Total Charges` vazio, e o contrato devolvia **422** para ela.

    No Telco são **11 clientes, todos com `Tenure Months = 0`**: quem ainda não
    teve ciclo de faturamento. A Etapa 2 decidiu que esse vazio é *medição
    verdadeira* e o imputa com 0 — o pipeline sempre soube tratá-los. Quem não
    sabia era o schema, e o resultado era a API recusando exatamente a população
    que uma campanha de retenção mais quer pontuar.

    🔑 *O contrato ficou mais estreito que o pipeline que ele protege* — o erro
    simétrico do que o módulo de schema existe para impedir.
    """
    corpo = cliente_real.post("/v1/predict", json={**BASE, "Total Charges": None})
    assert corpo.status_code == 200, corpo.json()
    p_api = corpo.json()["predicoes"][0]["probabilidade"]

    # E o número tem de ser o do pipeline, não um qualquer que "não falhou".
    import numpy as np
    import pandas as pd
    X = pd.DataFrame([{**BASE, "Total Charges": np.nan}])
    p_pipe = float(art.pipeline.predict_proba(X)[:, 1][0])
    assert p_api == pytest.approx(p_pipe, abs=1e-12)


def test_none_puro_quebraria_o_pipeline_se_nao_virasse_nan(art):
    """A outra metade da correção — e sozinha nenhuma das duas serve.

    Aceitar `null` no schema **sem** converter para `NaN` no serviço trocaria o
    422 por um **500**: `pd.DataFrame` com `None` produz `dtype=object`, o
    imputador não reconhece a ausência e o `LogisticRegression` levanta
    `ValueError: Input X contains NaN`. Este teste fixa o mecanismo, não só o
    resultado — se alguém "simplificar" a conversão no serviço, o teste acima
    continuaria verde por acaso? Não: quebraria com 500. Este aqui diz **por quê**.
    """
    import pandas as pd
    with pytest.raises(ValueError, match="NaN"):
        art.pipeline.predict_proba(pd.DataFrame([{**BASE, "Total Charges": None}]))


def test_vazio_NAO_e_aceito_onde_nao_significa_nada(cliente_real):
    """`Tenure Months: null` continua 422, e a assimetria é deliberada.

    O grupo `num` do pré-processamento também tem imputador (mediana), então o
    pipeline *tecnicamente* aceitaria o vazio aqui também. Não é o que se quer:
    em `Total Charges` o vazio significa "sem ciclo de faturamento" e o valor
    imputado É essa informação; em `Tenure Months` um vazio é dado faltando do
    integrador, e imputar a mediana do treino em silêncio transformaria uma falha
    de integração numa predição plausível — a sentinela de nulo entrando pela
    porta da frente. *O pipeline sabe imputar as duas; a API aceita o vazio só
    onde ele quer dizer algo.*
    """
    for coluna in ("Tenure Months", "Monthly Charges"):
        r = cliente_real.post("/v1/predict", json={**BASE, coluna: None})
        assert r.status_code == 422, f"{coluna}: {r.json()}"


def test_o_vazio_aceito_sai_do_ARTEFATO_e_nao_de_uma_lista_a_mao(art):
    """Quem decide onde `null` vale é o pipeline, como todo o resto do contrato.

    Se amanhã outra coluna entrar no grupo `zero` (ou sair dele), o schema
    acompanha sozinho. Uma lista escrita à mão aqui seria a terceira cópia do
    contrato — depois de `config.FEATURES` e do próprio artefato — e a que
    diverge em silêncio.
    """
    _, _, aceitam_vazio = schema.contrato_do_pipeline(art.pipeline)
    ct = art.pipeline.named_steps["preproc"]
    do_pipeline = [c for nome, _, cols in ct.transformers_ if nome == "zero" for c in cols]
    assert aceitam_vazio == do_pipeline
    assert aceitam_vazio, "o grupo `zero` não pode estar vazio: é o tratamento da Etapa 2"


def test_faixa_continua_valendo_na_coluna_que_aceita_vazio(cliente_real):
    """Aceitar `null` não pode ter aberto a porta para `-999` e `1e9`.

    O `ge`/`le` ficou no tipo INTERNO justamente por isso: no tipo externo ele
    rejeitaria o próprio `None` que a coluna existe para aceitar, e a tentação
    seria removê-lo — trocando um buraco por outro maior.
    """
    for valor in (-999, 1e9):
        r = cliente_real.post("/v1/predict", json={**BASE, "Total Charges": valor})
        assert r.status_code == 422, f"{valor}: {r.json()}"

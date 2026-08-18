"""Testes do log de inferência — Etapa 10a.

O que se verifica aqui **não é que o log parece certo** — é que ele é parseável
por máquina e que a decisão de máscara LGPD está onde foi declarada. O consumidor
do `.jsonl` é o `monitoring.py`, então a linha tem um schema, e schema é contrato.

🔑 Dois destes testes verificam **reprovando**, que é a disciplina do repo desde
o gate: o do 500 falharia com um middleware que lesse `resposta.status_code`
(porque o handler de `Exception` roda por fora dele), e o das features falharia
com um log que emitisse o payload por default.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_mod
from src.api import observabilidade as obs
from tests.test_api import BASE, PontuadorFalso

CANONICOS = {
    "timestamp", "request_id", "metodo", "rota",
    "status_code", "latency_ms", "artefato_sha256",
}


def _linhas(capsys) -> list[dict]:
    """Toda linha emitida, já como dict. Falha se alguma não for JSON.

    É o teste central disfarçado de helper: se um dia alguém acrescentar um
    prefixo ao formatter, ou reativar o access log do uvicorn dentro do mesmo
    stream, `json.loads` levanta aqui e o motivo fica óbvio.

    🚨 **Este helper já pegou um defeito de verdade.** Na primeira versão o
    handler era um `StreamHandler(sys.stdout)` comum, que congela o stream na
    construção: sob o pytest a linha saía num descritor que **nem `capsys` nem
    `capfd` liam**, e a asserção de que a linha existe é o que denunciou. Um
    teste mais frouxo ("a requisição respondeu 200, logo logou") teria ficado
    verde sem nunca ter visto uma linha. Daí `_linhas` sempre **parsear** e os
    testes sempre olharem o conteúdo — nunca a ausência de exceção.
    """
    saida = capsys.readouterr().out
    return [json.loads(ln) for ln in saida.splitlines() if ln.strip()]


@pytest.fixture
def cliente(capsys, art):
    """⚠️ `capsys` é pedido AQUI, e não só na função de teste.

    Declarar a dependência força a captura a estar ativa **antes** de
    `criar_app()` rodar. Com o handler resolvendo `sys.stdout` na emissão isso
    deixou de ser condição de correção, mas continua sendo o que mantém a ordem
    explícita em vez de depender de como o pytest resolve fixtures de mesmo
    escopo.
    """
    return TestClient(app_mod.criar_app(art), raise_server_exceptions=False)


def test_uma_linha_json_por_requisicao(cliente, capsys):
    """A linha inteira é JSON válido e traz os 7 campos canônicos."""
    cliente.post("/v1/predict", json=BASE)
    linhas = _linhas(capsys)
    assert len(linhas) == 1, "uma requisição, uma linha"
    assert set(linhas[0]) >= CANONICOS


def test_sha256_do_artefato_vai_na_linha(cliente, art, capsys):
    """O 7º campo — o que a resposta já tem, mas a resposta é efêmera.

    Sem ele, "trocaram o modelo no meio da janela" é uma hipótese que a análise
    de drift não consegue descartar.
    """
    cliente.post("/v1/predict", json=BASE)
    assert _linhas(capsys)[0]["artefato_sha256"] == art.sha256


def test_request_id_da_linha_e_o_da_resposta(cliente, capsys):
    """A correlação que substitui o eco do payload.

    O `request_id` só vale se for **o mesmo** dos dois lados: é ele que permite
    responder "por que este cliente recebeu este score?" sem devolver dado
    pessoal na resposta.
    """
    r = cliente.post("/v1/predict", json=BASE)
    assert _linhas(capsys)[0]["request_id"] == r.json()["request_id"] == r.headers["X-Request-ID"]


def test_422_tambem_e_logado(cliente, capsys):
    """Instrumentação que só mede o caminho feliz mede o que não precisa.

    Um payload inválido nunca chega a `_responder`; se a linha fosse escrita lá,
    a taxa de erro seria sempre zero.
    """
    cliente.post("/v1/predict", json={**BASE, "Contract": "Vitalicio"})
    linha = _linhas(capsys)[0]
    assert linha["status_code"] == 422
    # Nem `n_linhas` nem `scores`: ausente é diferente de zero. Nada foi pontuado.
    assert "n_linhas" not in linha and "scores" not in linha


def test_500_tambem_e_logado(art, capsys):
    """🚨 O teste que reprova o middleware ingênuo.

    O handler de `Exception` do FastAPI vive no `ServerErrorMiddleware`, que é o
    **mais externo** — logo ele roda por FORA deste middleware, que vê a exceção
    crua subindo e nunca vê a resposta 500. Um log escrito a partir de
    `resposta.status_code` perderia toda falha interna, que é justamente o evento
    que o log existe para registrar.
    """
    app = app_mod.criar_app(art)
    app.dependency_overrides[app_mod.obter_pontuador] = lambda: PontuadorFalso(explode=True)
    TestClient(app, raise_server_exceptions=False).post("/v1/predict", json=BASE)

    linha = _linhas(capsys)[0]
    assert linha["status_code"] == 500
    assert linha["latency_ms"] > 0, "a falha também é cronometrada"


def test_o_dado_pessoal_NAO_vai_no_log_por_default(cliente, capsys):
    """🚨 A decisão de máscara LGPD, fixada como teste.

    O payload do Telco carrega `Gender`, `Senior Citizen`, `Partner` e
    `Dependents` — mantidas nas features **de propósito** (Etapa 5), para que a
    10.5 possa medir o viés. Em container o log vai para stdout, e num PaaS o
    stdout é coletado pela plataforma: um terceiro. Não há `.gitignore` nem
    `.dockerignore` para essa camada.

    Este teste é o que impede a regressão silenciosa — alguém acrescenta
    `features` "para facilitar o debug" e o dado pessoal passa a sair por
    default, sem nada quebrar.
    """
    assert obs.LOGAR_FEATURES is False, "o default é a direção segura"
    cliente.post("/v1/predict", json=BASE)
    linha = _linhas(capsys)[0]
    assert "features" not in linha
    bruto = json.dumps(linha)
    for sensivel in ("Male", "Gender", "Senior Citizen", "Partner", "Dependents"):
        assert sensivel not in bruto


def test_scores_vao_sempre__prediction_drift_de_graca(cliente, capsys):
    """A metade da vigilância que não custa privacidade.

    Probabilidade sem atributo ao lado não identifica ninguém, então
    prediction drift roda em produção sem exposição nenhuma. É sintoma e não
    causa — mas é sintoma de graça.
    """
    lote = {"clientes": [BASE, {**BASE, "Contract": "Two year"}]}
    cliente.post("/v1/predict-batch", json=lote)
    linha = _linhas(capsys)[0]
    assert linha["n_linhas"] == 2
    assert len(linha["scores"]) == 2
    assert all(0.0 <= s <= 1.0 for s in linha["scores"])


def test_features_entram_quando_explicitamente_ligado(art, capsys, monkeypatch):
    """O outro lado da mesma decisão: `simulate_drift.py` precisa das entradas.

    Ligar é uma variável de ambiente, e ligar é a decisão de quem roda — a
    ausência dela nunca expõe nada, que é a diferença entre este `getenv` com
    default e o do item 100 (lá o default era o valor inseguro).
    """
    monkeypatch.setattr(obs, "LOGAR_FEATURES", True)
    TestClient(app_mod.criar_app(art)).post("/v1/predict", json=BASE)
    linha = _linhas(capsys)[0]
    assert linha["features"] == [BASE]


def test_rota_e_o_template_nao_a_url(cliente, capsys):
    """Cardinalidade: a rota é a chave de agregação, não o identificador.

    Hoje as rotas são fixas e template e URL coincidem — o teste existe para o
    dia em que uma delas ganhar `/{id}` e ninguém lembrar de voltar aqui.
    """
    cliente.get("/health")
    assert _linhas(capsys)[0]["rota"] == "/health"

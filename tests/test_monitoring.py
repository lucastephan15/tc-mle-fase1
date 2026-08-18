"""
Testes do leitor do log de inferência — Etapa 10a-3 / 10c.

O que se verifica aqui não é "o número está certo" (o PSI já tem os seus
controles em `test_referencia.py`): é **se o painel sabe quando NÃO pode
concluir**. Um monitoramento que responde com confiança sobre uma janela que
mistura dois modelos, ou que chama "estável" o que nunca foi medido, é pior que
não ter painel — ele produz a sensação de vigilância.

⚠️ O `simulate_drift.py` fica fora desta suíte, pelo mesmo critério do
`integracao_container.py` (9e): ele sobe um servidor de verdade e leva segundos.
A suíte prova a lógica; o script prova o caminho completo.
"""

from __future__ import annotations

import json

import pytest

from src import monitoring


def _linha(**campos) -> dict:
    base = {
        "timestamp": "2026-08-18T22:00:00+00:00",
        "request_id": "id-1",
        "metodo": "POST",
        "rota": "/v1/predict-batch",
        "status_code": 200,
        "latency_ms": 2.5,
        "artefato_sha256": "a" * 64,
    }
    return {**base, **campos}


@pytest.fixture
def sha(art) -> str:
    return art.sha256


def test_linhas_ilegiveis_sao_contadas_nao_engolidas(tmp_path):
    """A linha de texto puro no meio do `.jsonl` é o sintoma de outro emissor.

    É o defeito da 10a: `uvicorn` sem `--no-access-log` escreve uma linha de
    texto por requisição no mesmo stream. Descartá-la em silêncio faria o painel
    reportar metade do volume real e ninguém perguntaria pela outra metade.
    """
    caminho = tmp_path / "log.jsonl"
    caminho.write_text(
        json.dumps(_linha()) + "\n"
        + 'INFO:     127.0.0.1:52341 - "POST /v1/predict HTTP/1.1" 200 OK\n'
        + "\n"
        + json.dumps(_linha(request_id="id-2")) + "\n",
        encoding="utf-8",
    )
    linhas, invalidas = monitoring.ler(caminho)
    assert len(linhas) == 2
    assert invalidas == 1


def test_janela_que_mistura_modelos_e_denunciada(art, sha):
    """Duas versões na mesma janela ⇒ nenhum PSI daqui tem um denominador só.

    🔑 É o uso do `artefato_sha256` que a Etapa 10a previu: PSI alto tem duas
    explicações — a população mudou, ou trocaram o modelo no meio da janela — e
    sem esta checagem a segunda é indistinguível da primeira.
    """
    linhas = [
        _linha(artefato_sha256=sha, scores=[0.1, 0.2]),
        _linha(artefato_sha256="b" * 64, scores=[0.3], request_id="id-2"),
    ]
    painel = monitoring.analisar(linhas, art)
    assert painel["janela_homogenea"] is False
    assert "MISTURA MODELOS" in monitoring.relatorio(painel, art)


def test_log_de_outro_artefato_e_denunciado(art):
    """Baseline de um modelo contra predições de outro não falha — mede errado."""
    linhas = [_linha(artefato_sha256="c" * 64, scores=[0.1, 0.9])]
    painel = monitoring.analisar(linhas, art)
    assert painel["janela_homogenea"] is True
    assert painel["baseline_do_mesmo_artefato"] is False
    assert "NÃO É DESTE ARTEFATO" in monitoring.relatorio(painel, art)


def test_features_ausentes_sao_ausencia_de_medicao_nao_estabilidade(art, sha):
    """Sem `TC_LOG_FEATURES=1` não há data drift — e isso tem de aparecer.

    Não medir e medir zero produzem o mesmo silêncio no painel, e só um deles é
    informação. O texto tem de dizer a diferença, porque é o que separa "está
    tudo bem" de "não estamos olhando".
    """
    painel = monitoring.analisar([_linha(artefato_sha256=sha, scores=[0.2])], art)
    assert painel["data_drift"] is None

    texto = monitoring.relatorio(painel, art)
    assert "AUSÊNCIA DE MEDIÇÃO" in texto
    assert "não medido" in texto


def test_latencia_em_percentis_e_erros_contados(art, sha):
    """A média mente; o painel fala P50/P95/P99. E 4xx/5xx entram na taxa de erro.

    O 422 e o 500 são cronometrados de propósito (o `try/finally` da 9d): sem
    eles o P95 fica cego justamente para o que falha rápido.
    """
    linhas = [_linha(artefato_sha256=sha, scores=[0.2], latency_ms=float(ms))
              for ms in range(1, 100)]
    linhas.append(_linha(artefato_sha256=sha, status_code=422, latency_ms=0.4))
    linhas.append(_linha(artefato_sha256=sha, status_code=500, latency_ms=0.5))

    painel = monitoring.analisar(linhas, art)
    assert painel["latencia_ms"]["p50"] < painel["latencia_ms"]["p95"]
    assert painel["taxa_erro"] == pytest.approx(2 / 101, abs=1e-4)
    assert painel["status"][422] == 1
    # A janela recusada não pontuou ninguém: 'com_predicao' conta o que virou
    # inferência, não o que chegou.
    assert painel["com_predicao"] == 99


def test_janela_so_de_erros_nao_vira_drift_estavel(art, sha):
    """O cenário `categoria` do simulate_drift: tudo recusado no contrato.

    Zero predições não é "sem drift" — é volume zero, que a Etapa 10b lista como
    sintoma próprio (queda repentina de volume = possível falha upstream).
    """
    painel = monitoring.analisar(
        [_linha(artefato_sha256=sha, status_code=422, latency_ms=0.3)], art,
    )
    assert painel["prediction_drift"]["classificacao"] == "sem-dados"
    assert painel["taxa_erro"] == 1.0
    assert "volume zero" in monitoring.relatorio(painel, art)

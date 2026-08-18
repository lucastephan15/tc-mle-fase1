"""
Etapa 9e — teste de integração contra o CONTAINER, não contra o processo local.

    make docker-teste     # constrói a imagem, sobe, roda isto, derruba

🔑 **Por que não é um teste do `pytest`.** Exige Docker, uma imagem construída e
~1 min de build; pôr isso na suíte transformaria o CI leve num CI que alguém
desliga. A suíte prova que a **lógica** está certa; isto prova que o **artefato
carrega empacotado**. São as duas metades do 9d-sexies, e um teste que precisasse
das duas ao mesmo tempo não localizaria a falha em nenhuma.

🎯 **O que ele encontrou na primeira execução (18/08/2026), e por quê.** Manda ao
container as **1.409 linhas reais da validação**, não um payload sintético — e
uma delas (índice 487) tem `Total Charges` vazio: um dos 11 clientes do Telco sem
ciclo de faturamento, todos com `Tenure Months = 0`. O contrato devolvia **422**
para ela, porque o schema era mais estreito que o pipeline que ele protege. Um
smoke test com um cliente escolhido a dedo nunca teria visto. *A carga ser dado
real, e não inventado, é o que faz a diferença aqui.*

⚠️ `NaN` vira `null` no payload: `json.dumps(float("nan"))` emite `NaN` literal,
que **não é JSON válido**. O servidor responde 422 com `less_than_equal`, uma
mensagem que aponta para o lugar errado. Quem integra manda `null`.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import average_precision_score

from src import artefato as art_mod
from src import data

BASE = os.getenv("TC_API_BASE", "http://localhost:8010")


def post(rota: str, payload: dict) -> tuple[int, dict, float]:
    """Devolve (status, corpo, latência em ms). O 4xx/5xx é resposta, não exceção."""
    req = urllib.request.Request(
        BASE + rota,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    inicio = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            corpo, status = json.load(r), r.status
    except urllib.error.HTTPError as e:
        corpo, status = json.load(e), e.code
    return status, corpo, (time.perf_counter() - inicio) * 1000


def sem_nan(linha: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in linha.items()}


def medir(rota: str, payload: dict, n_conc: int, repeticoes: int) -> list[float]:
    """Latência sob concorrência. Um `curl` é UMA observação, quase certamente no P50."""
    def bater(_: int) -> float:
        return post(rota, payload)[2]

    lat: list[float] = []
    with ThreadPoolExecutor(max_workers=n_conc) as ex:
        list(ex.map(bater, range(n_conc)))          # aquecimento
        for _ in range(repeticoes):
            lat.extend(ex.map(bater, range(n_conc)))
    return sorted(lat)


def percentis(lat: list[float]) -> tuple[float, float, float]:
    return lat[len(lat) // 2], lat[int(len(lat) * 0.95)], lat[-1]


def verificar_identidade(artefato, Xv, linhas, y) -> list[str]:
    """O container prediz o MESMO que o pipeline local?

    A tolerância NÃO é ruído aceito: é a diferença de BLAS entre plataformas.
    Bit-identidade entre macOS e Linux não é propriedade que se possa exigir —
    **ausência de decisão trocada** é, e é essa que a operação consome.
    """
    local = artefato.pipeline.predict_proba(Xv)[:, 1]
    status, corpo, ms = post("/v1/predict-batch", {"clientes": linhas})
    if status != 200:
        print(f"❌ lote rejeitado ({status}): {corpo}")
        return ["o container recusou o lote da validação"]

    remoto = np.array([p["probabilidade"] for p in corpo["predicoes"]])
    dif = np.abs(local - remoto)
    n_dif = int((dif > 0).sum())
    trocadas = int(((local >= artefato.limiar) != (remoto >= artefato.limiar)).sum())

    print(f"\n[1] IDENTIDADE · {len(remoto)} linhas em {ms:.1f} ms")
    print(f"    max|dif| = {dif.max():.3e}   linhas que diferem: {n_dif} "
          f"({n_dif / len(local) * 100:.1f}%)")
    print(f"    decisões trocadas no limiar {artefato.limiar}: {trocadas}")
    print(f"    PR-AUC local {average_precision_score(y, local):.10f} "
          f"· container {average_precision_score(y, remoto):.10f}")

    falhas = []
    if dif.max() > 1e-9 or trocadas:
        falhas.append("as predições do container divergem do pipeline local")

    # Unitário × lote, sem igualdade exata de propósito: pontuar N linhas de uma
    # vez e uma a uma difere em ~1 ulp porque o BLAS muda o caminho de
    # vetorização com o número de linhas. Nada rio abaixo (cache por hash,
    # deduplicação, reconciliação do log da Etapa 10) pode comparar predições
    # por igualdade exata.
    st, corpo_unit, _ = post("/v1/predict", linhas[0])
    p_unit = corpo_unit["predicoes"][0]["probabilidade"]
    print(f"\n[2] UNITÁRIO × LOTE · dif = {abs(p_unit - remoto[0]):.3e}")
    if st != 200 or abs(p_unit - remoto[0]) > 1e-9:
        falhas.append("unitário e lote discordam")
    return falhas


def verificar_contrato_de_erro(linhas) -> list[str]:
    """Os quatro payloads que corrompiam a predição continuam barrados — e o
    vazio declarado continua passando."""
    print("\n[3] CONTRATO DE ERRO (dentro do container)")
    casos = {
        "categoria inédita": ({**linhas[0], "Contract": "Vitalicio"}, 422),
        "sentinela -999": ({**linhas[0], "Tenure Months": -999}, 422),
        "campo extra (CustomerID)": ({**linhas[0], "CustomerID": "3668-QPYBK"}, 422),
        "campo faltando": ({k: v for k, v in linhas[0].items() if k != "Contract"}, 422),
        "vazio declarado (null)": ({**linhas[0], "Total Charges": None}, 200),
    }
    falhas = []
    for nome, (payload, alvo) in casos.items():
        st, corpo, _ = post("/v1/predict", payload)
        texto = json.dumps(corpo)
        vazou = any(v in texto for v in ("3668-QPYBK", "-999", "Vitalicio"))
        ok = st == alvo and not vazou
        print(f"    {nome:26s} -> {st}  (esperado {alvo})  "
              f"vaza valor do payload: {vazou}   {'ok' if ok else 'FALHOU'}")
        if not ok:
            falhas.append(f"contrato de erro: {nome}")
    return falhas


def medir_latencia(linhas) -> None:
    """Um `curl` é UMA observação, quase certamente no P50. SLA se escreve em
    percentil, e a cauda só aparece sob concorrência."""
    print("\n[4] LATÊNCIA (inclui HTTP e a rede do Docker, não só o predict)")
    for rotulo, rota, payload, conc, rep in [
        ("unitário, serial", "/v1/predict", linhas[0], 1, 40),
        ("unitário, 8 concorrentes", "/v1/predict", linhas[0], 8, 5),
        ("lote de 1.409", "/v1/predict-batch", {"clientes": linhas}, 1, 20),
    ]:
        p50, p95, pmax = percentis(medir(rota, payload, conc, rep))
        print(f"    {rotulo:26s} p50={p50:7.2f} ms  p95={p95:7.2f} ms  máx={pmax:7.2f} ms")


def verificar_probe_sob_carga(linhas) -> list[str]:
    """O `def` × `async def` verificado no ambiente real.

    Com `async def` o event loop congela pelo lote inteiro, a probe do
    orquestrador expira e o container é reiniciado no meio da campanha — o
    serviço morto justamente por estar trabalhando.
    """
    print("\n[5] O /health SOBREVIVE À CARGA?")

    def health() -> float:
        inicio = time.perf_counter()
        urllib.request.urlopen(BASE + "/health", timeout=60).read()
        return (time.perf_counter() - inicio) * 1000

    with ThreadPoolExecutor(max_workers=9) as ex:
        lotes = [ex.submit(post, "/v1/predict-batch", {"clientes": linhas})
                 for _ in range(8)]
        time.sleep(0.005)
        probes = [ex.submit(health) for _ in range(5)]
        pior_lote = max(f.result()[2] for f in lotes)
        latencias = [f.result() for f in probes]

    print(f"    8 lotes de 1.409 em voo: {pior_lote:.1f} ms (pior)")
    print(f"    /health durante a carga: p50={statistics.median(latencias):.1f} ms  "
          f"máx={max(latencias):.1f} ms")
    # O HEALTHCHECK do Dockerfile expira em 5 s: acima disso o orquestrador
    # começaria a contar falhas e reiniciaria o container sob carga.
    if max(latencias) > 4_000:
        return ["o /health ficou preso na fila — a probe expiraria"]
    return []


def main() -> int:
    artefato = art_mod.carregar()
    dados = data.dividir()
    Xv = dados.validacao.X[artefato.features]
    linhas = [sem_nan(r) for r in Xv.to_dict(orient="records")]
    print(f"alvo: {BASE} · {len(linhas)} linhas · {len(artefato.features)} features")

    falhas = verificar_identidade(artefato, Xv, linhas, dados.validacao.y)
    falhas += verificar_contrato_de_erro(linhas)
    medir_latencia(linhas)
    falhas += verificar_probe_sob_carga(linhas)

    print()
    for f in falhas:
        print(f"❌ {f}")
    if falhas:
        return 1
    print("✅ o container serve o mesmo modelo, com o mesmo contrato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Leitura do log de inferência — Etapa 10a-3 / 10c.

    python -m src.monitoring logs/inferencia.jsonl

Transforma o `.jsonl` que a API emite (Etapa 10a) nas quatro famílias de métrica
do 10b, na medida em que cada uma é observável **sem ground truth**:

| família | aqui | por quê |
|---|---|---|
| serviço | latência em **percentis**, erros por status | a média mente: P99 subindo com P50 parado é gargalo sob estresse |
| drift | PSI dos scores (sempre) e das features (se logadas) | único sinal em tempo real, e o único que funciona **sem rótulo** |
| negócio | tamanho da fila de retenção, volume | "sem erro" ≠ "tudo bem": queda de volume é falha upstream |
| qualidade preditiva | **ausente, e declarado** | depende do churn se confirmar — a janela cega |

🚨 **A primeira coisa que este módulo faz não é estatística, é identidade.** Ele
conta quantos `artefato_sha256` distintos aparecem na janela e os confronta com o
artefato cujo baseline está sendo usado. É o que torna aquele campo do log
verificável em vez de decorativo: PSI alto tem duas explicações — *a população
mudou* ou *trocaram o modelo no meio da janela* — e sem essa checagem a segunda
fica indistinguível da primeira. Uma métrica que não sabe a que objeto se refere
não é medição, é coincidência.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src import artefato as art_mod
from src import referencia


def ler(caminho: Path) -> tuple[list[dict], int]:
    """Lê o `.jsonl` e devolve (linhas válidas, nº de linhas ilegíveis).

    As ilegíveis são **contadas, não engolidas**: elas são o sintoma de que
    algum outro emissor está escrevendo no mesmo stream — o access log do
    uvicorn é o caso conhecido (9f/10a), e o dia em que uma biblioteca começar a
    imprimir aviso em stdout o número aqui é que vai dizer.
    """
    linhas, invalidas = [], 0
    with open(caminho, encoding="utf-8") as f:
        for bruta in f:
            bruta = bruta.strip()
            if not bruta:
                continue
            try:
                linhas.append(json.loads(bruta))
            except json.JSONDecodeError:
                invalidas += 1
    return linhas, invalidas


def _percentis(valores: list[float]) -> dict[str, float]:
    """P50/P95/P99 — SLA se escreve em percentil, nunca em média."""
    if not valores:
        return {}
    a = np.asarray(valores, dtype=float)
    return {
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


def analisar(linhas: list[dict], art: art_mod.Artefato) -> dict:
    """O painel inteiro, como dados. A formatação fica em `relatorio`."""
    inferencia = [x for x in linhas if x.get("scores")]
    shas = Counter(x.get("artefato_sha256") for x in linhas if x.get("artefato_sha256"))

    saida: dict = {
        "requisicoes": len(linhas),
        "com_predicao": len(inferencia),
        "linhas_pontuadas": sum(x.get("n_linhas", 0) for x in inferencia),
        "status": dict(Counter(x.get("status_code") for x in linhas).most_common()),
        "latencia_ms": _percentis([x["latency_ms"] for x in linhas if "latency_ms" in x]),
        "artefatos_na_janela": dict(shas),
        # A janela mistura modelos? Então nenhum PSI daqui tem um denominador só.
        "janela_homogenea": len(shas) <= 1,
        "baseline_do_mesmo_artefato": set(shas) <= {art.sha256} and bool(shas),
    }

    erros = sum(n for s, n in saida["status"].items() if isinstance(s, int) and s >= 400)
    saida["taxa_erro"] = round(erros / len(linhas), 4) if linhas else None

    scores = [s for x in inferencia for s in x["scores"]]
    saida["prediction_drift"] = referencia.comparar_scores(art.referencia, scores)

    # Data drift só existe se as features tiverem sido logadas — e elas só são
    # logadas com TC_LOG_FEATURES=1. A ausência é reportada como ausência, nunca
    # como "sem drift": não medir e medir zero produzem o mesmo silêncio.
    registros = [linha for x in inferencia for linha in x.get("features", [])]
    if registros:
        saida["data_drift"] = referencia.comparar(art.referencia, pd.DataFrame(registros))
    else:
        saida["data_drift"] = None

    return saida


def relatorio(painel: dict, art: art_mod.Artefato, invalidas: int = 0) -> str:
    linhas = [
        f"Requisições      : {painel['requisicoes']} "
        f"({painel['com_predicao']} com predição · "
        f"{painel['linhas_pontuadas']} clientes pontuados)",
        f"Status           : {painel['status']}  ·  taxa de erro {painel['taxa_erro']}",
        f"Latência (ms)    : {painel['latencia_ms']}",
    ]
    if invalidas:
        linhas.append(
            f"⚠️ Linhas não-JSON: {invalidas} — outro emissor escreveu no mesmo "
            f"stream (access log do servidor ligado?)"
        )

    if not painel["janela_homogenea"]:
        linhas.append(
            f"🚨 A JANELA MISTURA MODELOS: {painel['artefatos_na_janela']}. "
            f"Qualquer drift medido abaixo tem duas explicações e nenhuma é "
            f"descartável — separe as janelas por artefato antes de concluir."
        )
    elif not painel["baseline_do_mesmo_artefato"]:
        linhas.append(
            f"🚨 O LOG NÃO É DESTE ARTEFATO: janela={list(painel['artefatos_na_janela'])}, "
            f"baseline={art.sha256[:16]}…. Baseline de um modelo contra "
            f"predições de outro não falha — mede errado."
        )

    p = painel["prediction_drift"]
    linhas.append("")
    linhas.append(f"PREDICTION DRIFT (sempre disponível — n={p['n']})")
    if p.get("psi") is None:
        linhas.append("  sem dados na janela — o que é 'volume zero', não 'estável'")
    else:
        linhas.append(
            f"  PSI {p['psi']:.4f} ({p['classificacao']})  ·  "
            f"média {p['media_ref']:.4f} → {p['media_janela']:.4f}"
        )
        if "fila_x" in p:
            linhas.append(
                f"  fila de retenção: {p['taxa_acima_do_limiar_ref']:.1%} → "
                f"{p['taxa_acima_do_limiar']:.1%} da carteira ({p['fila_x']}×)"
            )

    linhas.append("")
    if painel["data_drift"] is None:
        linhas.append(
            "DATA DRIFT: não medido — as features não estão no log "
            "(TC_LOG_FEATURES=0, que é o default em produção por LGPD).\n"
            "  ⚠️ Isto é AUSÊNCIA DE MEDIÇÃO, não ausência de drift."
        )
    else:
        linhas.append("DATA DRIFT (exige TC_LOG_FEATURES=1)")
        ordenado = sorted(painel["data_drift"].items(), key=lambda kv: -kv[1]["psi"])
        for coluna, r in ordenado:
            marca = {"estavel": "  ", "investigar": "⚠️", "agir": "🚨"}[r["classificacao"]]
            linha = f"  {marca} {coluna:<20} PSI {r['psi']:.4f}  {r['classificacao']}"
            if r.get("categorias_ineditas"):
                linha += f"  · categorias inéditas: {r['categorias_ineditas']}"
            linhas.append(linha)

    linhas.append("")
    linhas.append(
        "QUALIDADE PREDITIVA: não computável aqui, e a ausência é a informação.\n"
        "  Acurácia/PR-AUC exigem o rótulo, e o churn só se confirma ao fim do\n"
        "  ciclo de faturamento — a janela cega do ground truth. É por isso que\n"
        "  o drift é o que se vigia em tempo real: ele não precisa de y."
    )
    return "\n".join(linhas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("log", type=Path, help="arquivo .jsonl emitido pela API")
    parser.add_argument("--artefato", type=Path, default=None,
                        help="artefato cujo baseline será usado (default: o promovido)")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"❌ log inexistente: {args.log}")
        return 1

    art = art_mod.carregar(args.artefato) if args.artefato else art_mod.carregar()
    linhas, invalidas = ler(args.log)
    if not linhas:
        print(f"❌ nenhuma linha legível em {args.log}")
        return 1

    print(relatorio(analisar(linhas, art), art, invalidas))
    return 0


if __name__ == "__main__":
    sys.exit(main())

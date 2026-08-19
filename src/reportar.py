"""
A leitura FINAL do conjunto de teste — Etapa 11.

    python -m src.reportar               # lê o registro; só toca o teste se ele não existir
    python -m src.reportar --reexecutar  # recalcula e CONFERE contra o registro

⛔ **O conjunto de teste é tocado uma única vez, aqui, com o modelo já escolhido.**
Toda a seleção — features (Etapa 4-5), algoritmo (6), hiperparâmetros (7), arquitetura
da rede (8) — foi decidida na validação, e o teste ficou intocado desde a partição da
Etapa 2. O motivo é que o `max` sobre medições ruidosas captura ruído favorável: um
número usado para escolher deixa de ser estimativa imparcial daquilo que ele escolheu.

Duas consequências de projeto que este módulo implementa em vez de prometer:

1. **A disciplina virou mecanismo.** O resultado é gravado em `docs/resultado-teste-final.json`
   com data, commit e os hashes do artefato e do dataset. Existindo o registro, a execução
   seguinte **não recalcula nada**: ela imprime o que foi medido. Tocar o teste de novo para
   "ver como ficou" com outro modelo é exatamente o que corrói o instrumento, e o custo de
   fazê-lo tem de ser um `--reexecutar` explícito que **confere** em vez de substituir.

2. **Mede-se o ARTEFATO PROMOVIDO, não um modelo retreinado com o mesmo código.** É o objeto
   que a API serve, identificado por sha256 — a mesma distinção da Etapa 9c: comparar
   modelos treinando na hora é legítimo, *servir* não é, e *reportar* segue o que serve.

O número sai acompanhado do **piso** (a prevalência, que é o valor que a PR-AUC de um modelo
sem informação atinge) e de um **IC de bootstrap**: um ponto sem incerteza sobre 1.409 linhas
sugere uma precisão que o tamanho da amostra não tem.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display: o script roda em terminal e no CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

from src import artefato, config, data, evaluate
from src.train import commit_hash

REGISTRO = config.RAIZ / "docs" / "resultado-teste-final.json"
FIGURAS = config.RAIZ / "docs" / "figuras"

# Custos da Etapa 0, em reais por cliente. Repetidos aqui porque as duas estratégias
# triviais (não abordar ninguém / abordar a base inteira) não passam por `curva_custo`.
CUSTO_FN = 194
CUSTO_FP = 62

N_BOOTSTRAP = 2000


def ic_bootstrap(y, p, n: int = N_BOOTSTRAP, seed: int = config.SEED) -> tuple[float, float]:
    """IC95 percentil da PR-AUC por reamostragem com reposição.

    Responde "0,66 mais ou menos quanto?" — e a resposta importa porque seis candidatos
    couberam em 0,0057 de PR-AUC na validação. Se o IC de um único conjunto de teste é
    mais largo que a distância entre eles, a impossibilidade de desempatá-los deixa de ser
    uma escolha de protocolo e passa a ser uma propriedade do tamanho da amostra.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    valores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        # Reamostra que caia numa classe só não tem PR-AUC definida; com prevalência
        # de 26,5% e n=1.409 isso não acontece, mas o guarda evita um NaN silencioso.
        if y[idx].sum() in (0, len(idx)):
            valores[i] = np.nan
            continue
        valores[i] = average_precision_score(y[idx], p[idx])
    return float(np.nanpercentile(valores, 2.5)), float(np.nanpercentile(valores, 97.5))


def custos_triviais(y) -> dict[str, float]:
    """As duas estratégias sem modelo, em reais — o piso e o teto da conta de negócio.

    A segunda é a que mata o argumento de "otimizar recall puro": abordar a base inteira
    captura 100% dos churners e **também** é caro. Sem ela, dizer que a solução degenerada
    seria ruim é afirmação; com ela, é número.
    """
    y = np.asarray(y)
    churners = int(y.sum())
    return {
        "nao_fazer_nada_brl": float(churners * CUSTO_FN),
        "abordar_todos_brl": float((len(y) - churners) * CUSTO_FP),
    }


def medir_teste(a: artefato.Artefato, dados: data.Dados) -> dict:
    """A medição única. Recebe o artefato promovido — não treina nada."""
    p_teste = a.pipeline.predict_proba(dados.teste.X)[:, 1]
    p_val = a.pipeline.predict_proba(dados.validacao.X)[:, 1]

    # ⛔ O limiar é o DE OPERAÇÃO, que veio dentro do artefato (derivado do custo 3:1 na
    # validação). Re-derivá-lo no teste seria usar o teste para decidir — a única coisa
    # que esta etapa existe para não fazer. O `.predict()` do sklearn aplicaria 0,5, que
    # custaria R$ 7.546 por ciclo e 83 churners na validação: medido na Etapa 9i.
    limiar = a.limiar
    m_teste = evaluate.avaliar(dados.teste.y, p_teste, limiar=limiar)
    m_val = evaluate.avaliar(dados.validacao.y, p_val, limiar=limiar)
    lo, hi = ic_bootstrap(dados.teste.y, p_teste)

    return {
        "limiar_operacao": limiar,
        "n_teste": len(dados.teste),
        "prevalencia_teste": m_teste["prevalencia"],
        "teste": m_teste,
        "validacao": m_val,
        "pr_auc_ic95": [lo, hi],
        "gap_val_teste": m_val["pr_auc"] - m_teste["pr_auc"],
        "custos_triviais": custos_triviais(dados.teste.y),
    }


def curva_ganho(y, p, destino: Path) -> dict[str, list[float]]:
    """A curva de ganho cumulativo — três linhas, e as três são necessárias.

    Eixo x: fração da base contatada, em ordem decrescente de risco. Eixo y: fração dos
    churners capturada. As linhas:

      1. o modelo;
      2. a diagonal do acaso (contatar 10% ao acaso pega 10% dos churners);
      3. o TETO ESTRUTURAL `k / prevalência` — nos k% do topo cabem no máximo k×N clientes,
         então nenhum ranqueador pode capturar mais que isso. Sem a terceira linha, um
         ranking a 73,8% do máximo possível parece fraco, e é o gráfico que mente.

    É a métrica de negócio em forma contínua: "ligando para 10% da base, a campanha alcança
    X% de quem ia cancelar" é uma frase que não precisa de tradução.
    """
    y = np.asarray(y)
    ordem = np.argsort(np.asarray(p))[::-1]
    capturados = np.cumsum(y[ordem]) / y.sum()
    fracao = np.arange(1, len(y) + 1) / len(y)
    prevalencia = y.mean()
    teto = np.minimum(fracao / prevalencia, 1.0)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(fracao * 100, teto * 100, color="#9aa0a6", ls="--", lw=1.4,
            label=f"teto estrutural (k / prevalência = {prevalencia:.1%})")
    ax.plot(fracao * 100, capturados * 100, color="#1a73e8", lw=2.2,
            label="modelo (LogReg, 13 features)")
    ax.plot([0, 100], [0, 100], color="#c0c0c0", ls=":", lw=1.4, label="acaso")

    for k in config.KS_OPERACIONAIS:
        i = int(np.ceil(k * len(y))) - 1
        ax.plot(k * 100, capturados[i] * 100, "o", color="#1a73e8", ms=6)
        ax.annotate(
            f"  {capturados[i]:.1%} dos churners\n  contatando {k:.0%} da base",
            (k * 100, capturados[i] * 100), fontsize=8.5, va="top", color="#1a73e8",
        )

    ax.set_xlabel("% da base contatada (ordenada por risco decrescente)")
    ax.set_ylabel("% dos churners capturados")
    ax.set_title("Curva de ganho cumulativo — conjunto de teste (n=1.409, tocado uma vez)",
                 fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return {"fracao": fracao.tolist(), "capturados": capturados.tolist()}


def histograma_probabilidades(y, p, destino: Path) -> None:
    """As probabilidades preditas, separadas pela classe verdadeira.

    É o Brier em forma de figura, e explica visualmente por que o limiar ótimo difere por
    modelo: o que se corta é a sobreposição entre as duas distribuições, e onde cortar
    depende de quanto custa cada lado do erro.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, alpha=0.65, color="#5f6368", label="ficou (y=0)")
    ax.hist(p[y == 1], bins=bins, alpha=0.75, color="#d93025", label="cancelou (y=1)")
    ax.axvline(0.29, color="#1a73e8", lw=1.8, ls="--", label="limiar de operação (0,29)")
    ax.axvline(0.5, color="#9aa0a6", lw=1.2, ls=":", label="0,50 (o que .predict() usaria)")
    ax.set_xlabel("P(churn) predita")
    ax.set_ylabel("clientes")
    ax.set_title("Probabilidades preditas por classe verdadeira — teste", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def relatorio(res: dict) -> str:
    """O texto que vai para a documentação — métrica nunca aparece sozinha."""
    t, v = res["teste"], res["validacao"]
    lo, hi = res["pr_auc_ic95"]
    tri = res["custos_triviais"]
    piso = res["prevalencia_teste"]
    linhas = [
        "=" * 78,
        f"CONJUNTO DE TESTE — leitura única  ({res['medido_em'][:10]}, commit {res['commit']})",
        "=" * 78,
        f"artefato        : {res['artefato_sha256'][:16]}…  v{res['versao_modelo']}",
        f"n               : {res['n_teste']}   prevalência {piso:.2%}",
        "",
        f"PR-AUC          : {t['pr_auc']:.4f}   (piso {piso:.4f} = modelo sem informação; "
        f"IC95 [{lo:.4f}; {hi:.4f}])",
        f"                  validação {v['pr_auc']:.4f}  ⇒  gap val−teste "
        f"{res['gap_val_teste']:+.4f}",
        f"ROC-AUC         : {t['roc_auc']:.4f}   (piso 0,5000)",
        f"Brier           : {t['brier']:.4f}   (validação {v['brier']:.4f})",
        "",
        f"recall@10%      : {t['recall_at_10']:.3f}  "
        f"({t['recall_at_10_pct_teto']:.1%} do teto estrutural, lift {t['lift_at_10']:.2f}×)",
        f"recall@20%      : {t['recall_at_20']:.3f}  "
        f"({t['recall_at_20_pct_teto']:.1%} do teto estrutural, lift {t['lift_at_20']:.2f}×)",
        "",
        f"no limiar de operação {res['limiar_operacao']:.2f}:",
        f"  precisão {t['precision']:.4f} · recall {t['recall']:.4f} · F1 {t['f1']:.4f}",
        f"  TP {t['tp']:>4}  FP {t['fp']:>4}  FN {t['fn']:>4}  TN {t['tn']:>4}",
        "",
        f"custo do erro   : R$ {t['custo_erro_brl']:>9,.0f}   (o modelo)",
        f"                  R$ {tri['nao_fazer_nada_brl']:>9,.0f}   (não abordar ninguém)",
        f"                  R$ {tri['abordar_todos_brl']:>9,.0f}   (abordar a base inteira)",
        "=" * 78,
    ]
    return "\n".join(linhas)


def executar() -> dict:
    """Toca o teste, gera as figuras e devolve o registro completo."""
    a = artefato.carregar()
    dados = data.dividir()
    res = medir_teste(a, dados)
    p_teste = a.pipeline.predict_proba(dados.teste.X)[:, 1]
    curva_ganho(dados.teste.y, p_teste, FIGURAS / "curva-ganho.png")
    histograma_probabilidades(dados.teste.y, p_teste, FIGURAS / "probabilidades-por-classe.png")
    res |= {
        "medido_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit_hash(),
        "artefato_sha256": a.sha256,
        "versao_modelo": a.versao,
        "dataset_sha256": dados.sha256,
    }
    return res


def conferir(novo: dict, antigo: dict, tol: float = 1e-9) -> list[str]:
    """Compara a re-execução com o registro. Divergência aqui é notícia, não ruído.

    O artefato é fixo e a partição é determinística, então os dois números têm de bater
    até o último dígito na mesma máquina. A tolerância existe para o caso do §8 — outro
    sistema operacional muda o último bit do BLAS sem mudar decisão nenhuma.
    """
    problemas = []
    for chave in ("pr_auc", "brier", "recall_at_10", "custo_erro_brl"):
        a_, b_ = novo["teste"][chave], antigo["teste"][chave]
        if abs(a_ - b_) > tol:
            problemas.append(f"{chave}: {b_!r} (registro) × {a_!r} (agora)")
    if novo["artefato_sha256"] != antigo["artefato_sha256"]:
        problemas.append(
            f"artefato: {antigo['artefato_sha256'][:16]}… (registro) × "
            f"{novo['artefato_sha256'][:16]}… (agora) — é OUTRO modelo"
        )
    return problemas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reexecutar", action="store_true",
        help="recalcula e CONFERE contra o registro (não o substitui)",
    )
    args = ap.parse_args(argv)

    if REGISTRO.exists() and not args.reexecutar:
        antigo = json.loads(REGISTRO.read_text())
        print(relatorio(antigo))
        print("\nO teste já foi tocado. Este é o registro, não uma nova medição.")
        print("Para conferir que ele reproduz: python -m src.reportar --reexecutar")
        return 0

    res = executar()

    if REGISTRO.exists():
        antigo = json.loads(REGISTRO.read_text())
        problemas = conferir(res, antigo)
        print(relatorio(antigo))
        if problemas:
            print("\n❌ A re-execução NÃO reproduz o registro:")
            for p_ in problemas:
                print(f"   • {p_}")
            print("   O registro NÃO foi substituído. Se a mudança é intencional (modelo novo),")
            print("   isso é uma promoção — e promoção é um commit próprio, com justificativa.")
            return 1
        print("\n✅ A re-execução reproduz o registro em todos os eixos conferidos.")
        return 0

    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    REGISTRO.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(relatorio(res))
    print(f"\n📌 Registro gravado em {REGISTRO.relative_to(config.RAIZ)}")
    print(f"   Figuras em {FIGURAS.relative_to(config.RAIZ)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

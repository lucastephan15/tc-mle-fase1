"""
Promoção do campeão — Etapa 9c.

    python -m src.promover          (ou `make promover`)

Treina o campeão **pelo mesmo caminho que o gate do CI mede** (`gate.treinar_campeao`),
aplica o gate de dois eixos e só então grava o artefato que a API vai servir.

Três regras que o script implementa, e o motivo de cada uma:

1. ⛔ **Nada é promovido sem passar no gate.** Um script de promoção que grava
   incondicionalmente é uma máquina de pôr modelo pior em produção mais rápido —
   é literalmente o que a automação sem gate faz de melhor.

2. 🔑 **O limiar de operação é derivado aqui e viaja DENTRO do artefato.** Ele é
   propriedade do modelo servido (a distribuição de probabilidades muda quando o
   modelo muda), e escrevê-lo como literal na API criaria duas cópias obrigadas a
   mudar juntas — a que ficasse para trás não daria erro, só cortaria a fila no
   lugar errado. Medido no campeão: `.predict()` (0,50 implícito) custa
   R$ 7.546 por ciclo a mais e 83 churners que o limiar de 0,29 pega.

3. ✅ **Round-trip verificado no ato.** Depois de gravar, recarrega e compara as
   1.409 probabilidades da validação **bit a bit**. Promover sem reler é a mesma
   fé que o `/health` que só prova que o processo subiu: o objeto em memória
   passou, e o que está no disco é outro objeto.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np

from src import artefato, config, data, gate, referencia
from src.train import commit_hash


def montar_metadados(dados: data.Dados, metricas: dict, custo: dict,
                     features: list[str]) -> dict:
    """Tudo que a API precisa declarar sobre o que está servindo.

    As features saem do PIPELINE (`feature_names_in_`), não de `config.FEATURES`:
    o config acompanha o código, o artefato acompanha o modelo servido, e os
    dois só coincidem enquanto ninguém promove um modelo treinado com outro
    config. A fonte de verdade do contrato é o artefato.

    🔑 **A `referencia` entra aqui pelo mesmo argumento do limiar** (Etapa 10a-2):
    a distribuição do treino é propriedade DO MODELO SERVIDO, não do repositório
    — muda quando o modelo muda. Congelada junto, ela viaja com o artefato até o
    container, e é isso que permite calcular drift **sem o dataset**: o dado
    bruto é LGPD, não entra na imagem, e uma referência que exigisse relê-lo não
    rodaria onde o drift acontece.

    ⚠️ Calculada sobre `dados.treino.X` e só sobre ele: validação e teste julgam
    o modelo, e misturá-los ao baseline confundiria "o que o modelo viu" com "o
    que o mediu".
    """
    return {
        "versao_modelo": config.VERSAO_MODELO,
        "modelo": config.GATE_MODELO_REFERENCIA,
        "features": features,
        "limiar_operacao": float(custo["limiar_otimo"]),
        "metricas_validacao": {
            "pr_auc": float(metricas["pr_auc"]),
            "brier": float(metricas["brier"]),
            "recall_at_10": float(metricas["recall_at_10"]),
            "custo_erro_brl": float(metricas["custo_erro_brl"]),
        },
        # Linhagem: dados + código + (em artefato.salvar) ambiente.
        "dataset_sha256": dados.sha256,
        "commit": commit_hash(),
        "seed": config.SEED,
        "promovido_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "avaliado_em": "validacao",  # o teste segue intocado
        # Etapa 10a-2 — o denominador do drift. Sem isto não existe detecção,
        # só um gráfico comparando nada.
        "referencia": referencia.calcular(
            dados.treino.X, config.NUM_ZERO + config.NUM, config.CAT,
        ),
    }


def main() -> int:
    np.random.seed(config.SEED)
    dados = data.dividir()

    # Mesmo caminho do gate: o objeto promovido é O objeto medido, não um irmão.
    pipe = gate.treinar_campeao(dados)
    metricas, custo = gate.medir(dados, pipe=pipe)
    passou, motivo = gate.aprovado(metricas["pr_auc"], metricas["brier"])

    print(f"Promoção do campeão — {config.GATE_MODELO_REFERENCIA} "
          f"v{config.VERSAO_MODELO}")
    print(f"  PR-AUC (val)   : {metricas['pr_auc']:.4f} "
          f"(piso {config.GATE_PR_AUC_MIN:.4f})")
    print(f"  Brier (val)    : {metricas['brier']:.4f} "
          f"(teto {config.GATE_BRIER_MAX:.4f})")
    print(f"  limiar operação: {custo['limiar_otimo']:.2f} "
          f"(custo R$ {custo['custo_otimo_brl']:,.0f})")

    if not passou:
        print(f"\n❌ NADA PROMOVIDO: {motivo}.")
        print("   O artefato anterior continua sendo o servido — que é o "
              "comportamento certo:\n   promoção que grava mesmo reprovada não "
              "é promoção, é sobrescrita.")
        return 1

    metadados = montar_metadados(
        dados, metricas, custo, features=list(pipe.feature_names_in_),
    )
    sha = artefato.salvar(pipe, metadados)

    # Round-trip: o que está no DISCO prediz igual ao que está em memória?
    recarregado = artefato.carregar()
    p_memoria = pipe.predict_proba(dados.validacao.X)[:, 1]
    p_disco = recarregado.pipeline.predict_proba(dados.validacao.X)[:, 1]
    if not np.array_equal(p_memoria, p_disco):
        maior = float(np.abs(p_memoria - p_disco).max())
        print(f"\n❌ ROUND-TRIP FALHOU: maior diferença {maior:.3e} em "
              f"{len(p_disco)} predições. O artefato NÃO é o modelo avaliado.")
        return 1

    # E os METADADOS sobreviveram? O round-trip acima compara probabilidades, e
    # probabilidade não passa por `referencia`, `limiar_operacao` nem `features`
    # — ou seja, a checagem que existia deixava passar metadado corrompido (item
    # 109 do revisita). Desde a Etapa 10a-2 isso deixou de ser teórico: o
    # baseline de drift é um dict aninhado com ~2,5 KB que **nada** no caminho
    # feliz da API lê, então um erro ali só apareceria na primeira análise de
    # drift, semanas depois, como número errado — não como erro.
    if recarregado.metadados != {**metadados, "versoes": artefato.versoes_do_ambiente()}:
        print("\n❌ ROUND-TRIP DE METADADOS FALHOU: o que foi gravado não é o "
              "que foi montado. O artefato prediz igual e DECLARA outra coisa.")
        return 1

    print(f"\n✅ PROMOVIDO em {config.ARTEFATO.relative_to(config.RAIZ)}")
    ref = recarregado.referencia
    print(f"   sha256 {sha[:16]}…  ·  round-trip bit a bit em "
          f"{len(p_disco)} predições: idêntico")
    print(f"   baseline de drift: {len(ref['numericas'])} numéricas + "
          f"{len(ref['categoricas'])} categóricas sobre n={ref['n']} do treino")
    print("\n" + artefato.descrever(recarregado))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Gate de promoção — Etapa 9.5.

    python -m src.gate

Treina o modelo de referência e **falha com código de saída 1** se ele ficar
abaixo do piso registrado em `config.GATE_PR_AUC_MIN`. É o passo que separa um
CI que testa código de um CI que protege o *modelo*: sem ele, a automação vira
uma máquina de colocar modelo pior em produção mais rápido.

⛔ Mede na **VALIDAÇÃO**. O conjunto de teste não é tocado aqui — decidir
promoção olhando o teste a cada push o converte em validação, e ele deixa de
estimar generalização. Divergência deliberada do enunciado da Aula 08, registrada
no decision log.

Por que treinar em vez de ler uma métrica salva: o gate precisa provar que o
pipeline **inteiro** ainda roda de ponta a ponta nesta máquina, com estas
versões de biblioteca, a partir do dado bruto. Uma métrica lida de arquivo
provaria apenas que o arquivo existe.
"""

from __future__ import annotations

import sys

import numpy as np

from src import config, data, evaluate
from src.preprocess import construir_pipeline
from src.train import construir_modelo


def main() -> int:
    np.random.seed(config.SEED)
    dados = data.dividir()

    modelo, escalonar = construir_modelo(config.GATE_MODELO_REFERENCIA, None)
    pipe = construir_pipeline(modelo, escalonar=escalonar)
    pipe.fit(dados.treino.X, dados.treino.y)
    p = pipe.predict_proba(dados.validacao.X)[:, 1]
    # O limiar tem de ser o DE OPERAÇÃO (derivado do custo 3:1), não o 0,5 que a
    # função usa como referência: reportar custo a 0,5 daria um número que não
    # corresponde a nenhuma decisão que este projeto toma.
    custo = evaluate.curva_custo(dados.validacao.y, p)
    m = evaluate.avaliar(dados.validacao.y, p, limiar=custo["limiar_otimo"])

    piso = config.GATE_PR_AUC_MIN
    obtido = m["pr_auc"]
    passou = obtido >= piso

    print(f"Gate de promoção — modelo de referência: {config.GATE_MODELO_REFERENCIA}")
    print(f"  dataset sha256 : {dados.sha256[:16]}…")
    print(f"  n_features     : {len(config.FEATURES)}")
    print(f"  PR-AUC (val)   : {obtido:.4f}")
    print(f"  piso exigido   : {piso:.4f}")
    print(f"  margem         : {obtido - piso:+.4f}")
    print(f"  recall@10%     : {m['recall_at_10']:.3f} "
          f"({m['recall_at_10_pct_teto']:.1%} do teto estrutural)")
    print(f"  custo do erro  : R$ {m['custo_erro_brl']:,.0f} "
          f"(no limiar de operação {custo['limiar_otimo']:.2f})")

    if not passou:
        print(f"\n❌ GATE REPROVADO: {obtido:.4f} < {piso:.4f}. Nada é promovido.")
        print("   Se a queda for esperada (mudou feature, dado ou modelo), o piso em")
        print("   config.GATE_PR_AUC_MIN precisa ser revisto NUM COMMIT PRÓPRIO, com")
        print("   justificativa — não afrouxado de passagem para o CI ficar verde.")
        return 1

    print(f"\n✅ GATE APROVADO: {obtido:.4f} >= {piso:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

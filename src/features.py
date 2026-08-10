"""
Feature engineering — Etapa 4.

⚠️ A regra que este módulo existe para cumprir: a criação de features é um PASSO
DO PIPELINE, não uma célula de notebook. Se a lógica não está dentro do objeto
serializado, ela não viaja no artefato — o JSON de produção chega com os campos
crus, a coluna derivada não existe, e a API ou quebra ou (pior) prediz com
feature faltando devolvendo 200 OK.

Por isso `adicionar_features` é uma função pura de DataFrame -> DataFrame,
embrulhada por um FunctionTransformer no primeiro passo do Pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Os 6 serviços de valor agregado. 'Phone Service' e 'Internet Service' ficam de
# fora de propósito: são o serviço-base, não adicionais.
SERVICOS_ADICIONAIS = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]

# Métodos de pagamento que debitam sozinhos. O resto exige uma ação do cliente
# todo mês — e toda ação é uma oportunidade de reconsiderar.
PAGAMENTO_AUTOMATICO = ["Bank transfer (automatic)", "Credit card (automatic)"]

# Cortes do binning de tenure, em meses. Vêm da curva medida na EDA, não de
# quantis: 0-6m concentra 52,9% de churn e 4a+ apenas 9,5%.
CORTES_TENURE = [-np.inf, 6, 12, 24, 48, np.inf]
ROTULOS_TENURE = ["0-6m", "6-12m", "1-2a", "2-4a", "4a+"]

NOVAS_NUM = ["n_servicos_adicionais", "charge_por_servico"]
NOVAS_CAT = ["tenure_faixa", "pagamento_automatico"]


def adicionar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva as features da Etapa 4. Pura: não altera o DataFrame recebido."""
    X = df.copy()

    # --- custo de troca -----------------------------------------------------
    # Hipótese: cada serviço adicional é mais um fio prendendo o cliente. A EDA
    # mostrou o efeito isolado de cada um (sem Online Security 41,8% x com
    # 14,6%); a contagem captura o efeito CONJUNTO, que 6 dummies separadas só
    # representam de forma aditiva.
    # "No" e "No internet service" contam como não-contratado — o segundo é uma
    # categoria legítima (cliente sem internet), não um nulo.
    X["n_servicos_adicionais"] = sum(
        (X[c] == "Yes").astype(int) for c in SERVICOS_ADICIONAIS
    )

    # --- percepção de custo-benefício ---------------------------------------
    # Hipótese: o que incomoda não é o valor absoluto, é pagar caro por pouco.
    # O +1 no denominador não é truque estético: sem ele, os clientes com zero
    # serviços adicionais (que EXISTEM e são justamente os de maior risco)
    # produziriam divisão por zero. É a mesma armadilha do inf que a Etapa 1
    # documentou com 'Tenure Months', resolvida na origem.
    X["charge_por_servico"] = X["Monthly Charges"] / (1 + X["n_servicos_adicionais"])

    # --- não-linearidade do tempo de casa -----------------------------------
    # Hipótese: o risco não cai de forma constante a cada mês — despenca no
    # primeiro semestre e depois estabiliza. Binning entrega essa curva de mão
    # beijada para LogReg e MLP; é redundante para árvores, que acham os cortes
    # sozinhas. Por isso a feature é opcional no pipeline (ver preprocess.py).
    X["tenure_faixa"] = pd.cut(
        X["Tenure Months"], bins=CORTES_TENURE, labels=ROTULOS_TENURE
    ).astype(str)

    # --- fricção de pagamento -----------------------------------------------
    # Hipótese: débito automático é inércia a favor da empresa; pagamento manual
    # transforma cada mês numa decisão consciente de continuar.
    X["pagamento_automatico"] = np.where(
        X["Payment Method"].isin(PAGAMENTO_AUTOMATICO), "Sim", "Nao"
    )

    return X

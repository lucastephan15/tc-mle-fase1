"""
Feature engineering — Etapa 4.

⚠️ A regra que este módulo existe para cumprir: a criação de features é um PASSO
DO PIPELINE, não uma célula de notebook. Se a lógica não está dentro do objeto
serializado, ela não viaja no artefato — o JSON de produção chega com os campos
crus, a coluna derivada não existe, e a API ou quebra ou (pior) prediz com
feature faltando devolvendo 200 OK.

Por isso `adicionar_features` é uma função pura de DataFrame -> DataFrame,
embrulhada por um FunctionTransformer no primeiro passo do Pipeline.

────────────────────────────────────────────────────────────────────────────
HISTÓRICO — 4 features viraram 1, por duas razões independentes:

  Etapa 4 mediu as quatro por ablação com CV e NENHUMA agregou ganho
  (+0,0003 na LogReg contra desvio de 0,0280; −0,0067 na Random Forest).

  Etapa 5 então removeu 6 colunas do modelo, e isso deixou três delas ÓRFÃS:
    · `n_servicos_adicionais`  — 4 dos 6 serviços que ela somava saíram
    · `charge_por_servico`     — dependia da contagem acima
    · `pagamento_automatico`   — 'Payment Method' saiu do modelo

  Sobrou `tenure_faixa`, a única cujo insumo (`Tenure Months`) permanece — e a
  única que não era combinação linear de colunas já presentes.

Registrado assim de propósito: uma decisão posterior pode invalidar código
anterior, e apagar o rastro esconderia o encadeamento. O detalhe está no
decision log §3 e §4.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cortes do binning de tenure, em meses. Vêm da curva medida na EDA, não de
# quantis: 0-6m concentra 52,9% de churn e 4a+ apenas 9,5%.
CORTES_TENURE = [-np.inf, 6, 12, 24, 48, np.inf]
ROTULOS_TENURE = ["0-6m", "6-12m", "1-2a", "2-4a", "4a+"]

NOVAS_NUM: list[str] = []
NOVAS_CAT = ["tenure_faixa"]


def adicionar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva as features da Etapa 4. Pura: não altera o DataFrame recebido."""
    X = df.copy()

    # --- não-linearidade do tempo de casa -----------------------------------
    # Hipótese: o risco não cai de forma constante a cada mês — despenca no
    # primeiro semestre e depois estabiliza. Binning entrega essa curva de mão
    # beijada para LogReg e MLP; é redundante para árvores, que acham os cortes
    # sozinhas. Por isso a feature é opcional no pipeline (ver preprocess.py) e
    # será reavaliada na Etapa 8, com o MLP.
    X["tenure_faixa"] = pd.cut(
        X["Tenure Months"], bins=CORTES_TENURE, labels=ROTULOS_TENURE
    ).astype(str)

    return X

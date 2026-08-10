"""
Pré-processamento — Etapa 2 · Data Preparation.

Tudo que transforma dado vive AQUI DENTRO, num único objeto serializável. A regra
que sustenta o módulo: se a transformação não está dentro do objeto que você
serializa, ela é uma bomba-relógio — o JSON de produção chega cru e a coluna
esperada não existe, ou pior, existe errada e a API responde 200 OK.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def construir_preprocessador(escalonar: bool = True) -> ColumnTransformer:
    """Monta o pré-processamento em três grupos de colunas.

    Os grupos existem porque o tratamento correto é diferente em cada um — não
    por organização estética.

    `escalonar=False` para modelos de árvore, a que a escala é indiferente.
    """
    # Grupo 1 — 'Total Charges'. O vazio significa "não houve ciclo de faturamento"
    # (os 11 casos têm tenure = 0), então 0 é o valor VERDADEIRO. Imputar a mediana
    # (~R$ 1.400) inventaria histórico de pagamento para quem nunca foi faturado, e
    # de forma plausível o bastante para ninguém notar.
    # Constante não aprende parâmetro dos dados -> não é fonte de leakage. Está aqui
    # por causa da OUTRA regra: precisa viajar dentro do artefato.
    passos_zero = [("imputar", SimpleImputer(strategy="constant", fill_value=0.0))]

    # Grupo 2 — demais numéricas. Não há nulo no dataset, mas a API vai receber
    # o que a API receber. A mediana é preferida à média por ser robusta a outlier.
    # Este SimpleImputer SIM aprende parâmetro -> por isso o fit é só no treino,
    # garantido por estar dentro do Pipeline.
    passos_num = [("imputar", SimpleImputer(strategy="median"))]

    if escalonar:
        passos_zero.append(("escalar", StandardScaler()))
        passos_num.append(("escalar", StandardScaler()))

    # Grupo 3 — categóricas.
    passos_cat = [
        ("imputar", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": categoria inédita vira um vetor de zeros em vez
        # de derrubar a API. drop=None de propósito, contra a recomendação usual de
        # drop_first: com drop, o vetor de zeros da categoria desconhecida COLIDE
        # com o vetor da categoria dropada — um Gender="Other" seria silenciosamente
        # tratado como "Female". A dummy variable trap que o drop evita é inofensiva
        # sob a regularização L2 padrão do sklearn; a colisão não é.
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("zero", Pipeline(passos_zero), config.NUM_ZERO),
            ("num", Pipeline(passos_num), config.NUM),
            ("cat", Pipeline(passos_cat), config.CAT),
        ],
        remainder="drop",  # nada entra por engano: o contrato é config.FEATURES
        verbose_feature_names_out=False,
    )


def construir_pipeline(modelo, escalonar: bool = True) -> Pipeline:
    """Encadeia pré-processamento e modelo num único objeto.

    É este objeto que vai para o joblib/MLflow — nunca só o classificador. Salvar
    apenas o modelo é o erro que produz predição errada com status 200 OK, porque
    a API acaba reconstruindo o pré-processamento "na mão" de um jeito ligeiramente
    diferente do treino (training-serving skew).
    """
    return Pipeline([
        ("preproc", construir_preprocessador(escalonar=escalonar)),
        ("modelo", modelo),
    ])

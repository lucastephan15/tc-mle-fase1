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
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src import config, features


def construir_preprocessador(
    escalonar: bool = True,
    novas: list[str] | None = None,
    excluir: list[str] | None = None,
    encoding: str = "onehot",
) -> ColumnTransformer:
    """Monta o pré-processamento em três grupos de colunas.

    Os grupos existem porque o tratamento correto é diferente em cada um — não
    por organização estética.

    `escalonar=False` para modelos de árvore, a que a escala é indiferente.

    `encoding="ordinal"` para modelos de árvore (Etapa 6). NÃO é uma correção do
    one-hot: é reconhecer que **o encoding certo depende da família do modelo**.
    Para a LogReg, ordinal seria erro grosseiro — ela leria `Contract=2` como
    "o dobro de `Contract=1`", impondo uma ordem e uma escala que não existem.
    Para a árvore é inócuo, porque ela só usa o valor para cortar (`<= 1.5`), e
    qualquer particionamento das categorias continua alcançável com splits
    sucessivos. O que a árvore GANHA é não ter a informação de uma coluna
    espalhada por k dummies — ver §5a da nota da M02-Aula 04.
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
    if encoding == "onehot":
        # handle_unknown="ignore": categoria inédita vira um vetor de zeros em vez
        # de derrubar a API. drop=None de propósito, contra a recomendação usual de
        # drop_first: com drop, o vetor de zeros da categoria desconhecida COLIDE
        # com o vetor da categoria dropada — um Gender="Other" seria silenciosamente
        # tratado como "Female". A dummy variable trap que o drop evita é inofensiva
        # sob a regularização L2 padrão do sklearn; a colisão não é.
        codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding == "ordinal":
        # Mesma proteção da linha acima, por outro mecanismo: a categoria inédita
        # vira -1 em vez de derrubar a API. Como todas as categorias conhecidas
        # recebem 0..k-1, o -1 fica FORA do intervalo e a árvore pode isolá-lo com
        # um único split — a inédita não é silenciosamente confundida com nenhuma
        # categoria vista, que é a mesma exigência que motivou o drop=None.
        codificador = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
        )
    else:
        raise ValueError(f"encoding desconhecido: {encoding}")

    passos_cat = [
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", codificador),
    ]

    # As features da Etapa 4 entram nos mesmos grupos das originais — só a lista
    # de colunas muda. Receber a lista (em vez de um booleano) é o que permite o
    # estudo de ablação: ligar e desligar UMA de cada vez para medir a
    # contribuição marginal de cada uma.
    novas = novas or []
    # `excluir` remove a feature ORIGINAL inteira — não uma dummy solta. É a
    # diferença entre reduzir a dimensão do modelo e reduzir o custo de coleta,
    # validação e monitoramento em produção (ver decision log §4).
    fora = set(excluir or [])
    num = [c for c in config.NUM + [x for x in features.NOVAS_NUM if x in novas] if c not in fora]
    cat = [c for c in config.CAT + [x for x in features.NOVAS_CAT if x in novas] if c not in fora]
    zero = [c for c in config.NUM_ZERO if c not in fora]

    return ColumnTransformer(
        transformers=[
            ("zero", Pipeline(passos_zero), zero),
            ("num", Pipeline(passos_num), num),
            ("cat", Pipeline(passos_cat), cat),
        ],
        remainder="drop",  # nada entra por engano: o contrato é config.FEATURES
        verbose_feature_names_out=False,
    )


def mascara_categorica(
    novas: list[str] | None = None,
    excluir: list[str] | None = None,
) -> list[bool]:
    """Quais colunas da saída do preprocessador ORDINAL são categóricas.

    Serve ao `categorical_features` do `HistGradientBoostingClassifier`, que
    trata categórica de forma nativa (particiona o conjunto de níveis num só
    split) em vez de tratá-la como número ordenado.

    Vive aqui, e não em quem chama, porque depende da ORDEM em que o
    `ColumnTransformer` concatena os três grupos — zero, num, cat. Se essa ordem
    mudar lá em cima, é aqui que tem de mudar junto, e não num arquivo distante.

    ⚠️ Só vale para `encoding="ordinal"`: com one-hot, cada categórica vira k
    colunas e a contagem não fecha.
    """
    novas = novas or []
    fora = set(excluir or [])
    n_zero = len([c for c in config.NUM_ZERO if c not in fora])
    n_num = len([c for c in config.NUM + [x for x in features.NOVAS_NUM if x in novas]
                 if c not in fora])
    n_cat = len([c for c in config.CAT + [x for x in features.NOVAS_CAT if x in novas]
                 if c not in fora])
    return [False] * (n_zero + n_num) + [True] * n_cat


def construir_pipeline(
    modelo,
    escalonar: bool = True,
    novas: list[str] | None = None,
    excluir: list[str] | None = None,
    seletor=None,
    encoding: str = "onehot",
) -> Pipeline:
    """Encadeia pré-processamento e modelo num único objeto.

    É este objeto que vai para o joblib/MLflow — nunca só o classificador. Salvar
    apenas o modelo é o erro que produz predição errada com status 200 OK, porque
    a API acaba reconstruindo o pré-processamento "na mão" de um jeito ligeiramente
    diferente do treino (training-serving skew).
    """
    passos = []
    if novas:
        # PRIMEIRO passo do Pipeline, antes de qualquer transformação. É isto que
        # faz a feature engineering viajar dentro do joblib/MLflow: a API recebe
        # o JSON cru e o próprio artefato deriva as colunas, exatamente como no
        # treino. Sem isso, alguém teria de reimplementar a lógica dentro da API
        # — e "quase igual" é o bastante para produzir predição errada com 200 OK.
        passos.append(("features", FunctionTransformer(
            features.adicionar_features, validate=False, feature_names_out=None,
        )))
    passos.append(("preproc", construir_preprocessador(
        escalonar=escalonar, novas=novas, excluir=excluir, encoding=encoding)))
    if seletor is not None:
        # ⛔ O seletor entra COMO PASSO DO PIPELINE, nunca aplicado antes ao
        # dataset inteiro. Rodar SelectKBest fora daqui é o exemplo canônico de
        # leakage: a escolha das colunas passa a ser informada pelos dados de
        # teste, a métrica infla, e o gap treino-teste continua bonito porque os
        # dois lados foram contaminados igual.
        passos.append(("selecao", seletor))
    passos.append(("modelo", modelo))
    return Pipeline(passos)

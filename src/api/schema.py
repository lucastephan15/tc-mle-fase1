"""
O contrato de entrada e saída — Etapa 9d.

🔑 **A fonte de verdade do contrato é o ARTEFATO, não o `config.py`.** O config
acompanha o *código*; o artefato acompanha o *modelo servido*, e os dois só
coincidem enquanto ninguém promover um modelo treinado com outro config. Por
isso os 13 nomes, a ordem, os 25 valores categóricos permitidos e o número de
campos são **lidos do pipeline**, não digitados aqui.

O que este módulo existe para impedir (medido em 17/08/2026 contra o campeão
real, com validação de esquema no lugar de validação de domínio — os quatro
casos abaixo respondiam **HTTP 200** e prediziam normalmente):

    Contract = "Vitalicio"       categoria inexistente  -> 153 clientes (10,9%)
                                                            trocam de lado
    Tenure Months  = -999        P(churn) = 1,0000 nas 1.409 linhas
                                 -> 878 clientes (62,3%) cruzam o limiar
    Monthly Charges = -999       P(churn) = 0,0000 nas 1.409 linhas
    Total Charges  = 1e9         P(churn) = 1,0000 nas 1.409 linhas

As 10 categóricas aceitavam lixo em silêncio porque `handle_unknown="ignore"`
transforma valor desconhecido em **linha de zeros** — que para o modelo
significa *"a categoria de referência"*, não *"não sei"*. Correto no treino,
perigoso na inferência.

⚠️ `-999` não é hipótese acadêmica: é a sentinela de nulo mais comum em sistema
legado, que é exatamente quem consome uma API interna de churn.

O custo de fechar isso: **0,002 ms por requisição** — 0,1% de uma requisição de
1,815 ms, 800× menos que o `predict_proba` que ele protege. Não há trade-off a
ponderar.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model
from sklearn.pipeline import Pipeline

# --- Faixas numéricas: DECISÃO, não medição --------------------------------
#
# O treino dá Tenure 0–72, Monthly 18,40–118,75 e Total 18,85–8.684,80. Colar o
# `le` no máximo observado rejeitaria um cliente legítimo de 73 meses — o alvo
# aqui é `-999` e `1e9`, não a cauda da distribuição. Por isso os limites são de
# NEGÓCIO, folgados, e estão escritos onde alguém possa discordar deles.
#
# Toda coluna numérica do artefato tem de aparecer aqui: se uma nova entrar sem
# faixa declarada, `tests/test_api.py::test_toda_numerica_tem_faixa_declarada`
# falha. Sem esse teste, a coluna nova passaria sem limite nenhum e em silêncio —
# que é o modo de falha que este módulo inteiro existe para fechar.
FAIXAS: dict[str, tuple[float, float]] = {
    "Tenure Months": (0.0, 120.0),      # 10 anos de contrato
    "Monthly Charges": (0.0, 200.0),    # ~1,7× o maior plano visto
    "Total Charges": (0.0, 12_000.0),   # ~1,4× o maior acumulado visto
}

# Teto do lote em `/predict-batch`. Existe porque uma lista sem limite é o vetor
# de negação de serviço: o trabalho é síncrono e segura um worker do threadpool
# do começo ao fim. 5.000 linhas custam ~12 ms de `predict_proba` (medido: 1.409
# em 2,853 ms, ~2,0 µs por linha marginal) e cobrem a carteira inteira do Telco
# (7.043) em duas chamadas.
MAX_LOTE = 5_000


def identificador(coluna: str) -> str:
    """Nome de coluna -> nome de campo Python. `Tenure Months` -> `tenure_months`.

    Necessário, não estético: **9 das 13 features têm espaço no nome** e só 4
    são identificadores Python válidos. O nome real da coluna vira `alias`, e é
    ele que o cliente HTTP envia — o campo Python é detalhe interno.
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", coluna).strip("_").lower()


def contrato_do_pipeline(
    pipeline: Pipeline,
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Extrai do artefato (a) os nomes na ordem, (b) as categorias e (c) quem
    aceita vazio.

    Lê o `ColumnTransformer` em vez de `config`: é o objeto que vai receber o
    DataFrame, e é a discordância entre ele e o schema que produziria **500 no
    ColumnTransformer** em vez de 422 na validação — erro no lugar errado, com
    dado real, em produção.

    🚨 **O terceiro elemento existe por um defeito que só o container revelou.**
    O grupo `zero` do pré-processamento são as colunas cujo vazio é *medição
    verdadeira*: no Telco, `Total Charges` vem em branco para quem ainda não teve
    ciclo de faturamento — **11 clientes, todos com `Tenure Months = 0`**. A
    Etapa 2 decidiu tratá-los imputando 0 (e não a mediana), e o pipeline faz
    isso corretamente. O schema, não: como todo campo numérico era `float`
    obrigatório com `le`, esses clientes recebiam **422** — o cliente do primeiro
    mês, que é exatamente a população que uma campanha de retenção mais quer
    pontuar.

    🔑 *O contrato ficou mais estreito que o pipeline que ele protege.* É o erro
    simétrico do que este módulo foi escrito para impedir: lá o risco era aceitar
    lixo, aqui é recusar dado legítimo — e os dois têm a mesma raiz, o contrato
    escrito à parte do objeto que ele descreve. A correção mantém a regra: quem
    decide é o artefato.

    ⚠️ E o vazio é aceito **só** neste grupo, embora o grupo `num` também tenha
    imputador (mediana). Não é inconsistência: no grupo `zero` o vazio *significa*
    alguma coisa e o valor imputado é essa coisa; em `Tenure Months`, um vazio é
    dado faltando do integrador, e imputar a mediana do treino em silêncio seria
    transformar uma falha de integração em predição plausível. O pipeline sabe
    imputar as duas; a API só aceita o vazio onde ele quer dizer algo.
    """
    features = list(pipeline.feature_names_in_)
    ct = pipeline.named_steps["preproc"]
    categorias: dict[str, list[str]] = {}
    aceitam_vazio: list[str] = []
    for nome, _, colunas in ct.transformers_:
        if nome == "zero":
            aceitam_vazio.extend(colunas)
        if nome != "cat":
            continue
        ohe = ct.named_transformers_["cat"].named_steps["codificar"]
        for coluna, valores in zip(colunas, ohe.categories_, strict=True):
            categorias[coluna] = [str(v) for v in valores]
    return features, categorias, aceitam_vazio


def construir_modelo_entrada(pipeline: Pipeline) -> type[BaseModel]:
    """Gera a classe Pydantic de um cliente, a partir do artefato.

    `extra="forbid"` não é preferência: com o default (`ignore`), um campo
    desconhecido é **descartado em silêncio** e volta inteiro no corpo do 422
    quando falta outro campo. No Telco, o campo que um integrador naturalmente
    manda a mais é `CustomerID` — identificador e dado pessoal (LGPD), que o
    projeto descartou na Etapa 1 e que chegaria ao log de acesso sem ninguém
    ter escrito a linha que o coloca lá.

    ⚠️ `forbid` sozinho **não fecha** o vazamento — ele o concentra: o erro
    `extra_forbidden` devolve `loc` E `input` (nome e valor do campo proibido).
    Quem fecha é o handler de `RequestValidationError` em `app.py`, que remove
    `input` e `ctx`. As duas peças, não uma.
    """
    features, categorias, aceitam_vazio = contrato_do_pipeline(pipeline)

    campos: dict[str, Any] = {}
    for coluna in features:
        campo = identificador(coluna)
        if coluna in categorias:
            valores = categorias[coluna]
            tipo = Annotated[
                Literal[tuple(valores)],  # type: ignore[valid-type]
                Field(alias=coluna, description=f"um de: {', '.join(valores)}"),
            ]
        else:
            lo, hi = FAIXAS[coluna]
            numero = Annotated[float, Field(ge=lo, le=hi)]
            if coluna in aceitam_vazio:
                # `null` permitido, e a faixa continua valendo para o ramo float:
                # `Total Charges: null` passa, `Total Charges: -999` não. O `ge`/`le`
                # está no tipo interno de propósito — no externo ele rejeitaria o
                # próprio `None` que a coluna existe para aceitar.
                tipo = Annotated[
                    numero | None,
                    Field(alias=coluna,
                          description=f"faixa de negócio aceita: {lo:g} a {hi:g}; "
                                      f"`null` = sem ciclo de faturamento"),
                ]
            else:
                tipo = Annotated[
                    numero,
                    Field(alias=coluna,
                          description=f"faixa de negócio aceita: {lo:g} a {hi:g}"),
                ]
        campos[campo] = (tipo, ...)

    return create_model(
        "Cliente",
        __config__=ConfigDict(
            extra="forbid",
            # Só o nome REAL da coluna é aceito na entrada. Aceitar também o
            # identificador Python daria duas formas de escrever a mesma
            # requisição, e é a segunda que ninguém testa.
            populate_by_name=False,
        ),
        **campos,
    )


def construir_modelo_lote(modelo_cliente: type[BaseModel]) -> type[BaseModel]:
    """O lote — que é o endpoint principal, não o extra.

    Medido: as 1.409 linhas da validação custam **2,853 ms** numa chamada e
    **2.353 ms** em 1.409 chamadas unitárias — **825×**. O custo do sklearn é
    quase todo fixo por chamada (~1,67 ms) e a linha marginal custa 2,0 µs. A
    API de churn é intrinsecamente batch: o consumidor é a campanha de retenção
    rodando sobre a carteira, não um cliente por vez.
    """
    return create_model(
        "LoteRequest",
        __config__=ConfigDict(extra="forbid"),
        clientes=(
            Annotated[
                list[modelo_cliente],  # type: ignore[valid-type]
                Field(min_length=1, max_length=MAX_LOTE),
            ],
            ...,
        ),
    )


# --- Saída -----------------------------------------------------------------
#
# `response_model` em TODAS as rotas, e os modelos abaixo são o motivo: o campo
# que não está declarado NÃO SAI, mesmo que o `return` o inclua. Medido: um
# handler que devolve `CustomerID` + eco da entrada responde 36 bytes com
# `response_model` e 374 bytes sem. É a única defesa contra overexposure que não
# depende de alguém lembrar de não escrever o campo.
#
# ⚠️ Nenhum destes modelos ecoa a entrada. A correlação entre requisição e
# resposta é feita pelo `request_id` gerado NO SERVIDOR — devolver o payload
# significa devolver dado pessoal que o cliente já tem.


class Predicao(BaseModel):
    """O resultado para um cliente.

    Devolve **probabilidade**, não classe. `.predict()` aplicaria 0,5 implícito
    e apagaria, numa linha, a etapa que derivou o limiar pela economia do erro:
    medido no campeão, 0,50 custa R$ 39.296 por ciclo contra R$ 31.750 em 0,29 —
    **R$ 7.546 e 83 churners** de diferença.

    `decisao` vem junto porque é o que a campanha consome, e `limiar` vem junto
    porque sem ele a decisão não é auditável: quem lê a resposta seis meses
    depois precisa saber com que corte ela foi tomada.
    """

    probabilidade: float = Field(ge=0.0, le=1.0, examples=[0.6601])
    decisao: bool = Field(description="probabilidade >= limiar de operação")
    limiar: float = Field(ge=0.0, le=1.0, examples=[0.29])


class PredicaoResponse(BaseModel):
    request_id: str = Field(description="gerado no servidor; correlaciona com o log")
    versao_modelo: str
    predicoes: list[Predicao]


class Saude(BaseModel):
    """A resposta do `/health` — e ela precisa AFIRMAR algo que não se sabia.

    `{"status": "ok"}` responde 200 exatamente no cenário de falha que importa
    (modelo não carregado). Pior: quando o estado é lido de uma variável global
    paralela, o endpoint responde `healthy` com o artefato **apagado do disco** —
    medido, com `/predict` levantando `FileNotFoundError` na requisição seguinte.

    Por isso todo campo abaixo é lido do objeto que serve, e a identidade do
    artefato (`sha256`) está aqui: um health check que responde 200 sem dizer o
    que está carregado é o oitavo servidor da Knight Capital.
    """

    status: Literal["pronto", "degradado"]
    versao_modelo: str
    artefato_sha256: str
    n_features: int
    limiar_operacao: float
    # 🚨 DOIS campos, não um. O antigo `versoes` vinha dos metadados do artefato
    # (o ambiente que TREINOU) num endpoint que responde "o que este serviço tem".
    # Medido no container: dizia Python 3.12.5, a imagem rodava 3.12.14. Ninguém
    # mentiu — o rótulo respondia a outra pergunta, que é o modo de falha mais
    # comum deste repositório e o mais caro justo aqui, onde o endpoint existe
    # para declarar identidade.
    versoes_treino: dict[str, str] = Field(
        description="ambiente que treinou o modelo (carimbo gravado na promoção)")
    versoes_runtime: dict[str, str] = Field(
        description="ambiente que está servindo agora (medido no processo)")


class ErroResponse(BaseModel):
    """Erro sem `input` e sem `ctx` — ver o handler em `app.py`."""

    request_id: str
    erro: str
    detalhes: list[dict[str, Any]] = []

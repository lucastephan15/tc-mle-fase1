"""
As rotas — Etapa 9d.

⚠️ **Este módulo NÃO tem `from __future__ import annotations`, de propósito.** As
classes de entrada são geradas a partir do artefato (`schema.construir_modelo_entrada`)
e existem como variáveis locais da factory; com anotações adiadas elas virariam
strings que o FastAPI tentaria resolver no escopo do módulo, onde não existem.
Acrescentar o import "por padronização" quebra as duas rotas de predição.

🔑 **A carga do artefato acontece na CONSTRUÇÃO do app, e isso é uma decisão que
o mecanismo do `lifespan` não conseguiria cumprir sozinho.** O contrato de
entrada é derivado do artefato (13 nomes, 25 valores categóricos, 3 faixas), e um
`lifespan` roda *depois* de as rotas existirem — quando o schema já teria de estar
definido. Então quem cede é o mecanismo, não a fonte de verdade do contrato: a
factory carrega antes de qualquer rota ser declarada, o que é ainda mais cedo que
o `lifespan` e serve ao mesmo propósito medido (o **deploy** paga os ~715 ms, não
o primeiro cliente — 383× na primeira requisição contra carga preguiçosa com
`@lru_cache`, que não é carregamento antecipado, é carregamento preguiçoso com
memória).

🚨 **Todas as rotas são `def`, nunca `async def`.** Medido com 8 requisições
concorrentes, trocando só a palavra-chave: a vazão é idêntica (203,1 × 206,4 ms —
o GIL serializa de qualquer jeito), mas o atraso máximo do event loop vai de
**32,58 ms** (`def`, threadpool do Starlette) para **206,28 ms** (`async def`, o
lote inteiro). O que fica preso na fila junto é o `GET /health`, que é a probe do
orquestrador ⇒ healthcheck expira ⇒ **container reiniciado no meio da campanha**.
A pergunta que decide não é *async ajuda?*, é *async prejudica?*.
"""

import uuid
from time import perf_counter
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src import artefato as art_mod
from src.api import schema, servico

TITULO = "API de churn — Tech Challenge Fase 1"


def obter_pontuador() -> servico.Pontuador:
    """Marcador de dependência. A implementação é injetada em `criar_app`.

    É o "D" de SOLID: o handler depende de uma abstração ("algo que sabe
    pontuar"), e quem escolhe a implementação é quem monta o app. No teste,
    `app.dependency_overrides[obter_pontuador] = lambda: dublê` troca a
    implementação em uma linha — e o teste passa a exercitar **o contrato**
    (campo faltando, categoria inédita, limiar, formato da resposta) em vez do
    disco.
    """
    raise NotImplementedError("dependência não injetada — use criar_app()")


def criar_app(artefato: art_mod.Artefato | None = None) -> FastAPI:
    """Monta a aplicação em torno de UM artefato.

    Recebe o artefato em vez de ir buscá-lo: é o que permite ao teste montar o
    app sobre um artefato de fixture, e é o que torna explícito que a API serve
    um objeto específico — não "o modelo", genericamente.
    """
    artefato = artefato or art_mod.carregar()
    pontuador = servico.PontuadorArtefato(artefato)

    Cliente = schema.construir_modelo_entrada(artefato.pipeline)
    Lote = schema.construir_modelo_lote(Cliente)

    app = FastAPI(
        title=TITULO,
        version=artefato.versao,
        description=(
            "Ranqueador de clientes por risco de churn. Devolve **probabilidade**, "
            "não classe: o limiar de operação é parâmetro de negócio e acompanha a "
            "resposta.\n\n"
            "⚠️ Limitação declarada: **não há autenticação**. É uma API interna, e "
            "isto está registrado como limitação da entrega, não esquecido."
        ),
    )
    app.dependency_overrides[obter_pontuador] = lambda: pontuador

    # `Annotated[...]` em vez de `= Depends(...)` no default do argumento: as
    # duas formas são equivalentes para o FastAPI, e só esta não é uma chamada
    # de função avaliada na definição (B008 do bugbear, que o repo tem ligado).
    # Suprimir a regra com uma diretiva de exceção seria trocar uma correção de
    # duas palavras por uma afirmação sobre o linter que ninguém verifica.
    # (E escrever a própria diretiva dentro deste comentário fazia o ruff
    #  reclamar de diretiva malformada — a regra lê o comentário, o que é
    #  exatamente o mecanismo que o RUF100 existe para explorar.)
    Servico = Annotated[servico.Pontuador, Depends(obter_pontuador)]

    # --- Middleware: request_id + cronometragem ----------------------------
    #
    # O `request_id` é gerado NO SERVIDOR e é o que substitui o eco do payload:
    # a resposta correlaciona com o log sem devolver dado pessoal que o cliente
    # já tem. `try/finally` porque instrumentação que só mede o caminho feliz
    # mede o que não precisa de medição — o 422 e o 500 também são cronometrados,
    # e sem isso o P95 fica cego justamente para o que falha rápido.
    @app.middleware("http")
    async def identificar_e_cronometrar(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        inicio = perf_counter()
        try:
            resposta = await call_next(request)
        finally:
            ms = (perf_counter() - inicio) * 1000
        resposta.headers["X-Request-ID"] = request.state.request_id
        resposta.headers["X-Response-Time-ms"] = f"{ms:.3f}"
        return resposta

    # --- Handlers de erro: o que NÃO sai na resposta -----------------------

    @app.exception_handler(RequestValidationError)
    async def erro_de_validacao(request: Request, exc: RequestValidationError):
        """422 sem `input` e sem `ctx`.

        O default do FastAPI devolve o valor rejeitado — e no caso de campo
        faltando devolve o **objeto inteiro**. Com `extra="forbid"`, o erro
        `extra_forbidden` traz `loc: ["CustomerID"]` e `input: "3668-QPYBK"`:
        nome e valor do dado pessoal, sozinhos e no topo da lista. `loc` pode
        ficar (nome de campo não é dado pessoal); `input` e `ctx`, não.
        """
        detalhes = [
            {"tipo": e.get("type", ""), "campo": ".".join(str(p) for p in e.get("loc", []))}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=schema.ErroResponse(
                request_id=getattr(request.state, "request_id", "-"),
                erro="payload inválido",
                detalhes=detalhes,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def erro_interno(request: Request, exc: Exception):
        """500 que diz **que** falhou, nunca **por quê**.

        `str(e)` do sklearn cita o dado do cliente — medido: `could not convert
        string to float: 'setenta reais'`. O motivo é para o log (Etapa 10),
        correlacionado pelo `request_id`; a resposta leva só o identificador.
        """
        return JSONResponse(
            status_code=500,
            content=schema.ErroResponse(
                request_id=getattr(request.state, "request_id", "-"),
                erro="erro interno ao pontuar",
            ).model_dump(),
        )

    # --- Rotas -------------------------------------------------------------

    @app.get("/health", response_model=schema.Saude, tags=["operação"])
    def health(p: Servico):
        """Prontidão, não vitalidade: **posso receber tráfego?**

        Declara o que está carregado — versão, sha256 do artefato, nº de features
        e limiar — porque um 200 que não identifica o modelo é o oitavo servidor
        da Knight Capital: o que está servindo pode não ser o que foi validado, e
        nada no sistema perguntou.
        """
        return schema.Saude(
            status="pronto" if p.pronto() else "degradado",
            versao_modelo=p.versao,
            artefato_sha256=p.sha256,
            n_features=len(p.features),
            limiar_operacao=p.limiar,
            versoes_treino=p.versoes_treino,
            versoes_runtime=p.versoes_runtime,
        )

    @app.post("/v1/predict", response_model=schema.PredicaoResponse, tags=["inferência"])
    def predict(request: Request, cliente: Cliente, p: Servico):
        """Um cliente — o caso particular de lote 1.

        Existe por conveniência do integrador, não por ser o caso principal:
        1.409 chamadas unitárias custam 825× o mesmo trabalho em lote.
        """
        return _responder(request, p, [cliente])

    @app.post("/v1/predict-batch", response_model=schema.PredicaoResponse,
              tags=["inferência"])
    def predict_batch(request: Request, lote: Lote, p: Servico):
        """O endpoint principal: a campanha de retenção sobre a carteira."""
        return _responder(request, p, lote.clientes)

    return app


def _responder(request: Request, p: servico.Pontuador, clientes: list):
    """Monta a resposta. `by_alias=True` devolve os NOMES REAIS das colunas.

    É o ponto exato onde o contrato HTTP vira contrato de coluna: o DataFrame é
    construído com `Tenure Months`, não com `tenure_months`, e o
    `ColumnTransformer` seleciona por nome. Errar aqui não daria 422 — daria 500
    no pré-processamento, ou pior, uma coluna faltando tratada como ausente.
    """
    linhas = [c.model_dump(by_alias=True) for c in clientes]
    scores = p.pontuar(linhas)
    limiar = p.limiar
    return schema.PredicaoResponse(
        request_id=getattr(request.state, "request_id", "-"),
        versao_modelo=p.versao,
        predicoes=[
            schema.Predicao(probabilidade=s, decisao=s >= limiar, limiar=limiar)
            for s in scores
        ],
    )


# 🚨 NÃO existe um `app = criar_app()` aqui, e a ausência é a correção de um erro
# real — cometido, empurrado e pego pelo CI em 17/08/2026.
#
# Com o objeto de módulo, a carga do artefato virava efeito colateral do
# **import**: qualquer `import src.api.app` passava a exigir `models/campeao.joblib`
# no disco. `make ci` passava na máquina de quem escreveu (o artefato está lá) e a
# suíte inteira falhava **na coleta** no runner limpo, onde `models/` está vazio por
# decisão — 81 testes derrubados por um efeito colateral de import.
#
# É o acoplamento do 9d-quater na forma mais literal (lógica que só existe se o
# recurso externo existir), e o único lugar onde ele aparece é a máquina limpa.
#
# O uvicorn recebe a factory e a chama ele mesmo, o que preserva a propriedade que
# importava: a carga acontece **antes de a primeira conexão ser aceita**, e o
# processo MORRE na inicialização se o artefato não bater — em vez de subir
# degradado e responder 200 com a probabilidade de outro modelo.
#
#     uvicorn src.api.app:criar_app --factory --host 0.0.0.0 --port 8000

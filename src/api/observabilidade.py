"""Log estruturado de inferência — Etapa 10a.

Uma linha **JSON por requisição**, em stdout. É o insumo de tudo que a Etapa 10
faz depois: latência em percentis, taxa de erro, prediction drift e — quando
explicitamente ligado — data drift.

🚨 **A decisão que este módulo materializa é a máscara LGPD, e ela foi tomada
ANTES da primeira linha ser emitida.** O motivo é assimetria: aumentar o que se
loga é uma linha de configuração; *desfazer* o que já foi emitido não é
operação nenhuma. Log emitido não volta.

**Por que a decisão é mais dura aqui do que no `.gitignore` e no
`.dockerignore`** — as três camadas do mesmo problema, e só as duas primeiras
têm solução por exclusão:

    repositório  ->  `.gitignore`      (exclui `logs/*.jsonl`, motivo LGPD escrito)
    imagem       ->  `.dockerignore`   (allowlist, 9f)
    stdout       ->  🚨 NADA

Num PaaS, stdout é coletado **pela plataforma** — um terceiro, cuja retenção não
é nossa e que não assinou nada. Não há regra de exclusão possível, porque logar
é a finalidade do arquivo. Onde não se pode excluir, decide-se **o que se
escreve**.

🔑 **A saída não é "tudo ou nada", porque as duas famílias de drift têm custos de
privacidade OPOSTOS:**

| | precisa de | é dado pessoal? | quando |
|---|---|---|---|
| **prediction drift** (`P(ŷ)`) | as probabilidades de saída | **não** — um número sem atributo ao lado não identifica ninguém | **sempre** |
| **data drift** (`P(X)`) | as 13 features de entrada | **sim** — inclui `Gender`, `Senior Citizen`, `Partner`, `Dependents` | só com `TC_LOG_FEATURES=1` |

Ou seja: a vigilância que roda **de graça e sem exposição** fica ligada em
produção; a que custa privacidade é ligada **onde e quando** alguém decide —
na prática, localmente, para o `simulate_drift.py`. Prediction drift é sintoma e
não causa, mas é sintoma de graça.

⚠️ **As 4 demográficas estão nas features DE PROPÓSITO** (`config.CAT_DISPONIVEIS`):
foram mantidas para que a auditoria de fairness da Etapa 10.5 possa **medir** o
viés em vez de ficar cega para ele. A consequência é que este log é mais sensível
do que seria se elas tivessem sido descartadas — a decisão certa numa etapa cobra
o preço em outra, e o lugar de dizer isso é aqui.
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime

# --- A flag, e por que ESTE `getenv` com default é legítimo -----------------
#
# O item 100 do revisita proíbe `os.getenv("X", "um-default")` para segredo,
# porque lá o default transforma "esqueci de configurar" em "configurei com o
# valor público": a ausência da variável leva ao estado INSEGURO, e o código
# funciona, que é por que ninguém descobre.
#
# Aqui a assimetria é a oposta: a ausência da variável leva ao estado
# CONSERVADOR (não logar dado pessoal). Esquecer de configurar não expõe nada —
# no máximo deixa de coletar. A regra não é "nunca use default"; é **o default
# tem de ser a direção segura**.
LOGAR_FEATURES = os.getenv("TC_LOG_FEATURES", "0") == "1"

# `arredondamento` das probabilidades: 6 casas é MUITO mais resolução do que o
# PSI precisa (ele bina em ~10 faixas) e descarta de propósito o ruído de última
# casa que já sabemos existir entre plataformas — 40% das linhas diferem em até
# 4 ulps entre macOS e Linux, com zero decisões trocadas (9f-ter). Guardar essas
# casas convidaria alguém a comparar predições por igualdade exata, que é
# exatamente o que aquela medição proibiu.
CASAS = 6

NOME = "tc.inferencia"


class _HandlerStdout(logging.StreamHandler):
    """`StreamHandler` que resolve `sys.stdout` **na hora de emitir**.

    🚨 O `StreamHandler` da biblioteca padrão guarda o objeto de stream no
    momento da CONSTRUÇÃO. Como `configurar()` roda dentro de `criar_app()`, o
    handler ficaria preso ao `sys.stdout` que existia quando a aplicação foi
    montada — e passaria a escrever nele para sempre, mesmo depois de alguém
    trocar o `sys.stdout` do processo.

    Isso foi descoberto **porque o teste falhou** (18/08/2026), e o modo de falha
    é o mais caro que existe: sob o pytest, a linha saía num descritor que nem
    `capsys` nem `capfd` liam. Um teste escrito de forma um pouco mais frouxa —
    "não explodiu, logo logou" — teria ficado **verde sem nunca ter visto uma
    linha**, e a Etapa 10 inteira seria construída sobre um log que ninguém
    verificou.

    🔑 A lição é a mesma do `.gitignore` que o Docker não lê: *alguém guardou uma
    referência a um recurso e o recurso mudou por baixo*. A correção é não
    guardar — perguntar toda vez.
    """

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _valor):
        # O `__init__` do StreamHandler atribui `self.stream`. Aceitar e ignorar
        # é o que mantém a classe utilizável sem reescrever o construtor.
        pass


def configurar(nivel: str = "INFO") -> logging.Logger:
    """Logger que emite a linha **crua**, sem prefixo nenhum.

    `Formatter("%(message)s")` não é preferência de estilo: com qualquer prefixo
    (`INFO:tc.inferencia:...`) a linha deixa de ser JSON válido e o parser da
    Etapa 10 quebra na primeira. Quem consome este arquivo é código.

    Pelo mesmo motivo o `json.dumps` fica explícito no chamador em vez de vir de
    uma biblioteca de formatter: assim **as chaves da linha são decididas aqui**,
    e não pelos defaults de um pacote que muda no próximo `pip-compile`. Schema
    herdado de default de biblioteca é convenção; schema explícito é contrato —
    é o drill do `ColumnTransformer` (nome × posição) aplicado ao log.
    """
    logger = logging.getLogger(NOME)
    logger.setLevel(nivel)

    # stdout, nunca arquivo: o filesystem do container é efêmero, e num PaaS o
    # que a plataforma coleta é o stream padrão. `_HandlerStdout` em vez de
    # `StreamHandler(sys.stdout)` porque o segundo congela o stream na
    # construção — ver a docstring da classe.
    handler = _HandlerStdout()
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Substituir em vez de acrescentar: `configurar()` é chamada uma vez por
    # `criar_app()`, e a suíte monta vários apps no mesmo processo. Sem isto,
    # cada app novo somaria um handler e a linha sairia duplicada N vezes — um
    # defeito que só aparece com mais de um app vivo, ou seja, só nos testes.
    logger.handlers = [handler]

    # O uvicorn e o root logger não repassam esta linha adiante. O access log do
    # servidor é desligado por outro caminho (`--no-access-log` no CMD), porque
    # ele é outro emissor: `propagate=False` resolve duplicação DENTRO da
    # hierarquia do `logging`, e o access log não está nela.
    logger.propagate = False
    return logger


def linha(
    *,
    request_id: str,
    metodo: str,
    rota: str,
    status_code: int,
    latency_ms: float,
    artefato_sha256: str,
    n_linhas: int | None = None,
    scores: list[float] | None = None,
    features: list[dict] | None = None,
) -> dict:
    """Monta o registro. **7 campos canônicos**, sempre presentes.

    O 7º é o `artefato_sha256`, e ele não é redundante com o que a resposta já
    devolve: a resposta é efêmera e o log é o rastro. Quando o PSI cruzar o
    limiar daqui a três semanas, a primeira hipótese a descartar é "trocaram o
    modelo no meio da janela" — e sem o hash na linha não há como descartá-la.

    `features` só entra se o chamador o passar, e o chamador só o passa com
    `LOGAR_FEATURES`. A decisão fica em UM lugar.
    """
    registro = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "metodo": metodo,
        "rota": rota,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
        "artefato_sha256": artefato_sha256,
    }
    # Ausente é diferente de zero: um 422 não pontuou nenhuma linha, e escrever
    # `n_linhas: 0` faria a média de tamanho de lote incluir requisições que
    # nunca chegaram ao modelo.
    if n_linhas is not None:
        registro["n_linhas"] = n_linhas
    if scores is not None:
        registro["scores"] = [round(s, CASAS) for s in scores]
    if features is not None:
        registro["features"] = features
    return registro


def registrar(logger: logging.Logger, **campos) -> None:
    """Emite uma linha. `ensure_ascii=False` porque os nomes de coluna e os
    valores categóricos do Telco são ASCII, mas a mensagem de contexto pode não
    ser — e `\\uXXXX` no meio de um `.jsonl` é ilegível sem ganho nenhum."""
    logger.info(json.dumps(linha(**campos), ensure_ascii=False))

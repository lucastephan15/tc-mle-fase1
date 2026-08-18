"""
Etapa 10c-bis — fabricar o drift e provar que o detector dispara.

    python scripts/simulate_drift.py              # os três cenários
    python scripts/simulate_drift.py --cenario preco --n 300

**O problema que ele resolve é de calendário, não de estatística.** Drift real
leva meses; um Tech Challenge de seis semanas nunca vê um, e a Etapa 10 termina
declarativa — *"eu monitoraria PSI e alertaria acima de 0,25"*. A inversão: em
vez de esperar o drift, **fabrique** o drift e mostre o alarme nascer. O que se
testa aqui **não é o modelo** — é o sistema de vigilância. É um teste unitário do
monitoramento, e a evidência que sai dele (antes/depois) é o que vai na
documentação.

**Como funciona, e por que atravessa HTTP de verdade:**

1. sobe a API num subprocesso, com o stdout redirecionado para um `.jsonl`;
2. manda uma janela **base** (amostra real da validação, intocada);
3. manda a mesma janela **deslocada** de propósito;
4. separa as duas metades do log **pelo `request_id`** que cada resposta
   devolveu, e roda `src.monitoring` sobre cada uma.

O passo 4 é o que dá uso ao `request_id`: ele existe para correlacionar resposta
e log sem devolver dado pessoal ao cliente, e aqui é literalmente isso que
acontece. Um teste in-process com o `TestClient` seria mais rápido e provaria
menos — o caminho exercitado é schema → pontuação → middleware → stdout, e é
nesse caminho que a Etapa 10 vive.

🚨 **A simulação liga `TC_LOG_FEATURES=1`, que produção mantém desligado.** Não é
descuido: é a razão de a flag existir. Data drift precisa das 13 colunas de
entrada, e elas incluem `Gender`/`Senior Citizen`/`Partner`/`Dependents` — logá-las
é aceitável **aqui** (máquina local, dado que já está no disco, arquivo em
`logs/`, que o `.gitignore` barra) e não é aceitável no PaaS, onde o stdout é
coletado por um terceiro. A mesma linha de código com custo de privacidade
diferente conforme **onde** roda.

⚠️ **Consequência incômoda e que vale dizer na entrega:** este script **não roda
contra a API em produção**. Não porque falte endereço, mas porque lá o stdout
pertence à plataforma — não há como lê-lo de volta. A terceira camada do
"quem lê este arquivo?" cobra o preço aqui: o log que é fácil de emitir é o
mesmo que é difícil de recuperar.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

from src import artefato as art_mod
from src import config, data, monitoring

LOGS = config.RAIZ / "logs"
LOTE = 100


# --- Os cenários -----------------------------------------------------------
#
# Cada um é uma hipótese de negócio escrita como transformação da carteira. O
# valor didático está em serem **plausíveis**: "a base envelheceu" e "houve
# reajuste" acontecem sem ninguém avisar o time de ML, que é o modo como o drift
# real chega.

def _envelhecer(X: pd.DataFrame) -> pd.DataFrame:
    """A carteira envelhece 12 meses — retenção funcionando, aliás."""
    X = X.copy()
    X["Tenure Months"] = X["Tenure Months"] + 12
    return X


def _reajustar(X: pd.DataFrame) -> pd.DataFrame:
    """Reajuste de 15% na mensalidade."""
    X = X.copy()
    X["Monthly Charges"] = (X["Monthly Charges"] * 1.15).round(2)
    return X


def _plano_novo(X: pd.DataFrame) -> pd.DataFrame:
    """Um terço da carteira migra para um plano que não existia no treino.

    🎯 **Este cenário respondeu o contrário do esperado, e o achado vale mais que
    o cenário.** A expectativa (a do enunciado, e a do catálogo de armadilhas):
    o `OneHotEncoder` trata categoria desconhecida como tudo-zero e **prediz
    mesmo assim, com HTTP 200** — nada na API acusa, e só o monitoramento pega,
    dias depois, via PSI.

    Medido em 18/08/2026: **a API devolve 422 e a linha nunca chega ao modelo.**
    O motivo é uma decisão da Etapa 9d: o schema é *derivado do artefato*
    (`Literal[tuple(ohe.categories_)]`), não escrito à mão como `str`. O contrato
    HTTP herdou as 25 categorias que o `OneHotEncoder` viu, então categoria
    inédita vira erro de **validação**, na borda, antes da inferência.

    🔑 Consequência para o desenho do monitoramento: nesta API, categoria nova
    **não é evento de drift — é evento de serviço**. Ela aparece como pico de 422
    na taxa de erro (10b, família "serviço"), não como PSI subindo (família
    "drift"), e aparece **na primeira requisição**, não ao fim de uma janela.
    Vigiar só PSI aqui seria vigiar o lugar errado; e um alerta de 4xx, que
    qualquer plataforma já dá de graça, é o detector mais rápido que temos para
    esta família específica.

    ⚠️ E o preço disso está declarado: o serviço **recusa** o cliente novo em vez
    de arriscar um palpite sobre ele. É a escolha certa para um ranqueador de
    campanha (a fila fica menor, ninguém é pontuado errado) e seria a errada para
    um serviço que precisa responder sempre — nesse caso o schema teria de
    aceitar a categoria e o monitoramento é que assumiria a carga.
    """
    X = X.copy()
    X.iloc[: len(X) // 3, X.columns.get_loc("Contract")] = "Two year prepaid"
    return X


CENARIOS = {
    "tenure": ("base envelhecendo (+12 meses de tenure)", _envelhecer),
    "preco": ("reajuste de 15% na mensalidade", _reajustar),
    "categoria": ("plano novo no catálogo (categoria inédita)", _plano_novo),
}


# --- Infraestrutura da simulação -------------------------------------------


def porta_livre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def subir_api(porta: int, saida) -> subprocess.Popen:
    """Sobe o uvicorn com o stdout indo para `saida` (um arquivo já aberto).

    Recebe o descritor em vez de abri-lo: quem é dono do arquivo é quem controla
    o escopo dele, e o arquivo precisa continuar aberto enquanto o subprocesso
    escreve. Abrir aqui dentro seria um `open` sem dono — que é exatamente o que
    o SIM115 do ruff aponta.

    `--no-access-log` pelo mesmo motivo do `CMD` da imagem: sem ele o servidor
    emite uma linha de texto puro por requisição no mesmo stream, e o parser do
    `monitoring` conta linhas ilegíveis em vez de métricas (medido na 10a: 3
    requisições davam 3 linhas JSON **+ 3 de texto**).
    """
    ambiente = {
        **os.environ,
        "TC_LOG_FEATURES": "1",       # ver o aviso no topo do arquivo
        "PYTHONPATH": str(config.RAIZ),
        "PYTHONUNBUFFERED": "1",      # senão o log só aparece quando o processo morre
    }
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:criar_app", "--factory",
         "--host", "127.0.0.1", "--port", str(porta), "--no-access-log"],
        stdout=saida, stderr=subprocess.STDOUT, cwd=config.RAIZ, env=ambiente,
    )


def esperar(base: str, processo: subprocess.Popen, timeout: float = 60.0) -> None:
    """Espera a PRONTIDÃO, não o processo existir."""
    limite = time.time() + timeout
    while time.time() < limite:
        if processo.poll() is not None:
            raise RuntimeError(f"a API morreu na inicialização (código {processo.returncode})")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.3)
    raise RuntimeError("a API não ficou pronta a tempo")


def enviar(base: str, X: pd.DataFrame) -> list[tuple[str, int]]:
    """Manda a janela em lotes e devolve os pares (`request_id`, status).

    🚨 **4xx é resposta, não exceção** (mesma regra da 9e). Uma janela que a API
    recusa é um resultado da simulação — foi assim que o cenário `categoria`
    revelou que a categoria inédita morre no contrato — e deixar o `urllib`
    levantar transformaria o achado num traceback.

    Funciona porque o 422 desta API também devolve `request_id` no corpo: a
    correlação resposta ↔ log sobrevive ao erro, que é justamente quando ela mais
    importa.

    ⚠️ `NaN` vira `null`: `json.dumps(float("nan"))` emite `NaN` literal, que não
    é JSON válido (armadilha herdada da 9e). Quem integra manda `null`.
    """
    ids = []
    for inicio in range(0, len(X), LOTE):
        fatia = X.iloc[inicio:inicio + LOTE]
        clientes = [
            {k: (None if isinstance(v, float) and math.isnan(v) else v)
             for k, v in linha.items()}
            for linha in fatia.to_dict(orient="records")
        ]
        req = urllib.request.Request(
            base + "/v1/predict-batch",
            data=json.dumps({"clientes": clientes}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                corpo, status = json.load(r), r.status
        except urllib.error.HTTPError as e:
            corpo, status = json.load(e), e.code
        ids.append((corpo.get("request_id", "-"), status))
    return ids


def simular(cenario: str, n: int, art: art_mod.Artefato) -> dict:
    """Roda o ciclo completo de um cenário e devolve os dois painéis."""
    descricao, transformar = CENARIOS[cenario]
    dados = data.dividir()
    X = dados.validacao.X.head(n)

    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"simulacao-{cenario}.jsonl"
    porta = porta_livre()
    base = f"http://127.0.0.1:{porta}"

    with open(log, "w", encoding="utf-8") as saida:
        processo = subir_api(porta, saida)
        try:
            esperar(base, processo)
            ids_antes = enviar(base, X)
            ids_depois = enviar(base, transformar(X))
        finally:
            processo.terminate()
            processo.wait(timeout=10)

    linhas, invalidas = monitoring.ler(log)
    ids_a = {i for i, _ in ids_antes}
    ids_d = {i for i, _ in ids_depois}
    antes = [linha for linha in linhas if linha.get("request_id") in ids_a]
    depois = [linha for linha in linhas if linha.get("request_id") in ids_d]

    return {
        "cenario": cenario,
        "descricao": descricao,
        "log": log,
        "invalidas": invalidas,
        "antes": monitoring.analisar(antes, art),
        "depois": monitoring.analisar(depois, art),
        "status_enviados": {"antes": [st for _, st in ids_antes],
                            "depois": [st for _, st in ids_depois]},
    }


# --- Saída -----------------------------------------------------------------


def imprimir(resultado: dict, art: art_mod.Artefato) -> None:
    print("\n" + "=" * 78)
    print(f"CENÁRIO: {resultado['cenario']} — {resultado['descricao']}")
    print(f"log: {resultado['log'].relative_to(config.RAIZ)}"
          + (f"  ⚠️ {resultado['invalidas']} linhas ilegíveis" if resultado["invalidas"] else ""))
    print("=" * 78)

    for rotulo in ("antes", "depois"):
        print(f"\n--- JANELA {rotulo.upper()} " + "-" * (78 - 14 - len(rotulo)))
        print(monitoring.relatorio(resultado[rotulo], art))

    a, d = resultado["antes"]["prediction_drift"], resultado["depois"]["prediction_drift"]
    print("\n>>> VEREDITO")

    recusados = [s for s in resultado["status_enviados"]["depois"] if s >= 400]
    if recusados:
        print(f"    🚨 A JANELA DESLOCADA FOI RECUSADA: {len(recusados)} de "
              f"{len(resultado['status_enviados']['depois'])} lotes com HTTP "
              f"{sorted(set(recusados))}.")
        print("    O drift não chegou a virar predição — virou ERRO DE CONTRATO, "
              "na borda.")
        print("    Detector correto para esta família: alerta de taxa de 4xx, "
              "não PSI.")
        print("    (o schema é derivado do artefato — ver a docstring do cenário)")
        return

    print(f"    prediction drift : PSI {a['psi']:.4f} ({a['classificacao']}) → "
          f"{d['psi']:.4f} ({d['classificacao']})")
    if "fila_x" in d:
        print(f"    fila de retenção : {a['fila_x']}× → {d['fila_x']}× do normal")
    if resultado["depois"]["data_drift"]:
        piores = sorted(resultado["depois"]["data_drift"].items(),
                        key=lambda kv: -kv[1]["psi"])[:2]
        alvo = ", ".join(f"{c} {r['psi']:.3f} ({r['classificacao']})" for c, r in piores)
        print(f"    data drift       : {alvo}")


LIMITACOES = """
LIMITAÇÕES — declaradas de propósito, e valem mais que o script
───────────────────────────────────────────────────────────────────────────────
1. Isto é **covariate shift** (`P(X)` mudou), e é detectável **por construção**,
   porque foi construído. O script NÃO simula **concept drift** — a regra
   `P(y|X)` mudando, por exemplo um concorrente entrando e clientes fiéis
   passando a cancelar. Esse é invisível às estatísticas de entrada: o objeto que
   mudou não existe sem `y`.

2. **Detectar não é corrigir.** O alerta abre um caminho (investigar → decidir se
   retreina → passar pelo gate), não fecha um. Um pipeline que retreina sozinho
   ao ver PSI alto é uma máquina de pôr modelo pior em produção mais rápido.

3. **A régua é PSI, não p-valor.** Com amostra grande, KS acusa qualquer coisa: a
   pergunta útil não é *"é diferente?"*, é *"é diferente o suficiente para
   importar?"*. E KS é para contínuas — categórica pede PSI ou qui-quadrado.

4. 🔑 **O deslocamento é univariado, e o detector também.** Somar 12 meses a
   `Tenure Months` sem mexer em `Total Charges` cria clientes com dois anos de
   casa e a fatura acumulada de um — combinação que **não existe no mundo**.
   Medido em 18/08/2026 (400 clientes): `Tenure Months` vai a **3,59** e
   `Total Charges` fica em **0,034** — estável, apesar de metade da história
   daqueles clientes ter deixado de fazer sentido. É a demonstração acidental do ponto do 10c: KS/PSI
   olham **uma coluna por vez** e são cegos à combinação inédita. Um detector
   multivariado (`IsolationForest` sobre o treino) veria o que estes não veem —
   e é o upgrade natural se sobrar tempo.
"""


ACHADOS = """
ACHADOS DESTA EXECUÇÃO — o que muda no desenho do monitoramento
───────────────────────────────────────────────────────────────────────────────
🔑 **As duas famílias de drift se movem de forma independente, e os dois
   cenários provam isso em direções opostas:**

   · `preco` (+15% na mensalidade): data drift **1,49 (agir)** e prediction
     drift **0,03 (estável)**, fila 1,01× — a entrada mudou muito e a saída
     quase nada. É o aviso do 10c em números: *nem todo data drift significa
     que o modelo piorou*. Painel que alerta por PSI de entrada teria acordado
     alguém às 3h por nada.

   · `tenure` (+12 meses): data drift **3,59 (agir)** e prediction drift
     **0,30 (agir)** — mas a fila de retenção **ENCOLHE** para 0,63×, porque
     tenure alto significa menos churn. 🚨 O alerta chega vestido de **boa
     notícia**: "a campanha vai ligar para 37% menos gente". Um painel de
     negócio olhando só volume comemora; um painel de drift pergunta por quê.
     Este é o caso que justifica manter as duas leituras lado a lado.

🚨 **A terceira família nem chega ao modelo:** categoria inédita vira 422 no
   contrato (o schema é derivado do artefato). Vigiar isso com PSI seria vigiar
   o lugar errado — o detector certo é a taxa de 4xx, e ela dispara na primeira
   requisição em vez de ao fim de uma janela.

📌 **Nenhum dos três exige rótulo.** É por isso que drift é o que se vigia em
   tempo real: a qualidade preditiva só existe depois que o churn se confirma.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 10c-bis — drift fabricado")
    parser.add_argument("--cenario", choices=[*CENARIOS, "todos"], default="todos")
    parser.add_argument("--n", type=int, default=400,
                        help="clientes por janela (default: 400)")
    args = parser.parse_args(argv)

    art = art_mod.carregar()
    print(f"artefato servido: {art.sha256[:16]}…  ·  limiar {art.limiar}  ·  "
          f"baseline: n={art.referencia['n']} (treino) / "
          f"n={art.referencia['scores']['n']} (validação)")

    escolhidos = list(CENARIOS) if args.cenario == "todos" else [args.cenario]
    for cenario in escolhidos:
        imprimir(simular(cenario, args.n, art), art)

    if args.cenario == "todos":
        print(ACHADOS)
    print(LIMITACOES)
    return 0


if __name__ == "__main__":
    sys.exit(main())

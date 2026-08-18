# Imagem de SERVIÇO da API de churn — Etapa 9f.
#
# 🔑 A distinção que organiza este arquivo: **o Dockerfile é a receita; a imagem é
# o artefato.** Só a imagem é reprodutível por construção (imutável, endereçada por
# digest). A receita só é reprodutível se fixar tudo o que ela resolve em tempo de
# build — base, pacotes e plataforma. Um Dockerfile com tag flutuante e `pip install`
# sem pins produz imagens diferentes em meses diferentes, e é exatamente o que um
# material sobre reprodutibilidade costuma ensinar.
#
# Build (a plataforma é explícita de propósito — ver §PLATAFORMA no fim):
#     docker buildx build --platform linux/amd64 -t tc-churn:1.0.0 .
#
# Executar:
#     docker run --rm -p 8000:8000 tc-churn:1.0.0

# --- BASE ------------------------------------------------------------------
#
# 🚨 Fixada por DIGEST, não por tag. `python:3.12-slim` é uma tag mutável: aponta
# para uma imagem hoje e para outra no mês que vem, silenciosamente. O digest é o
# que torna o build repetível.
#
# 🚨 E é 3.12, não 3.9. O `python:3.9-slim` de tutorial **não constrói este
# projeto**: `scikit-learn 1.9.0` e `numpy 2.4.6` exigem `>=3.11`. Essa falha alta e
# imediata só existe porque o lock tem pins — sem eles, o pip resolveria versões
# antigas compatíveis com 3.9, a imagem construiria normalmente e o artefato
# quebraria na carga dentro do container. A versão do Python é parte do contrato do
# artefato serializado, não escolha de conveniência.
#
# Digest do índice multi-plataforma (18/08/2026) = Python 3.12.14-slim-trixie.
# ⚠️ Divergência conhecida e aceita: o venv local roda 3.12.5. Mesma série 3.12
# (ABI e bytecode compatíveis) e o lock trava as bibliotecas, que é onde a
# serialização mora — `src/artefato.py` verifica a versão do scikit-learn na carga
# e mata o processo se divergir.
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

# PYTHONDONTWRITEBYTECODE: bytecode gerado em runtime só engorda a camada de
# escrita do container. PYTHONUNBUFFERED: sem ele o stdout fica em buffer e o log
# da Etapa 10 chega picado (ou não chega, se o processo morrer) — em container se
# loga em stdout, porque o filesystem é efêmero.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- DEPENDÊNCIAS ----------------------------------------------------------
#
# 🔑 Esta camada vem ANTES do código, e a ordem é a decisão. O Docker reaproveita
# uma camada enquanto os arquivos que ela copia não mudam: com o lock copiado
# primeiro, alterar uma linha de `src/api/app.py` reaproveita a instalação inteira
# e o build leva segundos. Na ordem invertida (`COPY . .` antes do pip), cada
# vírgula de código reinstala scipy, numpy, pandas e scikit-learn.
#
# ⚠️ É o `requirements-serve.txt` (30 dists, 273,9 MB), não o `requirements.txt`
# (118 dists, 1.498,8 MB). Ver o cabeçalho do requirements-serve.in: 82% do
# ambiente é treino, e nada disso é importado por `src/api/`.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# --- APLICAÇÃO -------------------------------------------------------------
#
# Só o pacote e o artefato — o `.dockerignore` é uma allowlist e já garante que
# `data/`, `mlruns/`, `logs/`, `.venv` e `.git` não têm como entrar. Estes dois
# COPY são explícitos mesmo assim: `COPY . .` dependeria do .dockerignore estar
# certo para não vazar dado pessoal, e uma cópia explícita não depende de nada.
COPY src/ ./src/
COPY models/campeao.joblib ./models/campeao.joblib

# O caminho do artefato é configuração, não constante: `src/config.py` já o lê de
# `TC_ARTEFATO`. Declarar aqui documenta o layout da imagem e mantém a porta
# aberta para o dia em que o artefato vier de um volume ou de um registry.
ENV TC_ARTEFATO=/app/models/campeao.joblib \
    PYTHONPATH=/app

# --- USUÁRIO ---------------------------------------------------------------
#
# Um processo que só lê um .joblib e responde JSON não precisa ser root. Se a
# aplicação for comprometida, a diferença entre root e `appuser` é a diferença
# entre escrever em qualquer lugar da imagem e não escrever em lugar nenhum.
# `--no-create-home` porque o serviço não usa `$HOME`.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# --- HEALTHCHECK -----------------------------------------------------------
#
# 🎯 Checa PRONTIDÃO, não vitalidade. Um healthcheck que só confirma que a porta
# abriu responde "saudável" exatamente no cenário de falha que importa. Este exige
# `status == "pronto"`, que a API só declara com o artefato carregado e verificado.
#
# ⚠️ Sem `curl` na imagem slim (e instalá-lo seria acrescentar superfície para uma
# requisição HTTP que o Python já sabe fazer). A porta vem do ambiente porque um
# PaaS pode injetá-la — healthcheck fixado em 8000 falharia em silêncio ali.
#
# 🚨 O `/health` fica FORA de qualquer credencial e de qualquer limitador de taxa.
# Uma dependência de autenticação no app inteiro faria esta probe receber 401, e o
# orquestrador reiniciaria o container em laço — o serviço morto por estar protegido.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "\
import os, sys, json, urllib.request; \
porta = os.environ.get('PORT', '8000'); \
r = urllib.request.urlopen(f'http://127.0.0.1:{porta}/health', timeout=4); \
sys.exit(0 if json.load(r).get('status') == 'pronto' else 1)"

# --- COMANDO ---------------------------------------------------------------
#
# 🚨 `criar_app --factory`, nunca `app`. O módulo não expõe um objeto de app: se
# expusesse, carregar o artefato viraria efeito colateral do `import` — erro real
# cometido em 17/08/2026, que derrubou 81 testes na coleta do runner limpo e passava
# na máquina de quem escreveu. O uvicorn chama a factory ele mesmo, o que preserva a
# propriedade que importava (o artefato carrega **antes** de a primeira conexão ser
# aceita) e mantém a outra (o processo MORRE no boot se o artefato não bater, em vez
# de subir degradado e responder 200 com a probabilidade de outro modelo).
#
# 🚨 Nunca `--reload`: é recarregador de código, não modo verboso.
#
# 🚨 UM worker. O número de workers sai da RAM, não da vazão — medido: 193,6 MB de
# RSS por worker, dos quais 93% é import de biblioteca científica. Quatro workers
# = 774 MB, acima do teto de 512 MB de um plano gratuito típico (OOM kill). E a
# vazão não os pede: 1,70 ms por predição unitária e 2,853 ms para as 1.409 linhas
# da validação num único `/v1/predict-batch`. Onde houver RAM medida e sobrando, o
# ajuste é `--workers N` aqui, com o número saindo da conta de memória.
#
# `${PORT:-8000}` porque um PaaS reivindica esse parâmetro; `exec` para o uvicorn
# virar PID 1 e receber o SIGTERM do `docker stop` em vez de o shell o engolir.
CMD ["sh", "-c", "exec uvicorn src.api.app:criar_app --factory --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

# --- PLATAFORMA ------------------------------------------------------------
#
# 🚨 Um contêiner Linux no macOS roda dentro de uma VM Linux **arm64**, então
# `docker build` no Apple Silicon produz uma imagem arm64 por padrão. O runner do
# GitHub Actions e a maioria das clouds são `x86_64` ⇒ `exec format error`, e o
# erro aparece **no deploy**, não no build local que passou. Daí o
# `--platform linux/amd64` na linha de build documentada no topo — declarado aqui
# porque não há como expressá-lo dentro do próprio Dockerfile.

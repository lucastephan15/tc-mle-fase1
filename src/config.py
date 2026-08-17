"""
Configuração central — caminhos, seed e o contrato de colunas.

Existe para que a lista de colunas descartadas na Etapa 1 seja CÓDIGO, e não um
comentário no decision log. Toda decisão da §1 do log tem uma linha aqui.
"""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "Telco_customer_churn.xlsx"
PROCESSED = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"

# --- Artefato promovido (Etapa 9c) -----------------------------------------

# UM arquivo, com pipeline e metadados dentro. Dois arquivos ("modelo.joblib" +
# "metadados.json") sem verificação de que combinam produzem rótulo errado com
# HTTP 200 — o modo de falha mais caro de uma API de ML, porque não falha.
#
# O caminho é configurável por variável de ambiente porque no container o
# artefato pode vir de um volume ou de outra camada — e porque é isso que permite
# testar, num processo limpo, que importar a API **não** exige o arquivo.
ARTEFATO = Path(os.getenv("TC_ARTEFATO", str(MODELS / "campeao.joblib")))

# Versão DO MODELO servido, não do código. Sobe quando o artefato promovido
# passa a ser outro objeto (novo campeão, novo conjunto de features, novo
# limiar); não sobe por refatoração que não muda a predição. É o que a API
# devolve em /health e em cada resposta de /predict: sem isso, o log de
# inferência da Etapa 10 não sabe qual modelo produziu cada linha, e a análise
# de drift fica sem denominador.
VERSAO_MODELO = "1.0.0"

# Seed única para todo o projeto. Sem isso, comparar dois runs é ilusão —
# a diferença pode ser só a partição que mudou.
SEED = 42

# Partição 60/20/20. TRÊS conjuntos, não dois:
#   treino     -> ajusta os parâmetros
#   validação  -> seleção de modelo, early stopping do MLP, gate do CI
#   teste      -> tocado UMA vez, no fim, só para reportar
# O teste ficar fora da seleção é o que impede o "test set overfitting":
# decidir 20 vezes olhando o teste transforma o teste em validação.
FRAC_TESTE = 0.20
FRAC_VAL = 0.20

TARGET = "Churn Value"

# --- Colunas fora das features (Etapa 1, §1 do decision log) ---------------

# Leakage: a informação não existiria no momento real da predição.
LEAKAGE = [
    "Churn Reason",   # só é preenchida DEPOIS do cancelamento; notna() acerta 100%
    "Churn Label",    # o alvo em texto
]

# Fora do treino por procedência opaca, mas guardadas para uso na avaliação.
AUDITORIA = [
    "Churn Score",    # score da IBM -> benchmark competidor, não feature
    "CLTV",           # valor do cliente -> ordena a fila (P(churn) x CLTV)
]

# Sem poder preditivo, ou proibidas.
DESCARTE = [
    "CustomerID",                      # identificador + dado pessoal (LGPD)
    "Count", "Country", "State",       # 1 único valor: variância zero
    "Lat Long",                        # redundante com Latitude/Longitude
]

# Geográficas: fora das features na v1 por cardinalidade (4,3 clientes por CEP)
# E por serem proxy de renda/raça. Guardadas para a auditoria de fairness.
GEOGRAFICAS = ["City", "Zip Code", "Latitude", "Longitude"]

# --- Features -------------------------------------------------------------

# 'Total Charges' é tratada em separado das demais numéricas: seu vazio é
# medição verdadeira (cliente sem ciclo de faturamento) e vira 0, não mediana.
NUM_ZERO = ["Total Charges"]
NUM = ["Tenure Months", "Monthly Charges"]

CAT_DISPONIVEIS = [
    # demográficas — mantidas nas features DE PROPÓSITO. Remover 'Senior Citizen'
    # não removeria o viés (proxies: Contract, Dependents), só a capacidade de
    # medi-lo. Ver "fairness through unawareness" no catálogo de armadilhas.
    # A Etapa 5 confirmou que mantê-las custa zero: 13 e 10 features dão o mesmo
    # PR-AUC. A decisão de tirá-las é de governança e cabe à Etapa 10.5, com o
    # dado da auditoria em mãos.
    "Gender", "Senior Citizen", "Partner", "Dependents",
    # contrato e serviços
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
]

# Removidas na Etapa 5 por ablação: cada uma medida individualmente e as seis em
# conjunto. Sem custo de performance (PR-AUC 0,6868 -> 0,6912) e com o desvio
# entre folds caindo de 0,0236 para 0,0163 — modelo mais estável.
# O ganho real não é métrica: são 6 colunas a menos para coletar, validar e
# vigiar contra drift em produção (ver Etapa 10).
REMOVIDAS_ETAPA5 = [
    "Phone Service",       # 90% têm; quase sem variância útil
    "Online Backup",       # redundante com Online Security / Tech Support
    "Device Protection",   # idem
    "Streaming TV",        # idem, e correlacionado com Streaming Movies
    "Streaming Movies",    # idem
    "Payment Method",      # 78% dos 'Electronic check' são mês-a-mês: o sinal
                           # já está em Contract (correlação espúria confirmada)
]

CAT = [c for c in CAT_DISPONIVEIS if c not in REMOVIDAS_ETAPA5]

FEATURES = NUM_ZERO + NUM + CAT

# Colunas mantidas ao lado do X para auditar sem contaminar o treino.
AUX = AUDITORIA + GEOGRAFICAS

# Capacidade operacional assumida na Etapa 0: o time trabalha 10-20% da base
# por ciclo. É nesses pontos que a métrica de negócio é lida.
KS_OPERACIONAIS = [0.10, 0.20]

# --- Gate do CI (Etapa 9.5) ------------------------------------------------

# Piso de PR-AUC na VALIDAÇÃO abaixo do qual o CI falha e nada é promovido.
#
# Três decisões embutidas neste número:
#
# 1. É medido na VALIDAÇÃO, não no teste. O enunciado da Aula 08 sugere o teste —
#    e seguir isso transformaria o teste em validação depois de alguns pushes:
#    ele deixaria de ser estimativa honesta de generalização. O sintoma é
#    traiçoeiro porque o gap treino-teste continua bonito.
#
# 2. É ABSOLUTO e comparativo contra o campeão corrente (LogReg 13 features,
#    0,6646), não relativo do tipo "80% do baseline" — que com 0,6646 aceitaria
#    0,53, um modelo pior que muitos que já rejeitamos.
#
# 3. A folga de ~0,005 cobre variação numérica entre plataformas (BLAS, versão de
#    biblioteca), NÃO degradação aceita. Mesmo modelo, mesma partição e mesma seed
#    deveriam dar o mesmo número; a folga existe porque "deveriam" não é "dão".
#
# ⚠️ Quando o campeão mudar (Etapas 7/8), este número sobe junto. Gate que não
# acompanha o campeão vira decoração.
GATE_PR_AUC_MIN = 0.66

# Segundo eixo do gate — CALIBRAÇÃO (item 41 do revisita, M02-A06 §5).
#
# Um gate de um eixo só aprova um modelo que subiu a PR-AUC e piorou o Brier —
# e isso não é hipótese: a Etapa 6 mediu que os dois podem andar em direções
# opostas (ranking bom com probabilidade descalibrada). O estrago é silencioso:
# probabilidade deslocada => o limiar de operação corta a fila no lugar errado
# => mais custo em reais, com o CI verde. Contratos reais de detecção
# especificam DUAS condições simultâneas ("sensibilidade >= X E especificidade
# >= Y") justamente por isso.
#
# Importa aqui porque a fila de retenção é ordenada por P(churn) x CLTV: a
# probabilidade é MULTIPLICADA por um valor em reais, então ela precisa
# significar alguma coisa, não só ordenar.
#
# 0,14 com o campeão em 0,1339: a mesma lógica de folga do piso de PR-AUC.
# Medidos na validação até aqui: LogReg 0,1339 · MLP (32,) 0,1344 ·
# HGB tunado 0,1351 · MLP 1-SE 0,1354 — nenhum barrado hoje, e é justamente
# por estar barato que o eixo entra antes de precisar dele.
GATE_BRIER_MAX = 0.14

GATE_MODELO_REFERENCIA = "logreg"

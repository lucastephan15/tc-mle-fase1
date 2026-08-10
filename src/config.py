"""
Configuração central — caminhos, seed e o contrato de colunas.

Existe para que a lista de colunas descartadas na Etapa 1 seja CÓDIGO, e não um
comentário no decision log. Toda decisão da §1 do log tem uma linha aqui.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "Telco_customer_churn.xlsx"
PROCESSED = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"

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

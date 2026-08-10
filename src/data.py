"""
Carregamento e particionamento — Etapa 2 · Data Preparation.

Responsabilidade única: entregar as três partições sempre iguais, com o alvo e as
colunas de auditoria separados das features. Nada aqui aprende parâmetro dos dados
(isso é trabalho do Pipeline, e só sobre o treino).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


@dataclass(frozen=True)
class Particao:
    """Um dos três conjuntos: features, alvo e as colunas guardadas para auditoria."""

    X: pd.DataFrame
    y: pd.Series
    aux: pd.DataFrame

    def __len__(self) -> int:
        return len(self.y)

    @property
    def taxa_churn(self) -> float:
        return float(self.y.mean())


@dataclass(frozen=True)
class Dados:
    treino: Particao
    validacao: Particao
    teste: Particao
    sha256: str


def sha256_dataset() -> str:
    """Hash do arquivo bruto — responde 'qual snapshot dos dados gerou este modelo?'."""
    h = hashlib.sha256()
    with open(config.RAW, "rb") as f:
        for bloco in iter(lambda: f.read(8192), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_bruto() -> pd.DataFrame:
    """Lê o Excel e corrige APENAS o que é defeito de formato do arquivo.

    'Total Charges' vem como texto porque 11 linhas trazem espaço em branco.
    A conversão para número é característica do .xlsx, não do fluxo de produção
    (na API o campo já chega como float, garantido pelo Pydantic). Por isso ela
    fica aqui — e o preenchimento do vazio NÃO: aquilo é regra de domínio e
    precisa viajar dentro do artefato serializado.
    """
    df = pd.read_excel(config.RAW)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Todas as numéricas como float64, inclusive 'Tenure Months' (que é int no
    # arquivo). Motivo não é estético: inteiro em Python não representa nulo, então
    # um campo faltando na API viraria float e quebraria a validação de schema do
    # MLflow — erro que só apareceria em produção, com dado real. Declarar double
    # desde o treino faz o contrato aceitar o nulo e o imputador tratá-lo.
    df[config.NUM_ZERO + config.NUM] = df[config.NUM_ZERO + config.NUM].astype("float64")
    return df


def dividir(df: pd.DataFrame | None = None) -> Dados:
    """Divide em treino/validação/teste, estratificado pelo alvo.

    Estratificar é obrigatório em base desbalanceada: sem isso as partições saem
    com taxas de churn diferentes entre si e a comparação entre modelos passa a
    medir a sorte do sorteio.

    Duas chamadas em vez de uma: primeiro separa o teste (20%), depois fatia o
    restante em treino e validação. A fração da segunda chamada é recalculada
    sobre o que sobrou, para que o resultado final seja 60/20/20 do total.
    """
    if df is None:
        df = carregar_bruto()

    X = df[config.FEATURES]
    y = df[config.TARGET]
    aux = df[config.AUX]

    idx_resto, idx_teste = train_test_split(
        df.index,
        test_size=config.FRAC_TESTE,
        stratify=y,
        random_state=config.SEED,
    )
    frac_val = config.FRAC_VAL / (1 - config.FRAC_TESTE)  # 0.20/0.80 = 0.25
    idx_treino, idx_val = train_test_split(
        idx_resto,
        test_size=frac_val,
        stratify=y.loc[idx_resto],
        random_state=config.SEED,
    )

    def montar(idx) -> Particao:
        return Particao(X=X.loc[idx], y=y.loc[idx], aux=aux.loc[idx])

    return Dados(
        treino=montar(idx_treino),
        validacao=montar(idx_val),
        teste=montar(idx_teste),
        sha256=sha256_dataset(),
    )


if __name__ == "__main__":
    dados = dividir()
    print(f"sha256 do dataset : {dados.sha256}")
    print(f"{'partição':<12} {'n':>6} {'churn':>8}")
    for nome in ("treino", "validacao", "teste"):
        p = getattr(dados, nome)
        print(f"{nome:<12} {len(p):>6} {p.taxa_churn:>7.2%}")
    print(f"\nfeatures: {len(config.FEATURES)} "
          f"({len(config.NUM_ZERO + config.NUM)} numéricas, {len(config.CAT)} categóricas)")

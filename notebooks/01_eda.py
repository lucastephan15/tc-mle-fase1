"""
EDA — Etapa 1 · Data Understanding

Reproduz todos os achados registrados em docs/decision-log.md §1.
Exploração, não produção: nada aqui é importado pelo pipeline.

Uso:  python notebooks/01_eda.py
"""

import hashlib
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco_customer_churn.xlsx"
TARGET = "Churn Value"


def sha256(path: Path) -> str:
    """Hash do arquivo bruto — responde 'qual snapshot dos dados gerou este modelo?'."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(8192), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar() -> pd.DataFrame:
    df = pd.read_excel(RAW)
    # Total Charges vem como texto; os 11 vazios são clientes com tenure=0 (sem ciclo
    # de faturamento). O vazio é medição verdadeira → 0, não mediana. Ver §1 do log.
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)
    return df


def secao(titulo: str) -> None:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


def main() -> None:
    secao("PROCEDÊNCIA")
    print(f"arquivo : {RAW.name}")
    print(f"sha256  : {sha256(RAW)}")

    df = carregar()
    print(f"shape   : {df.shape[0]} linhas x {df.shape[1]} colunas")

    secao("ALVO E BASELINE DE CHANCE")
    taxa = df[TARGET].mean()
    print(f"taxa de churn      : {taxa:.2%}  ({df[TARGET].sum()} churners)")
    print(f"baseline de chance : {1 - taxa:.2%}  (chutar sempre 'No')")
    print("→ todo resultado deve ser lido contra esse piso, não contra 0%.")

    secao("LEAKAGE — Churn Reason prediz o alvo pela EXISTÊNCIA")
    print("nulos entre churners     :", df.loc[df[TARGET] == 1, "Churn Reason"].isna().sum())
    print("nulos entre não-churners :", df.loc[df[TARGET] == 0, "Churn Reason"].isna().sum())
    print("→ notna() acerta 100%. Descartada.")

    secao("LEAKAGE — Churn Score separa quase perfeitamente")
    print(df.groupby("Churn Label")["Churn Score"].agg(["mean", "min", "max"]).round(1))
    print("→ usá-la seria prever o modelo da IBM, não churn. Vira benchmark.")

    secao("COLUNAS DEGENERADAS (variância zero)")
    print([c for c in df.columns if df[c].nunique() == 1])

    secao("DRIVERS DE CHURN — amplitude entre categorias")
    categoricas = [
        "Contract", "Internet Service", "Payment Method", "Online Security",
        "Tech Support", "Paperless Billing", "Senior Citizen", "Gender",
        "Partner", "Dependents",
    ]
    linhas = []
    for col in categoricas:
        t = df.groupby(col)[TARGET].mean() * 100
        linhas.append({"variavel": col, "amplitude_pp": round(t.max() - t.min(), 1),
                       "min_%": round(t.min(), 1), "max_%": round(t.max(), 1)})
    ranking = pd.DataFrame(linhas).sort_values("amplitude_pp", ascending=False)
    print(ranking.to_string(index=False))
    print("\n→ Contract domina (39,9 pp). Gender é ruído (0,7 pp) — mas ainda assim")
    print("  precisa ser auditado: ausência de sinal ≠ ausência de disparidade.")

    secao("TENURE — a não-linearidade que justifica binning")
    faixas = pd.cut(df["Tenure Months"], [-1, 6, 12, 24, 48, 72],
                    labels=["0-6m", "6-12m", "1-2a", "2-4a", "4a+"])
    print((df.groupby(faixas, observed=True)[TARGET]
             .agg(churn_pct=lambda s: round(s.mean() * 100, 1), n="size")).to_string())
    print("→ 52,9% → 9,5%, monotônico e curvo. Binning entrega isso de mão beijada")
    print("  para LogReg e MLP; é redundante para árvores.")

    secao("PREVALÊNCIA POR GRUPO SENSÍVEL — insumo da Etapa 10.5")
    for col in ["Senior Citizen", "Gender", "Partner", "Dependents"]:
        t = df.groupby(col)[TARGET].mean() * 100
        print(f"{col:<16} {t.round(1).to_dict()}   amplitude: {t.max() - t.min():.1f} pp")
    print("\n→ Senior Citizen: 41,7% x 23,6%. Prevalência REAL diferente entre grupos")
    print("  torna paridade demográfica, equalized odds e calibração mutuamente")
    print("  incompatíveis (teorema de impossibilidade). A Etapa 10.5 terá de")
    print("  DECLARAR qual definição de justiça otimiza — não dá para as três.")

    secao("NUMÉRICAS — outliers e correlação")
    for col in ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]:
        s = df[col]
        z = ((s - s.mean()) / s.std()).abs()
        print(f"{col:<16} min={s.min():>8.1f} max={s.max():>8.1f} "
              f"|z|>3: {(z > 3).sum():>3}  corr_alvo={s.corr(df[TARGET]):>6.3f}")
    print("→ nenhum outlier univariado; correlações lineares modestas.")
    print("  Nenhuma variável isolada resolve — o sinal está nas combinações,")
    print("  o que é um argumento a favor do MLP.")

    secao("REDUNDÂNCIA — Total Charges ≈ Tenure x Monthly")
    ok = df["Tenure Months"] > 0
    estimado = df["Tenure Months"] * df["Monthly Charges"]
    corr = df.loc[ok, "Total Charges"].corr(estimado[ok])
    erro = ((df.loc[ok, "Total Charges"] - estimado[ok]) / df.loc[ok, "Total Charges"]).abs()
    print(f"correlação            : {corr:.4f}")
    print(f"erro relativo mediano : {erro.median():.2%}")
    print("→ multicolinearidade: atrapalha LogReg, indiferente para árvores. Etapa 5.")

    secao("HIPÓTESE TESTADA E REJEITADA — reajuste de preço")
    razao = (df.loc[ok, "Total Charges"] / df.loc[ok, "Tenure Months"]) / df.loc[ok, "Monthly Charges"]
    print(f"mediana da razão histórica/atual : {razao.median():.3f}")
    print(f"desvio-padrão                    : {razao.std():.3f}")
    quartis = df.loc[razao.index].assign(q=pd.qcut(razao, 4))
    print(quartis.groupby("q", observed=True)[TARGET].mean().round(3).to_string())
    print("→ sem padrão monotônico. O dataset é estático demais para carregar")
    print("  histórico de reajuste. Feature descartada ANTES de ser construída.")

    secao("ARMADILHA CONFIRMADA — divisão por tenure gera inf")
    n_zero = int((df["Tenure Months"] == 0).sum())
    print(f"linhas com Tenure Months = 0: {n_zero}")
    print("→ qualquer feature com tenure no denominador produz inf nessas linhas.")
    print("  Vira teste unitário obrigatório no CI (Etapa 9.5).")


if __name__ == "__main__":
    main()

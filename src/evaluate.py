"""
Avaliação — as métricas decididas na Etapa 0, em código.

A métrica primária é PR-AUC porque o artefato é um RANQUEADOR, não um
classificador: a pergunta do negócio é "quem eu ligo primeiro?", não "esse
cliente é churn: sim ou não?". Métrica que depende de limiar responderia à
pergunta errada.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


def recall_at_k(y_true, y_score, k: float) -> float:
    """Fração dos churners capturada nos k% clientes mais bem ranqueados.

    É a métrica que o gerente de retenção entende: "se a equipe consegue
    trabalhar 20% da base por ciclo, quantos dos que iam sair eu pego?".

    Empates no score são desempatados arbitrariamente pela ordenação — com
    scores contínuos isso é irrelevante, mas seria um problema num modelo que
    produzisse muitos valores idênticos (uma árvore rasa, por exemplo).
    """
    y_true = np.asarray(y_true)
    n = int(np.ceil(k * len(y_true)))
    topo = np.argsort(np.asarray(y_score))[::-1][:n]
    return float(y_true[topo].sum() / y_true.sum())


def avaliar(y_true, y_score, limiar: float = 0.5) -> dict[str, float]:
    """Pacote completo de métricas para um conjunto.

    O limiar de 0,5 aparece aqui só para produzir a matriz de confusão e as
    métricas pontuais — é um valor de referência, NÃO o limiar de operação.
    Quem define o corte real é a capacidade da equipe, e ele pode mudar sem
    retreinar nada. Por isso a API devolverá probabilidade, não classe.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    m = {
        # --- primária: independente de limiar ---
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        # --- calibração: a fila é ordenada por P(churn) x CLTV, então a
        # probabilidade precisa significar alguma coisa, não só ordenar ---
        "brier": brier_score_loss(y_true, y_score),
        # --- pontuais, ao limiar de referência ---
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # falso negativo = o cliente que ia sair e passou batido (R$ 194)
        # falso positivo = demos atenção a quem ia ficar de qualquer jeito (R$ 62)
        "fn": int(fn),
        "fp": int(fp),
        "tp": int(tp),
        "tn": int(tn),
        "custo_erro_brl": float(fn * 194 + fp * 62),
    }

    # --- operacionais: a curva de ganho nos pontos de capacidade da equipe ---
    prevalencia = y_true.mean()
    for k in config.KS_OPERACIONAIS:
        r = recall_at_k(y_true, y_score, k)
        m[f"recall_at_{int(k * 100)}"] = r
        # lift = quantas vezes melhor que sortear k% da base ao acaso
        m[f"lift_at_{int(k * 100)}"] = r / k if k else float("nan")
        # Teto ESTRUTURAL: nos k% do topo cabem no máximo k*N clientes, então
        # nenhum ranqueador pode capturar mais que k/prevalência dos churners.
        # Com prevalência 26,5%, recall@10% nunca passa de 0,377 — ler o número
        # cru sem isso faz um ranking perfeito parecer medíocre.
        teto = min(k / prevalencia, 1.0)
        m[f"recall_at_{int(k * 100)}_pct_teto"] = r / teto
    m["prevalencia"] = float(prevalencia)
    return m


def curva_custo(y_true, y_score, custo_fn: int = 194, custo_fp: int = 62) -> dict:
    """Varre os limiares e devolve o que minimiza o custo esperado do erro.

    Existe para sustentar em número a tese da Etapa 0: o limiar é parâmetro DE
    NEGÓCIO, não do modelo. Quem escolhe o corte é a economia da operação
    (R$ 194 por churner perdido × R$ 62 por atenção desperdiçada), e mudá-lo não
    exige retreinar nada — é o argumento para a API devolver probabilidade e não
    classe.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    limiares = np.unique(np.round(np.linspace(0.01, 0.99, 99), 2))
    custos = []
    for t in limiares:
        y_pred = (y_score >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        custos.append(fn * custo_fn + fp * custo_fp)
    i = int(np.argmin(custos))
    return {
        "limiar_otimo": float(limiares[i]),
        "custo_otimo_brl": float(custos[i]),
        "custo_em_0.5": float(custos[list(limiares).index(0.5)]),
    }


def formatar(nome: str, m: dict[str, float]) -> str:
    """Uma linha legível no terminal — o que se olha durante a execução."""
    return (
        f"{nome:<12} "
        f"PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
        f"R@10% {m['recall_at_10']:.3f} ({m['recall_at_10_pct_teto']:>5.1%} do teto) | "
        f"R@20% {m['recall_at_20']:.3f} ({m['recall_at_20_pct_teto']:>5.1%}) | "
        f"F1 {m['f1']:.3f} | FN {m['fn']:>4} FP {m['fp']:>4}"
    )

"""
Treino — Etapa 3 · Baseline / MVP.

Executável e parametrizável de propósito (não notebook): é isto que o CI vai
importar e rodar, e é o que permite reproduzir um número meses depois.

    python -m src.train --modelo majoritaria
    python -m src.train --modelo ibm_score
    python -m src.train --modelo logreg
    python -m src.train --modelo logreg --class-weight balanced

DECISÃO METODOLÓGICA — divergência deliberada do enunciado da Aula 08:
toda seleção de modelo é feita na VALIDAÇÃO. O conjunto de teste não é tocado
aqui e só será avaliado uma vez, ao fim do projeto, para reportar. Decidir vinte
vezes olhando o teste transformaria o teste em validação, e ele deixaria de ser
estimativa honesta de generalização. O sintoma é traiçoeiro porque o gap
treino-teste continua bonito: não é o modelo que aprendeu mal, é o instrumento
de medição que foi gasto pelo uso.
"""

from __future__ import annotations

import argparse
import subprocess

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src import config, data, evaluate, features
from src.preprocess import construir_pipeline

EXPERIMENTO = "churn-fase1"


def commit_hash() -> str:
    """Linhagem: qual commit gerou este modelo. Uma linha, e o run fica rastreável."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.RAIZ, capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconhecido"


def construir_modelo(nome: str, class_weight: str | None):
    """Fábrica dos estimadores comparáveis nesta etapa."""
    if nome == "majoritaria":
        # Piso absoluto: prevê sempre a prevalência do treino. Serve para
        # responder "o modelo bate o chute na classe majoritária, e por quanto?".
        return DummyClassifier(strategy="prior", random_state=config.SEED), False
    if nome == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=config.SEED,
        ), True
    raise ValueError(f"modelo desconhecido: {nome}")


def pontuar_ibm_score(dados: data.Dados) -> dict[str, np.ndarray]:
    """Benchmark externo: o Churn Score que veio pronto da IBM.

    Não é um modelo nosso — é um concorrente. Ele foi barrado como feature na
    Etapa 1 (usá-lo seria prever o modelo da IBM, não churn), mas como régua
    externa é legítimo e informativo: dá uma noção de teto.

    ⚠️ Ressalva registrada: se o score foi calculado retrospectivamente sobre a
    base já com o desfecho conhecido, o benchmark é inatingível por construção.
    Procedência desconhecida -> ler o número com desconfiança, não como meta.
    """
    # /100 apenas para pôr na escala [0,1]; NÃO é probabilidade calibrada,
    # então o Brier e o limiar ótimo deste run não são interpretáveis.
    return {
        nome: getattr(dados, nome).aux["Churn Score"].to_numpy() / 100
        for nome in ("treino", "validacao")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino do TC Fase 1 — churn telecom")
    ap.add_argument("--modelo", default="logreg",
                    choices=["majoritaria", "logreg", "ibm_score"])
    ap.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    ap.add_argument("--features", action="store_true",
                    help="liga as 4 features derivadas da Etapa 4")
    ap.add_argument("--tag", default="", help="rótulo livre para identificar o run")
    args = ap.parse_args()

    np.random.seed(config.SEED)
    dados = data.dividir()

    # Backend SQLite, não o file store: o MLflow 3 pôs './mlruns' em modo de
    # manutenção, e o Model Registry (Etapa 9) nunca funcionou sobre arquivos.
    # Decidir isso agora evita migrar experimentos no meio do projeto.
    mlflow.set_tracking_uri(f"sqlite:///{config.RAIZ / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENTO)

    nome_run = args.modelo + (f"-{args.class_weight}" if args.class_weight else "")
    nome_run += "+feat" if args.features else ""
    with mlflow.start_run(run_name=nome_run + (f"-{args.tag}" if args.tag else "")):
        # Linhagem: dataset + commit + seed respondem "como reproduzo este número?"
        mlflow.set_tags({
            "dataset_sha256": dados.sha256,
            "commit": commit_hash(),
            "etapa": "3-baseline",
            "avaliado_em": "validacao",  # o teste segue intocado
        })
        mlflow.log_params({
            "modelo": args.modelo,
            "class_weight": args.class_weight or "none",
            "seed": config.SEED,
            "n_features": len(config.FEATURES) + (4 if args.features else 0),
            "features_etapa4": args.features,
            "n_treino": len(dados.treino),
            "n_validacao": len(dados.validacao),
        })

        if args.modelo == "ibm_score":
            scores = pontuar_ibm_score(dados)
        else:
            modelo, escalonar = construir_modelo(args.modelo, args.class_weight)
            novas = (features.NOVAS_NUM + features.NOVAS_CAT) if args.features else []
            pipe = construir_pipeline(modelo, escalonar=escalonar, novas=novas)
            # fit SÓ no treino. Mediana do imputador, média/desvio do scaler e
            # categorias do encoder são aprendidas aqui e em nenhum outro lugar.
            pipe.fit(dados.treino.X, dados.treino.y)
            scores = {
                nome: pipe.predict_proba(getattr(dados, nome).X)[:, 1]
                for nome in ("treino", "validacao")
            }

            assinatura = infer_signature(dados.treino.X, pipe.predict_proba(dados.treino.X))
            mlflow.sklearn.log_model(
                sk_model=pipe, name="model",
                # signature + input_example são o CONTRATO do modelo: registram
                # quais colunas, em que tipos e em que ordem. É o que a API vai
                # ter de honrar, e o que separa "convenção implícita" de contrato.
                signature=assinatura, input_example=dados.treino.X.head(5),
                # O MLflow 3 serializa com skops em vez de pickle — skops não
                # executa código arbitrário ao carregar, o que importa quando o
                # artefato vem de um registry. Em troca, exige declarar os tipos
                # não triviais. numpy.dtype vem dos dtypes do ColumnTransformer.
                # 'src.features.adicionar_features' precisa ser declarada porque
                # o FunctionTransformer carrega uma função NOSSA. Consequência que
                # vale para a Etapa 9: o artefato fica ACOPLADO AO CÓDIGO-FONTE —
                # o ambiente de inferência tem de ter `src.features` importável no
                # mesmo caminho, senão o modelo não carrega. É a versão skops do
                # problema clássico do pickle com funções customizadas.
                skops_trusted_types=["numpy.dtype", "src.features.adicionar_features"],
            )

        resultados = {
            nome: evaluate.avaliar(getattr(dados, nome).y, s)
            for nome, s in scores.items()
        }
        for conjunto, m in resultados.items():
            mlflow.log_metrics({f"{conjunto}_{k}": v for k, v in m.items()})

        # Sinal de overfitting: quanto a métrica primária cai do treino para a
        # validação. Vale lembrar que este gap NÃO detecta leakage — se a
        # contaminação atingiu os dois lados igualmente, ele continua bonito.
        gap = resultados["treino"]["pr_auc"] - resultados["validacao"]["pr_auc"]
        mlflow.log_metric("gap_pr_auc", gap)

        # Limiar de operação escolhido pela economia, na validação. Reportado
        # junto para deixar explícito que o corte é decisão de negócio.
        custo = evaluate.curva_custo(dados.validacao.y, scores["validacao"])
        mlflow.log_metrics({f"val_{k}": v for k, v in custo.items()})

        print(f"\n{'=' * 78}\n{nome_run}   (run {mlflow.active_run().info.run_id[:8]})\n{'=' * 78}")
        for conjunto, m in resultados.items():
            print(evaluate.formatar(conjunto, m))
        print(f"{'gap PR-AUC':<12} {gap:+.4f}")
        print(f"{'custo do erro':<12} R$ {custo['custo_em_0.5']:>9,.0f} no limiar 0,5"
              f"   ->   R$ {custo['custo_otimo_brl']:>9,.0f} no limiar ótimo "
              f"({custo['limiar_otimo']:.2f})")


if __name__ == "__main__":
    main()

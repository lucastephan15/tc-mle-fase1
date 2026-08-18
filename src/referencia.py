"""
Estatísticas de referência do treino — Etapa 10a-2 · o denominador do drift.

    Sem baseline não existe drift para detectar, só um gráfico comparando nada.

Este módulo tem duas metades que **precisam** morar juntas: `calcular()`, que
congela a distribuição do treino, e `psi()`/`comparar()`, que a confrontam com
uma janela de produção. Separá-las é o convite para alguém rebinar do outro
lado — e PSI com bins recalculados na janela compara duas escalas diferentes,
o que faz o número parar de significar o que a regra de bolso diz que significa
(`<0,10` estável · `0,10–0,25` investigar · `>0,25` agir).

🔑 **Por que a referência viaja DENTRO do artefato** (item 109 do revisita,
decidido em 18/08/2026): a distribuição de referência é propriedade *daquele*
modelo, não do repositório — muda quando o modelo muda, exatamente como o
limiar de operação, que já viaja dentro pelo mesmo argumento. A alternativa
(`models/referencia.json` ao lado) criaria o par que a regra nº 1 do 9c existe
para proibir, e aqui a falha é pior que rótulo errado: baseline de um modelo
comparado com predições de outro **não falha, só mede errado** — drift fantasma,
ou drift real que não aparece. É também o que torna o `artefato_sha256` da linha
de log verificável em vez de decorativo.

🚨 **Consequência prática que decide o formato: o baseline é auto-suficiente.**
Guardamos as proporções esperadas por bin, não os dados. Quem calcula PSI em
produção **não precisa do dataset de treino** — que é justamente o que não pode
estar na imagem (`data/raw` é LGPD e o `.dockerignore` o barra). Uma referência
que exigisse reler o Excel seria uma referência que não roda onde o drift
acontece.

⚠️ **A referência é sobre o X CRU, não sobre o pré-processado.** O que o log de
inferência registra é o payload de entrada (`Contract: "Month-to-month"`), não a
matriz de 30 e poucas colunas que sai do `ColumnTransformer`. Baseline calculado
depois do pipeline compararia com o que não existe do outro lado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 10 bins equifrequentes: o padrão do PSI, e com decis a proporção esperada é
# ~0,10 em cada — o que torna qualquer desvio legível a olho na tabela.
N_BINS = 10

# Casas guardadas. 6 é MUITO mais resolução do que o PSI usa (ele agrega em 10
# faixas) e é o mesmo arredondamento do `scores` no log — os dois lados da
# comparação com a mesma régua, de propósito.
CASAS = 6

# Regra de bolso do PSI. Ficam aqui, e não no script de análise, porque quem lê
# o número tem de ler o corte junto: PSI é adimensional e não se interpreta
# sozinho.
PSI_ESTAVEL = 0.10
PSI_AGIR = 0.25

# Piso para proporção zerada. Um bin vazio na janela atual daria `ln(0)` = -inf,
# e um inf contamina a soma inteira: a feature deixaria de ter número em vez de
# ter um número grande. Medido com este epsilon e bin esperado de 0,10, um bin
# que ZERA em produção contribui 1,16 sozinho — muito acima de 0,25, que é o
# comportamento certo (categoria que sumiu é drift grave), e finito.
EPSILON = 1e-6


def _proporcoes(valores: np.ndarray, bordas: list[float]) -> list[float]:
    """Fração da amostra em cada bin definido por `bordas`.

    🚨 `np.digitize` com as bordas INTERNAS (p10…p90) produz `len(bordas)+1`
    faixas, e as duas das pontas são **abertas por construção**. Isso não é
    detalhe de implementação: produção tem valores fora do intervalo visto no
    treino (é literalmente um dos sintomas de drift), e binning fechado faria
    esses valores **sumirem da contagem** — a distribuição mais deslocada
    possível daria o PSI mais tranquilo, porque o que saiu do intervalo não
    seria contado.
    """
    if valores.size == 0:
        return [0.0] * (len(bordas) + 1)
    indices = np.digitize(valores, bordas, right=False)
    contagem = np.bincount(indices, minlength=len(bordas) + 1)
    return (contagem / valores.size).tolist()


def calcular(X: pd.DataFrame, num: list[str], cat: list[str],
             scores: np.ndarray | None = None, limiar: float | None = None,
             n_bins: int = N_BINS) -> dict:
    """Congela as duas distribuições de referência — o que vai dentro do artefato.

    `X` é o que ENTRA (data drift) e `scores` é o que SAI (prediction drift), e
    as duas metades vêm de **partições diferentes de propósito**:

    - `X` = `dados.treino.X`. Validação e teste julgam o modelo; usar a base
      inteira faria o baseline conter as partições em que ele foi medido,
      misturando "o que o modelo viu" com "o que o mediu".
    - `scores` = probabilidades na **validação**, não no treino. In-sample o
      modelo é otimista por construção, e um baseline otimista faria produção
      parecer deslocada para sempre — drift fantasma permanente, que treina o
      time a ignorar o alerta. Fora da amostra é o que se parece com produção.
      ⚠️ Medido em 18/08/2026 neste campeão: PSI treino × validação nos scores =
      **0,0022**, ou seja, a escolha custaria quase nada aqui (LogReg
      regularizada com 13 features mal overfitta). A decisão é por princípio, e
      a medição diz que ela é **barata**, não que é dispensável: com um HGB
      profundo no lugar, o mesmo erro não sairia de graça.

    🚨 **Por que `scores` não é opcional na prática:** é a única vigilância que
    roda **sempre** em produção. As features só entram no log com
    `TC_LOG_FEATURES=1` (custam `Gender`/`Senior Citizen`/`Partner`/`Dependents`
    no stream de um terceiro); as probabilidades vão em toda linha, porque
    número sem atributo ao lado não identifica ninguém. Congelar só `P(X)` seria
    dar baseline exatamente à metade que quase nunca é coletada, e deixar sem
    baseline a que sempre é.
    """
    numericas: dict[str, dict] = {}
    for coluna in num:
        serie = pd.to_numeric(X[coluna], errors="coerce")
        valores = serie.dropna().to_numpy(dtype=float)
        # As bordas internas dos decis. `np.unique` porque quantis empatam em
        # coluna com massa concentrada num único valor, e borda repetida gera bin
        # de largura zero — que nunca recebe ninguém, tem proporção esperada 0 e
        # envenena a razão do PSI.
        #
        # ⚠️ Medido em 18/08/2026: **nenhuma das 3 numéricas de hoje empata**
        # (9 decis distintos em cada, `Total Charges` sem nenhum zero — o vazio
        # de quem não teve ciclo de faturamento é NaN no dado cru e só vira 0
        # dentro do pipeline). Ou seja, esta linha é proteção para o dia em que
        # entrar uma coluna com massa concentrada, e o dado atual **não a
        # exercita** — por isso o teste que a cobre é sintético e explícito, em
        # vez de confiar que o dataset a exercitaria.
        quantis = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        bordas = np.unique(np.round(np.quantile(valores, quantis), CASAS)).tolist()
        numericas[coluna] = {
            "n": int(valores.size),
            # Ausente é sinal, não sujeira: campo que começa a faltar é falha
            # upstream, e ela aparece aqui antes de aparecer na métrica.
            "faltantes": round(float(serie.isna().mean()), CASAS),
            "media": round(float(valores.mean()), CASAS),
            "desvio": round(float(valores.std(ddof=1)), CASAS),
            "min": round(float(valores.min()), CASAS),
            "max": round(float(valores.max()), CASAS),
            "quantis": {
                f"p{int(q * 100):02d}": round(float(np.quantile(valores, q)), CASAS)
                for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
            },
            "bordas_psi": bordas,
            "proporcoes_psi": [round(p, CASAS) for p in _proporcoes(valores, bordas)],
        }

    categoricas: dict[str, dict] = {}
    for coluna in cat:
        serie = X[coluna]
        freq = serie.dropna().astype(str).value_counts(normalize=True)
        categoricas[coluna] = {
            "n": int(serie.notna().sum()),
            "faltantes": round(float(serie.isna().mean()), CASAS),
            # Ordenado por nome, não por frequência: o dict entra num artefato
            # versionado, e ordem que depende do dado faz o mesmo baseline
            # produzir bytes diferentes conforme empates.
            "frequencias": {
                str(k): round(float(v), CASAS) for k, v in sorted(freq.items())
            },
        }

    saida = {
        "particao": "treino",
        "n": len(X),
        "n_bins": n_bins,
        "numericas": numericas,
        "categoricas": categoricas,
    }
    if scores is not None:
        saida["scores"] = _bloco_scores(np.asarray(scores, dtype=float),
                                        limiar=limiar, n_bins=n_bins)
    return saida


def _bloco_scores(scores: np.ndarray, limiar: float | None, n_bins: int) -> dict:
    """Baseline de prediction drift — a distribuição de saída do modelo.

    🔑 Guarda também a **taxa acima do limiar**, que é o número que o negócio
    entende: é o tamanho da fila de retenção. Prediction drift de PSI 0,3 exige
    um analista para virar decisão; *"a fila dobrou"* não exige nenhum, e as duas
    frases descrevem o mesmo evento. Métrica de monitoramento que só existe em
    unidade estatística é métrica que ninguém age em cima.
    """
    quantis = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    bordas = np.unique(np.round(np.quantile(scores, quantis), CASAS)).tolist()
    bloco = {
        "particao": "validacao",
        "n": int(scores.size),
        "media": round(float(scores.mean()), CASAS),
        "quantis": {
            f"p{int(q * 100):02d}": round(float(np.quantile(scores, q)), CASAS)
            for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
        },
        "bordas_psi": bordas,
        "proporcoes_psi": [round(p, CASAS) for p in _proporcoes(scores, bordas)],
    }
    if limiar is not None:
        bloco["limiar"] = round(float(limiar), CASAS)
        bloco["taxa_acima_do_limiar"] = round(float((scores >= limiar).mean()), CASAS)
    return bloco


def comparar_scores(referencia: dict, scores: list[float]) -> dict:
    """Prediction drift: a janela de saídas contra o baseline da validação.

    Roda **sem rótulo e em tempo real**, o que a qualidade preditiva não
    consegue: acurácia depende de ground truth, e o churn só se confirma no fim
    do ciclo de faturamento — a janela cega. Prediction drift é sintoma e não
    causa, mas é sintoma de graça e disponível hoje.
    """
    ref = referencia.get("scores")
    if not ref:
        raise KeyError("referência sem baseline de scores — artefato pré-10a-2?")

    valores = np.asarray(scores, dtype=float)
    if valores.size == 0:
        return {"n": 0, "psi": None, "classificacao": "sem-dados"}

    atuais = _proporcoes(valores, ref["bordas_psi"])
    valor = psi(ref["proporcoes_psi"], atuais)
    saida = {
        "n": int(valores.size),
        "psi": round(valor, CASAS),
        "classificacao": classificar(valor),
        "media_ref": ref["media"],
        "media_janela": round(float(valores.mean()), CASAS),
    }
    if "limiar" in ref:
        taxa = float((valores >= ref["limiar"]).mean())
        saida["taxa_acima_do_limiar_ref"] = ref["taxa_acima_do_limiar"]
        saida["taxa_acima_do_limiar"] = round(taxa, CASAS)
        # Quantas vezes a fila de retenção cresceu. O sinal que se leva para uma
        # reunião sem traduzir.
        if ref["taxa_acima_do_limiar"] > 0:
            saida["fila_x"] = round(taxa / ref["taxa_acima_do_limiar"], 3)
    return saida


def psi(esperadas: list[float], atuais: list[float]) -> float:
    """Population Stability Index entre duas distribuições já binadas.

    `Σ (a_i − e_i) · ln(a_i / e_i)` — simétrico e adimensional. Preferido ao
    p-valor de KS de propósito: com amostra grande **tudo** dá significativo, e
    a pergunta útil não é *"é diferente?"*, é *"é diferente o suficiente para
    importar?"*.
    """
    e = np.clip(np.asarray(esperadas, dtype=float), EPSILON, None)
    a = np.clip(np.asarray(atuais, dtype=float), EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


def classificar(valor: float) -> str:
    """Regra de bolso do PSI, num lugar só."""
    if valor < PSI_ESTAVEL:
        return "estavel"
    if valor < PSI_AGIR:
        return "investigar"
    return "agir"


def comparar(referencia: dict, X: pd.DataFrame) -> dict[str, dict]:
    """Confronta uma janela de produção com o baseline congelado.

    Devolve um dict por feature com `psi`, a classificação e o `n` da janela.
    Colunas ausentes em `X` são **omitidas**, não zeradas: janela que não trouxe
    a coluna é outro problema (contrato), e reportá-la como PSI 0 diria
    "estável" sobre o que nem foi medido.
    """
    saida: dict[str, dict] = {}

    for coluna, ref in referencia.get("numericas", {}).items():
        if coluna not in X.columns:
            continue
        valores = pd.to_numeric(X[coluna], errors="coerce").dropna().to_numpy(float)
        atuais = _proporcoes(valores, ref["bordas_psi"])
        valor = psi(ref["proporcoes_psi"], atuais)
        saida[coluna] = {
            "tipo": "numerica",
            "psi": round(valor, CASAS),
            "classificacao": classificar(valor),
            "n": int(valores.size),
            "media_ref": ref["media"],
            "media_janela": round(float(valores.mean()), CASAS) if valores.size else None,
        }

    for coluna, ref in referencia.get("categoricas", {}).items():
        if coluna not in X.columns:
            continue
        serie = X[coluna].dropna().astype(str)
        # 🚨 O universo de categorias é o DA REFERÊNCIA, mais as inéditas da
        # janela. Categoria nova é um dos sinais que a Etapa 10 tem de pegar
        # (o modelo nunca a viu; o `OneHotEncoder` a trata como desconhecida e
        # prediz mesmo assim), então ela precisa entrar na conta com proporção
        # esperada ~0 — o que faz o PSI saltar, que é o alarme certo.
        universo = sorted(set(ref["frequencias"]) | set(serie.unique()))
        esperadas = [ref["frequencias"].get(c, 0.0) for c in universo]
        vistas = serie.value_counts(normalize=True)
        atuais = [float(vistas.get(c, 0.0)) for c in universo]
        valor = psi(esperadas, atuais)
        ineditas = sorted(set(serie.unique()) - set(ref["frequencias"]))
        saida[coluna] = {
            "tipo": "categorica",
            "psi": round(valor, CASAS),
            "classificacao": classificar(valor),
            "n": int(serie.size),
            "categorias_ineditas": ineditas,
        }

    return saida

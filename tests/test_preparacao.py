"""
Testes das Etapas 2 e 3.

Cada teste protege uma DECISÃO registrada no decision log — não são testes de
cobertura, são travas contra os erros específicos que este projeto pode cometer.
O comentário de cada um diz qual armadilha ele fecha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config, data
from src.preprocess import construir_pipeline, construir_preprocessador


@pytest.fixture(scope="module")
def bruto() -> pd.DataFrame:
    return data.carregar_bruto()


@pytest.fixture(scope="module")
def dados() -> data.Dados:
    return data.dividir()


# --- Anti-leakage ----------------------------------------------------------

def test_colunas_de_leakage_fora_das_features():
    """'Churn Reason' só existe DEPOIS do cancelamento; notna() acerta 100%.

    Este é o teste que impede o erro mais caro do projeto voltar por descuido
    numa refatoração futura.
    """
    proibidas = set(config.LEAKAGE + config.AUDITORIA + config.DESCARTE) | {config.TARGET}
    assert not proibidas & set(config.FEATURES)


def test_alvo_nao_vaza_para_dentro_do_X(dados):
    """O alvo não pode estar entre as features, nem sob outro nome."""
    assert config.TARGET not in dados.treino.X.columns
    assert "Churn Label" not in dados.treino.X.columns


def test_geograficas_ficam_fora_das_features():
    """Fora por cardinalidade (4,3 clientes por CEP) e por serem proxy de renda/raça."""
    assert not set(config.GEOGRAFICAS) & set(config.FEATURES)


def test_nenhum_cliente_em_duas_particoes(dados):
    """Duplicata do mesmo cliente em treino e teste infla a métrica silenciosamente."""
    idx = [set(getattr(dados, n).X.index) for n in ("treino", "validacao", "teste")]
    assert idx[0] & idx[1] == set()
    assert idx[0] & idx[2] == set()
    assert idx[1] & idx[2] == set()


# --- Schema e contrato dos dados -------------------------------------------

def test_schema_do_dataset_bruto(bruto):
    """Se a fonte mudar de forma, é melhor falhar aqui que treinar sobre lixo."""
    assert len(bruto) == 7043
    assert set(config.FEATURES).issubset(bruto.columns)
    assert config.TARGET in bruto.columns


def test_numericas_sao_float_para_aceitar_nulo(bruto):
    """Inteiro em Python não representa nulo: um campo faltando na API viraria
    float e quebraria a validação de schema do MLflow em produção."""
    for col in config.NUM_ZERO + config.NUM:
        assert bruto[col].dtype == np.float64, col


def test_dominio_das_numericas(bruto):
    """Validação de dados no espírito do Great Expectations: um bug a montante
    que mande cobrança negativa deve ABORTAR, não treinar um modelo errado."""
    assert (bruto["Tenure Months"] >= 0).all()
    assert (bruto["Monthly Charges"] > 0).all()
    assert (bruto["Total Charges"].dropna() >= 0).all()


# --- Split -----------------------------------------------------------------

def test_split_e_deterministico():
    """Sem isso, comparar dois runs mede a sorte do sorteio, não o modelo."""
    a, b = data.dividir(), data.dividir()
    assert list(a.treino.X.index) == list(b.treino.X.index)
    assert a.sha256 == b.sha256


def test_split_e_estratificado(dados):
    """Taxas diferentes entre partições invalidariam a comparação entre modelos."""
    taxas = [getattr(dados, n).taxa_churn for n in ("treino", "validacao", "teste")]
    assert max(taxas) - min(taxas) < 0.01


def test_proporcao_60_20_20(dados):
    total = sum(len(getattr(dados, n)) for n in ("treino", "validacao", "teste"))
    assert len(dados.teste) / total == pytest.approx(0.20, abs=0.01)
    assert len(dados.validacao) / total == pytest.approx(0.20, abs=0.01)


# --- Pré-processamento -----------------------------------------------------

def test_total_charges_vazio_vira_zero_e_nao_mediana(bruto):
    """O vazio é medição verdadeira (cliente sem ciclo de faturamento). Imputar a
    mediana inventaria ~R$ 1.400 de histórico para quem nunca foi faturado — e de
    forma plausível o bastante para ninguém notar."""
    vazios = bruto[bruto["Total Charges"].isna()]
    assert len(vazios) == 11
    assert (vazios["Tenure Months"] == 0).all()

    pre = construir_preprocessador(escalonar=False).fit(bruto[config.FEATURES])
    saida = pd.DataFrame(pre.transform(vazios[config.FEATURES]),
                         columns=pre.get_feature_names_out())
    assert (saida["Total Charges"] == 0).all()


def test_categoria_desconhecida_nao_colide_com_categoria_existente(bruto):
    """Com drop='if_binary', um Gender='Other' viraria o MESMO vetor de 'Female'
    e a API responderia 200 OK com predição errada. Sem drop, vira vetor de zeros,
    que não colide com nenhuma categoria conhecida."""
    pre = construir_preprocessador(escalonar=False).fit(bruto[config.FEATURES])
    linha = bruto[config.FEATURES].head(1).copy()
    linha["Gender"] = "Nao-binario"

    cols = list(pre.get_feature_names_out())
    dummies_gender = [i for i, c in enumerate(cols) if c.startswith("Gender")]
    saida = pre.transform(linha)[0][dummies_gender]
    assert saida.sum() == 0, "categoria inédita deveria ser vetor de zeros"


def test_pipeline_nao_produz_nan_nem_inf(bruto):
    """Invariante geral que protege a Etapa 4: qualquer feature derivada com
    'Tenure Months' no denominador produzirá inf nas 11 linhas com tenure = 0.
    O dataset TEM essas linhas — este é um teste que pega erro de verdade."""
    pre = construir_preprocessador().fit(bruto[config.FEATURES])
    saida = pre.transform(bruto[config.FEATURES])
    assert np.isfinite(saida).all()


def test_arquivo_bruto_esta_ordenado_pelo_alvo(bruto):
    """⚠️ O arquivo vem ORDENADO: as 1.869 primeiras linhas são todos os churners.

    Descoberto por um teste que quebrou. Três consequências que este teste fixa
    como conhecimento em vez de deixar como surpresa:
      1. um split com shuffle=False daria treino 100% churner e teste 100% não;
      2. qualquer amostragem por head()/tail() traz uma classe só — inclusive o
         'subset minúsculo' que a Etapa 9.5 recomenda para o CI leve, que
         portanto precisa ser ESTRATIFICADO;
      3. explorar a base com df.head() dá leitura completamente enviesada.
    """
    y = bruto[config.TARGET]
    assert y.head(1869).mean() == 1.0
    assert y.iloc[1869:].mean() == 0.0


def test_pipeline_treina_e_prediz(dados):
    """Sanity test: fit num subset pequeno e predict devolvem shape correto.

    A amostra vem do TREINO já particionado, não de bruto.head(200) — que traria
    só churners (ver teste acima). É a forma correta de montar o subset do CI leve.
    """
    X = dados.treino.X.head(200)
    y = dados.treino.y.head(200)
    assert y.nunique() == 2, "o subset precisa ter as duas classes"

    pipe = construir_pipeline(LogisticRegression(max_iter=1000, random_state=config.SEED))
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (200,)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.isfinite(p).all()


def test_escalonamento_ajustado_so_no_treino(dados):
    """A média do StandardScaler tem de vir do TREINO. Se fosse ajustada sobre
    tudo, a validação teria média exatamente zero — o sinal de leakage."""
    pipe = construir_pipeline(LogisticRegression(max_iter=1000, random_state=config.SEED))
    pipe.fit(dados.treino.X, dados.treino.y)
    pre = pipe.named_steps["preproc"]

    num = list(pre.get_feature_names_out()).index("Monthly Charges")
    media_val = pre.transform(dados.validacao.X)[:, num].mean()
    assert media_val != 0.0
    assert abs(media_val) < 0.2, "distribuição da validação não deveria destoar tanto"

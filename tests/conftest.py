"""
Fixtures compartilhadas.

`data.dividir()` lê o Excel (~0,6 s) e treinar o campeão custa ~13 ms: o caro é
a leitura, não o treino. Por isso as fixtures têm escopo de **sessão** — a suíte
inteira trabalha sobre a mesma partição, que é também o que garante que os
números batam com o gate e com o teste de caracterização.

⚠️ O artefato usado nos testes é gravado em `tmp_path`, nunca em `models/`. Um
teste que sobrescreve o artefato promovido é um teste que muda o que a API serve.
"""

from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from src import artefato, data, gate, promover


@pytest.fixture(scope="session")
def dados() -> data.Dados:
    return data.dividir()


@pytest.fixture(scope="session")
def promovido(dados, tmp_path_factory) -> tuple[Pipeline, object]:
    """Treina o campeão e o serializa — o par (objeto em memória, caminho).

    Os dois lados são necessários: o round-trip compara memória contra disco, e
    comparar o disco com ele mesmo não prova nada.

    🔑 **Os metadados saem de `promover.montar_metadados`, não de um dict escrito
    aqui.** Eram duas construções paralelas do mesmo objeto, e o custo apareceu
    na Etapa 10a-2: `carregar(estrito=True)` passou a exigir a `referencia`, e a
    fixture — que montava o dict à mão — teria derrubado a suíte inteira por
    estar desatualizada em relação ao que a promoção real grava. Um teste que
    reconstrói o objeto sob teste testa a reconstrução.

    É o mesmo movimento do item 85, quando `gate.medir()` foi extraída para que
    o CI e o teste de caracterização medissem O MESMO objeto em vez de dois
    irmãos: **o teste puxa o desenho**.
    """
    pipe = gate.treinar_campeao(dados)
    metricas, custo = gate.medir(dados, pipe=pipe)
    caminho = tmp_path_factory.mktemp("modelo") / "campeao.joblib"
    artefato.salvar(
        pipe,
        promover.montar_metadados(
            dados, metricas, custo, features=list(pipe.feature_names_in_),
        ),
        caminho=caminho,
    )
    return pipe, caminho


@pytest.fixture(scope="session")
def art(promovido) -> artefato.Artefato:
    """O artefato carregado do disco — é sobre ELE que a API é montada."""
    _, caminho = promovido
    return artefato.carregar(caminho)

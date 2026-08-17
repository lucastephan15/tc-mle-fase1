"""
Testes do artefato promovido — Etapa 9c / 9d-sexies.

A suíte que existia até aqui tem 54 testes e **nenhum toca o objeto
serializado** — ela exercita o pipeline em memória, que é justamente o que a API
NÃO vai usar. Tudo que separa os dois (serialização, carimbo de versão,
disponibilidade do código-fonte no ambiente de destino) ficava sem cobertura, e
a primeira execução real aconteceria dentro do container.

Testar o que você treina não é testar o que você serve.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from src import artefato, config

# As fixtures `dados`, `promovido` e `art` vivem em `tests/conftest.py`: elas são
# compartilhadas com os testes da API, e duplicá-las faria a suíte ler o Excel
# duas vezes para produzir o mesmo objeto.


def test_round_trip_e_bit_a_bit(dados, promovido):
    """Salvar → carregar → as predições são IDÊNTICAS, não "próximas".

    `np.array_equal`, não `approx`: serialização não é aproximação numérica. Se
    aparecer diferença na décima casa, alguma coisa foi reconstruída em vez de
    restaurada — e a hora de descobrir isso é aqui, não quando o número
    reportado na documentação não bater com o que a API devolve.
    """
    pipe, caminho = promovido
    carregado = artefato.carregar(caminho)

    p_memoria = pipe.predict_proba(dados.validacao.X)[:, 1]
    p_disco = carregado.pipeline.predict_proba(dados.validacao.X)[:, 1]

    assert np.array_equal(p_memoria, p_disco), (
        f"maior diferença: {np.abs(p_memoria - p_disco).max():.3e} em "
        f"{len(p_disco)} predições"
    )


def test_metadados_e_pipeline_nao_podem_divergir(promovido):
    """O contrato declarado é o contrato que o pipeline exige.

    Duas listas mantidas iguais pela memória de quem escreveu divergem em
    silêncio, e esta divergência específica não apareceria como 422 na
    validação da API — apareceria como 500 no ColumnTransformer, com dado real.
    """
    _, caminho = promovido
    a = artefato.carregar(caminho)

    assert a.features == list(a.pipeline.feature_names_in_)
    assert a.features == config.FEATURES, (
        "o artefato divergiu de config.FEATURES — o que pode ser legítimo "
        "(modelo promovido antes de uma mudança no config), mas nunca "
        "silencioso: quem manda na API é o artefato."
    )
    assert 0.0 < a.limiar < 1.0
    assert a.versao == config.VERSAO_MODELO
    assert len(a.sha256) == 64


def test_carga_recusa_ambiente_com_outra_versao_de_sklearn(promovido, tmp_path):
    """Versão divergente TEM de barrar a carga — o sklearn não barra.

    Medido: forjando `_sklearn_version`, a carga emite 21
    `InconsistentVersionWarning` (subclasse de UserWarning ⇒ não interrompe),
    prediz mesmo assim com número diferente, e o `__setstate__` ainda apaga o
    carimbo do objeto. Como o aviso não para nada e o carimbo some, a checagem
    tem de ser explícita e na carga.

    Verificado REPROVANDO, que é o único jeito de saber que o controle existe.
    """
    import joblib

    _, caminho = promovido
    payload = joblib.load(caminho)
    payload["metadados"]["versoes"]["scikit-learn"] = "0.0.0-inexistente"
    forjado = tmp_path / "forjado.joblib"
    joblib.dump(payload, forjado)

    with pytest.raises(artefato.ArtefatoIncompativel, match="scikit-learn"):
        artefato.carregar(forjado)

    # e o modo de inspeção continua podendo abrir o artefato antigo
    assert artefato.carregar(forjado, estrito=False).versao == config.VERSAO_MODELO


def test_carga_recusa_formato_desconhecido(tmp_path):
    """Pipeline nu (formato anterior) não passa por artefato válido."""
    import joblib

    caminho = tmp_path / "nu.joblib"
    joblib.dump({"qualquer": "coisa"}, caminho)
    with pytest.raises(artefato.ArtefatoIncompativel, match="formato"):
        artefato.carregar(caminho)


def test_carrega_sem_o_repo_no_sys_path(promovido):
    """O ambiente do container: `src` não é importável, e o artefato tem de abrir.

    Não é hipótese. Dos 9 artefatos do `mlruns/`, **8 carregam sem `src` no
    path e 1 não** — o `logreg+feat`, por causa do `FunctionTransformer`, que
    guarda uma referência à NOSSA função. O campeão está do lado seguro, e este
    teste é o que impede que ele saia dele sem ninguém notar (ligar as features
    da Etapa 4 no campeão passaria em todo o resto da suíte).

    Roda num subprocesso com a raiz do repo removida do `sys.path` — dentro do
    mesmo processo o módulo já está importado e o teste não provaria nada.
    """
    _, caminho = promovido
    codigo = textwrap.dedent(f"""
        import sys
        raiz = {str(config.RAIZ)!r}
        sys.path[:] = [p for p in sys.path if p not in ("", ".", raiz)]
        import joblib
        payload = joblib.load({str(caminho)!r})
        pipe = payload["pipeline"]
        assert "src" not in sys.modules, "o repo entrou no path por outra via"
        print(len(pipe.feature_names_in_))
    """)
    r = subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True, text=True, cwd="/",
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(len(config.FEATURES))

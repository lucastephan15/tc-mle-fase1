"""
O serviço de pontuação — Etapa 9d.

A camada que sabe pontuar, e **nada sobre HTTP**. Testável sem subir servidor,
que é o critério a usar quando a separação parecer cerimônia: *isto fica
testável sem `TestClient`?*

Duas decisões medidas moram aqui:

1. 🚨 **`pd.DataFrame(linhas)`, nunca `np.array([[...]])`.** O artefato é um
   `Pipeline(ColumnTransformer, modelo)` e o `ColumnTransformer` **seleciona
   colunas por nome**: com ndarray levanta `ValueError`, com DataFrame funciona
   — e funciona **com as colunas fora de ordem, dando probabilidade idêntica ao
   dígito**. O padrão `np.array([[f.x1, f.x2, f.x3]])` fixa a ordem no corpo da
   função, onde nada a verifica: o Pydantic valida os campos do JSON, não a
   sequência em que alguém os desempacota. Reordenar dois atributos do schema —
   refatoração que nenhum linter, teste ou revisor barra — trocaria duas
   features de lugar e a API continuaria respondendo **200 OK com número
   errado**. *Nome de coluna é contrato verificável; posição é convenção não
   verificada.*

2. 🚨 **`predict_proba(...)[:, 1]` + limiar lido do artefato, nunca `.predict()`.**
   `.predict()` aplica 0,5 implícito e custa R$ 7.546 por ciclo (+23,8%) e 83
   churners, medido na validação.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src import artefato as art_mod


class Pontuador(Protocol):
    """O que uma rota precisa saber sobre "algo que pontua".

    Existe para a inversão de dependência: o handler depende desta abstração, e
    quem escolhe a implementação é quem monta o app. Sem isso, testar a lógica
    da rota (limiar, formato da resposta, contrato de erro) exigiria o artefato
    real no disco — e "o teste do `/predict` precisa do `.joblib`" é como um
    teste deixa de ser escrito.
    """

    @property
    def versao(self) -> str: ...

    @property
    def limiar(self) -> float: ...

    @property
    def features(self) -> list[str]: ...

    @property
    def sha256(self) -> str: ...

    @property
    def versoes(self) -> dict[str, str]: ...

    def pontuar(self, linhas: list[dict]) -> list[float]: ...

    def pronto(self) -> bool: ...


class PontuadorArtefato:
    """A implementação real: pontua com o artefato promovido."""

    def __init__(self, artefato: art_mod.Artefato) -> None:
        self._a = artefato

    @property
    def versao(self) -> str:
        return self._a.versao

    @property
    def limiar(self) -> float:
        return self._a.limiar

    @property
    def features(self) -> list[str]:
        return self._a.features

    @property
    def sha256(self) -> str:
        return self._a.sha256

    @property
    def versoes(self) -> dict[str, str]:
        return dict(self._a.metadados.get("versoes", {}))

    def pontuar(self, linhas: list[dict]) -> list[float]:
        """Um DataFrame, uma chamada, uma coluna de probabilidades.

        UMA chamada para o lote inteiro, não uma por linha: o custo do sklearn é
        quase todo fixo por chamada (~1,67 ms contra 2,0 µs por linha marginal),
        então iterar aqui devolveria as 825× ao consumidor.
        """
        X = pd.DataFrame(linhas)
        return [float(p) for p in self._a.pipeline.predict_proba(X)[:, 1]]

    def pronto(self) -> bool:
        """Prontidão: **pode receber tráfego?**, não *o processo está vivo?*.

        Lida do objeto que serve — nunca de uma variável global setada no
        import, que continua dizendo `healthy` depois de a carga do modelo mudar
        de lugar (ou de o artefato sumir do disco). Verifica o que de fato
        quebraria a predição: o pipeline sabe pontuar, e o contrato de colunas
        que o artefato declara é o que o pipeline espera.
        """
        pipe = self._a.pipeline
        return (
            hasattr(pipe, "predict_proba")
            and list(getattr(pipe, "feature_names_in_", [])) == self.features
        )


def carregar_pontuador() -> PontuadorArtefato:
    """Carrega o artefato promovido e o embrulha. Levanta se não bater.

    Chamado **uma vez**, na construção do app — nunca por requisição, e nunca
    preguiçosamente com cache. Medido em processo frio: com carga preguiçosa a
    primeira requisição custa **727,9 ms** contra 1,9 ms da segunda; carregando
    antes de aceitar tráfego, a primeira já sai a 1,9 ms — **383×**. O custo
    total é o mesmo (~715 ms, 93% import de biblioteca); o que muda é **quem
    paga**: o deploy ou o primeiro cliente. E `--workers 4` são quatro desses.
    """
    return PontuadorArtefato(art_mod.carregar())

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

import numpy as np
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
    def versoes_treino(self) -> dict[str, str]: ...

    @property
    def versoes_runtime(self) -> dict[str, str]: ...

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
    def versoes_treino(self) -> dict[str, str]:
        """O ambiente que TREINOU o modelo — carimbo gravado na promoção."""
        return dict(self._a.metadados.get("versoes", {}))

    @property
    def versoes_runtime(self) -> dict[str, str]:
        """O ambiente que está SERVINDO agora — medido, não lido do artefato.

        🚨 Separado do anterior por um defeito que só a imagem revelou. O
        `/health` declarava um único campo `versoes`, alimentado pelos metadados
        do artefato, e qualquer leitor entende um campo de prontidão como "o que
        este serviço tem". Medido no container em 18/08/2026: o campo dizia
        **Python 3.12.5** (a máquina que treinou) enquanto o processo rodava
        **3.12.14** (a imagem `python:3.12-slim`). Nada estava errado no
        artefato; o *rótulo* é que respondia a outra pergunta.

        É a família de erro do repo inteiro — uma afirmação sobre um sistema que
        ninguém confronta com o sistema — e no `/health` ela é a pior variante,
        porque este endpoint existe justamente para declarar identidade. Um campo
        ambíguo é pior que um campo ausente: parece resposta.

        ⚠️ A checagem de `artefato.carregar()` compara **só o scikit-learn**
        (onde a serialização mora, e ali diverge = processo morre no boot). As
        demais linhas divergirem é normal e não impede nada — o que não pode é
        ficar invisível.
        """
        return art_mod.versoes_do_ambiente()

    def pontuar(self, linhas: list[dict]) -> list[float]:
        """Um DataFrame, uma chamada, uma coluna de probabilidades.

        UMA chamada para o lote inteiro, não uma por linha: o custo do sklearn é
        quase todo fixo por chamada (~1,67 ms contra 2,0 µs por linha marginal),
        então iterar aqui devolveria as 825× ao consumidor.

        🚨 **`None` vira `NaN` antes do DataFrame, e a troca não é cosmética.**
        O schema aceita `null` onde o vazio é medição verdadeira (`Total
        Charges` de quem não teve ciclo de faturamento). Medido contra o campeão:
        com `np.nan` o pipeline imputa e devolve **0,2449336585**; com `None`
        puro a coluna vira `dtype=object`, o imputador não a reconhece como
        ausente e o `LogisticRegression` levanta **`ValueError: Input X contains
        NaN`** — ou seja, aceitar `null` no schema **sem** esta linha trocaria um
        422 honesto por um **500**. As duas peças são uma correção só.
        """
        X = pd.DataFrame(
            [{k: (np.nan if v is None else v) for k, v in linha.items()}
             for linha in linhas]
        )
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

"""
O artefato servido — Etapa 9c · empacotamento e identidade.

Este módulo responde a UMA pergunta, e ela não é "como salvo o modelo?":

    o que está carregado agora é o mesmo objeto que foi avaliado?

Em 2012 a Knight Capital implantou código novo em sete de oito servidores; o
oitavo seguiu rodando o legado e custou US$ 440 milhões em 45 minutos. A falha
não foi um bug — foi versão inconsistente que ninguém verificou. A versão
doméstica disso é o `.joblib` do treino de ontem no container de hoje: responde
200, com a predição errada, e o gate do CI não vê nada, porque ele retreina em
vez de conferir o artefato.

Três decisões de desenho, cada uma contra um modo de falha silencioso:

1. **UM arquivo.** Pipeline e metadados são serializados juntos. Modelo e
   metadados em arquivos separados, sem verificação de que combinam, produzem
   rótulo errado com HTTP 200.

2. **Um `dict` puro no disco, nunca uma classe deste módulo.** Se o objeto
   serializado fosse uma dataclass daqui, carregá-lo exigiria `src.artefato`
   importável no ambiente de destino — o mesmo acoplamento ao código-fonte que
   o `FunctionTransformer` já impõe ao pipeline com FE (medido: dos 9 artefatos
   do `mlruns/`, o único que não carrega sem `src` no path é o `logreg+feat`).
   O campeão não usa FE; serializar um `dict` mantém essa propriedade em vez de
   destruí-la por comodidade de tipagem.

3. **A versão da biblioteca é gravada à mão.** `BaseEstimator.__setstate__` faz
   `state.pop("_sklearn_version")` ao desserializar: o objeto carregado **não
   sabe mais** com que versão foi treinado. E versão divergente não impede a
   carga — emite `InconsistentVersionWarning`, que é subclasse de `UserWarning`,
   prediz mesmo assim (medido: 0,6601 contra 0,6646 esperado) e ninguém lê.
   Se o carimbo não for escrito no ato de promover, ele não existe depois.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src import config

# Chave do dicionário serializado. Mudar isto quebra artefatos já promovidos —
# é o formato do arquivo, não um detalhe interno.
_CHAVE_PIPELINE = "pipeline"
_CHAVE_METADADOS = "metadados"


class ArtefatoIncompativel(RuntimeError):
    """O artefato do disco não é o que este ambiente/código espera servir.

    Levantada na CARGA, de propósito. Subir degradado — servir mesmo assim,
    avisando no log — é escolher a falha silenciosa: a API responderia 200 com
    probabilidade de outro modelo. Falhar na inicialização transforma um erro
    invisível de predição num erro visível de deploy.
    """


@dataclass(frozen=True)
class Artefato:
    """O que a API tem em mãos depois de carregar. Vive só em memória."""

    pipeline: Pipeline
    metadados: dict[str, Any]
    caminho: Path
    sha256: str  # do ARQUIVO — a identidade que o /health declara

    @property
    def versao(self) -> str:
        return self.metadados["versao_modelo"]

    @property
    def features(self) -> list[str]:
        """Os nomes das colunas, na ordem, como o pipeline os espera."""
        return list(self.metadados["features"])

    @property
    def limiar(self) -> float:
        """Limiar de operação derivado da economia do erro, na promoção.

        Vem do artefato e não de uma constante no código da API porque é
        propriedade DO MODELO SERVIDO: o corte ótimo muda quando a distribuição
        de probabilidades muda, e ela muda quando o modelo muda. Um `0.29`
        literal na API é a segunda cópia que ninguém atualiza junto.
        """
        return float(self.metadados["limiar_operacao"])

    @property
    def referencia(self) -> dict[str, Any]:
        """Distribuição do treino congelada na promoção — Etapa 10a-2.

        Mesma justificativa do `limiar`, um passo adiante: o baseline contra o
        qual se mede drift é propriedade DAQUELE modelo. Um `referencia.json`
        ao lado do `.joblib` seria o par "dois arquivos que podem não combinar"
        que a decisão nº 1 deste módulo existe para proibir — e nesta variante a
        falha é pior que rótulo errado, porque não falha: baseline de um modelo
        contra predições de outro mede drift fantasma, ou deixa de ver o real.
        """
        return dict(self.metadados["referencia"])


def sha256_arquivo(caminho: Path) -> str:
    """Identidade verificável do artefato — o equivalente barato de um run_id."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(8192), b""):
            h.update(bloco)
    return h.hexdigest()


def versoes_do_ambiente() -> dict[str, str]:
    """O terceiro artefato de Sculley: código, dados e AMBIENTE.

    O repo já loga `commit_hash()` e `sha256_dataset()` no MLflow. Sem isto, a
    pergunta "alguém reproduz o seu número?" fica sem a metade que mais quebra
    — é a versão de biblioteca, não o código, que muda o dígito.
    """
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def salvar(pipeline: Pipeline, metadados: dict[str, Any],
           caminho: Path = config.ARTEFATO) -> str:
    """Serializa pipeline + metadados num único arquivo. Devolve o sha256 dele.

    Nada de `models/` no Git (o `.gitignore` já diz por quê: o registry é a
    fonte de verdade, não o repositório). O artefato é reproduzido por
    `make promover`, e a identidade que se comunica é o sha256, não o arquivo.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        _CHAVE_PIPELINE: pipeline,
        _CHAVE_METADADOS: {**metadados, "versoes": versoes_do_ambiente()},
    }
    joblib.dump(payload, caminho)
    return sha256_arquivo(caminho)


def carregar(caminho: Path = config.ARTEFATO, estrito: bool = True) -> Artefato:
    """Carrega e VERIFICA. As quatro checagens são o conteúdo da função.

    `estrito=False` existe para inspeção manual de artefato antigo (arqueologia
    de um número já reportado). Nunca para a API: quem serve verifica.
    """
    if not caminho.exists():
        raise ArtefatoIncompativel(
            f"artefato ausente em {caminho}. Rode `make promover` — a API não "
            f"treina no startup de propósito (o dado bruto é LGPD e não entra "
            f"na imagem; e o serviço tem de servir O MESMO modelo avaliado)."
        )

    payload = joblib.load(caminho)

    # (1) formato — um artefato de outra época pode ser o Pipeline nu.
    if not isinstance(payload, dict) or _CHAVE_PIPELINE not in payload:
        raise ArtefatoIncompativel(
            f"{caminho} não tem o formato esperado (dict com "
            f"'{_CHAVE_PIPELINE}' e '{_CHAVE_METADADOS}'). Artefato de versão "
            f"anterior do empacotamento? Repromova."
        )
    pipeline = payload[_CHAVE_PIPELINE]
    metadados = payload.get(_CHAVE_METADADOS, {})

    # (2) ambiente — o carimbo que o __setstate__ apagou do objeto.
    #     Comparação exata contra o pin: o requirements.txt trava a versão, então
    #     qualquer diferença significa que este ambiente NÃO é o declarado, e
    #     "não quebra" não é o mesmo que "reproduz".
    gravada = metadados.get("versoes", {}).get("scikit-learn")
    atual = sklearn.__version__
    if estrito and gravada is not None and gravada != atual:
        raise ArtefatoIncompativel(
            f"artefato treinado com scikit-learn {gravada}, ambiente tem "
            f"{atual}. O sklearn CARREGA assim mesmo (InconsistentVersionWarning "
            f"é UserWarning e não interrompe) e prediz números diferentes — por "
            f"isso a checagem é aqui e não no log."
        )

    # (3) contrato de colunas — o que o schema da API vai ter de honrar.
    esperadas = list(metadados.get("features", []))
    reais = list(getattr(pipeline, "feature_names_in_", []))
    if estrito and esperadas and reais and esperadas != reais:
        raise ArtefatoIncompativel(
            f"os metadados declaram {len(esperadas)} features e o pipeline "
            f"espera {len(reais)}, ou a ordem difere. Duas listas mantidas "
            f"iguais pela memória de quem escreveu divergem em silêncio — e "
            f"esta divergência apareceria como 500 no ColumnTransformer, não "
            f"como 422 na validação."
        )

    # (4) baseline de drift — Etapa 10a-2. Um artefato sem referência CARREGA
    #     sem erro (todos os acessos a metadados usam `.get`), sobe, responde
    #     200 e só falha três semanas depois, na análise: o PSI seria calculado
    #     contra um dict vazio, ou não seria calculado e ninguém notaria a
    #     ausência de alerta. Ausência de alarme é indistinguível de ausência de
    #     problema — por isso a exigência é aqui, na carga, e não lá.
    #
    #     A cobertura é comparada com as features DECLARADAS pelo mesmo motivo da
    #     checagem (3): duas listas mantidas iguais pela memória de quem escreveu
    #     divergem em silêncio. Baseline de 13 colunas para um modelo de 12 mede
    #     drift de uma coluna que o modelo não usa mais.
    if estrito:
        ref = metadados.get("referencia") or {}
        cobertas = set(ref.get("numericas", {})) | set(ref.get("categoricas", {}))
        if not cobertas:
            raise ArtefatoIncompativel(
                f"{caminho} não tem as estatísticas de referência do treino "
                f"(chave 'referencia'). Artefato promovido antes da Etapa 10a-2? "
                f"Rode `make promover` — sem baseline não existe drift para "
                f"detectar, só um gráfico comparando nada."
            )
        if not ref.get("scores"):
            raise ArtefatoIncompativel(
                f"{caminho} tem baseline de features e NÃO tem baseline de "
                f"scores ('referencia.scores'). É a metade que mais importa: as "
                f"features só entram no log com TC_LOG_FEATURES=1, as "
                f"probabilidades vão em toda linha — sem este bloco, a única "
                f"vigilância que roda sempre em produção é a que fica sem "
                f"denominador. Rode `make promover`."
            )
        if esperadas and cobertas != set(esperadas):
            faltam = sorted(set(esperadas) - cobertas)
            sobram = sorted(cobertas - set(esperadas))
            raise ArtefatoIncompativel(
                f"a referência de drift não cobre as features servidas — "
                f"faltam {faltam}, sobram {sobram}. O baseline e o contrato de "
                f"colunas saíram de promoções diferentes."
            )

    return Artefato(
        pipeline=pipeline,
        metadados=metadados,
        caminho=caminho,
        sha256=sha256_arquivo(caminho),
    )


def descrever(a: Artefato) -> str:
    """Resumo de uma tela — o que o /health vai declarar, em texto."""
    m = a.metadados
    v = m.get("versoes", {})
    linhas = [
        f"artefato        : {a.caminho}",
        f"sha256          : {a.sha256[:16]}…",
        f"versao_modelo   : {m.get('versao_modelo')}",
        f"modelo          : {m.get('modelo')}",
        f"features        : {len(a.features)}",
        f"limiar_operacao : {m.get('limiar_operacao')}",
        f"referencia      : {len(m.get('referencia', {}).get('numericas', {}))} "
        f"numéricas + {len(m.get('referencia', {}).get('categoricas', {}))} "
        f"categóricas · n={m.get('referencia', {}).get('n')} "
        f"({m.get('referencia', {}).get('particao')})",
        f"scores (ref)    : n={m.get('referencia', {}).get('scores', {}).get('n')} "
        f"da validação · média "
        f"{m.get('referencia', {}).get('scores', {}).get('media')} · "
        f"{m.get('referencia', {}).get('scores', {}).get('taxa_acima_do_limiar')} "
        f"acima do limiar",
        f"promovido_em    : {m.get('promovido_em')}",
        f"commit          : {m.get('commit')}",
        f"dataset_sha256  : {str(m.get('dataset_sha256'))[:16]}…",
        f"ambiente        : python {v.get('python')} · scikit-learn "
        f"{v.get('scikit-learn')} · numpy {v.get('numpy')} · pandas {v.get('pandas')}",
    ]
    return "\n".join(linhas)


if __name__ == "__main__":
    # Inspeção: `python -m src.artefato` mostra o que está promovido hoje.
    try:
        print(descrever(carregar()))
    except ArtefatoIncompativel as e:
        print(f"❌ {e}")
        sys.exit(1)

# ML Canvas — Predição de churn em telecom

**Tech Challenge Fase 1 · Pós-graduação em Machine Learning Engineering (FIAP + Alura PosTech)**
Autor: Luca Stephan · Modalidade individual · v1.0 · 25/08/2026

> **O que é este documento.** O ML Canvas (Louis Dorard) força as decisões de **negócio** de um
> sistema de ML a existirem antes das de modelagem. Ele foi preenchido na **Etapa 0**, antes da
> primeira linha de código, e o registro cru está na §0 do
> [`decision-log.md`](decision-log.md). Esta é a versão de leitura — **os números vieram da
> execução**, então cada bloco fecha com o que de fato aconteceu, e não com o que se pretendia.
>
> 🔑 A fase de *Business Understanding* do CRISP-DM responde por **38%** das falhas em projetos de
> dados — mais que Data Understanding (24%). A maior parte dos projetos morre aqui, não na técnica.

---

## 🎯 Proposta de valor

A equipe de Retenção de uma operadora de telecom não consegue abordar toda a base. **O sistema
ordena os clientes por risco de cancelamento numa fila de trabalho**, para que o esforço humano —
que é o recurso escasso — seja gasto onde tem maior chance de evitar a perda.

🚨 **O artefato é um RANQUEADOR, não um classificador.** A pergunta do negócio não é *"esse cliente
vai cancelar: sim ou não?"* — é ***"quem eu ligo primeiro?"***. Essa distinção determina tudo o que
vem depois: a métrica é independente de limiar (PR-AUC), o relatório principal é uma curva de
ganho, a API devolve **probabilidade**, e o corte da fila é **parâmetro de negócio**, alterável sem
retreinar nada.

**Resultado medido (conjunto de teste, tocado uma única vez):** contatando **10% da base**, a
campanha alcança **28,6%** de quem ia cancelar — **75,9% do máximo matematicamente possível**.

---

## 👥 Stakeholders

| Papel | Quem é | O que espera | O que decide |
|---|---|---|---|
| **Quem AGE sobre a previsão** | agente de Retenção | uma fila priorizada no CRM, com contexto suficiente para escolher a abordagem | como abordar cada cliente (ligação, desconto, upgrade) |
| **Quem paga a conta** | gerência de Retenção | redução do churn dentro do orçamento de campanha | **o `k` da fila** — quantos clientes por ciclo |
| **Dono dos dados** | engenharia de dados / faturamento | que o consumo não crie dependência frágil | disponibilidade e formato das 13 features |
| **Governança** | jurídico / DPO | base legal, LGPD Art. 20, não discriminação | se o uso é permitido, e sob que condições |
| **Quem constrói e opera** | o autor (papéis acumulados) | pipeline reprodutível e reversível | modelo, limiar técnico, política de retreino |

⚠️ **RACI declarado, não fingido.** Este é um projeto **individual**: quem treina, quem aprova a
promoção e quem monitora são a mesma pessoa. Num contexto real seriam três papéis separados — e
duas decisões deste projeto **exigiriam dono que ele não tem**: aceitar a disparidade de 58,89 pp
em `Dependents` (§5g) e adotar limiar por grupo, que é tratamento explicitamente diferente por
atributo protegido. Registrar quem *deveria* decidir é mais honesto que decidir sozinho e não dizer.

---

## 🧠 Tarefa de ML

| Campo | Definição | Por quê |
|---|---|---|
| **Paradigma** | supervisionado | o rótulo já existe no histórico e a pergunta é `P(cancelar \| perfil)` ⇒ o objeto a estimar é `P(y\|X)`. O que separa os paradigmas é o **tipo de sinal disponível**, não o algoritmo |
| **Tipo** | classificação binária, saída em **probabilidade** | a fila precisa do score; `0/1` esconde a diferença entre 0,51 e 0,97 |
| **Alvo** | `Churn = Yes` — o cliente cancelou o contrato | é decisão **de negócio**, não técnica. "90 dias sem uso" produziria outro rótulo por linha e outro projeto. ⚠️ Definição **herdada do dataset**: não sabemos se inclui suspensão, inadimplência ou downgrade total — num projeto real, esta seria a primeira pergunta ao negócio |
| **Janela de predição** | 30 dias | é o horizonte em que a campanha ainda é acionável |
| **Prevalência** | **26,54%** | desbalanceado, **não raro** — muda a métrica, não exige reamostragem |

---

## 💰 A economia do erro — o bloco que define a métrica

| Erro | O que é na prática | Custo unitário |
|---|---|---|
| **Falso negativo** | o cliente ia cancelar e **não entrou na fila** | **R$ 194** |
| **Falso positivo** | gastamos atenção com quem ficaria de qualquer jeito | **R$ 62** |

**Assimetria ≈ 3:1.** 🔑 O custo do FN é **valor esperado** (× taxa de conversão da campanha), não
valor nominal: um churner perdido só custa quando a abordagem *teria* funcionado. Usar o valor
cheio inflaria a assimetria e deslocaria o limiar para baixo sem justificativa.

**As duas estratégias triviais, medidas, para que o valor do modelo tenha referência:**

| Estratégia | Custo por ciclo |
|---|---|
| Não abordar ninguém | R$ 72.556 |
| Abordar a base inteira | R$ 64.170 |
| **O modelo, no limiar 0,29** | **R$ 32.882** (−54,7%) |

🔑 A segunda linha é a que mata o argumento de "otimizar recall puro": ela mostra que a solução
degenerada **também é cara**, em vez de apenas afirmar que seria.

---

## 📏 Métricas — três níveis, três perguntas

| Nível | Métrica | Pergunta que responde | Quem consome | Piso |
|---|---|---|---|---|
| 1 · **seleção** | **PR-AUC** (average precision) | qual modelo ordena melhor *em geral*? | comparação de candidatos | **0,2654** (prevalência) |
| 2 · **operação** | **recall@10%** e **recall@20%** | quanto a campanha de verdade captura? | negócio | **0,10** / **0,20** (teto estrutural `k/prevalência`) |
| 3 · **decisão** | **custo em R$** → limiar | onde eu corto a fila? | limiar, por modelo | as duas estratégias triviais acima |

🚨 **Nenhuma métrica é reportada sem o piso ao lado.** *Métrica sem piso não é medida, é opinião* —
o chute na majoritária já entrega **73,46%** de acurácia neste problema.

⚠️ **Por que o nível 2 não é redundante:** trocar ROC-AUC por PR-AUC corrige o desbalanceamento, e
só. PR-AUC continua sendo uma **integral sobre todos os limiares**, inclusive sobre a faixa onde a
operação nunca vai rodar. Quem responde à pergunta operacional é `recall@k` — e é `k` que a
operação escolhe, não o modelo.

**Cadeia KPI:** `recall@10% = 28,6%` → 1 em cada 3,5 clientes contatados ia mesmo cancelar →
**−54,7% no custo do erro por ciclo** contra não fazer nada.

---

## 🗄️ Fontes de dados

| Item | Valor |
|---|---|
| **Origem** | Telco Customer Churn (IBM), 7.043 clientes, 33 colunas · `sha256 1bcbc0cc…` |
| **Natureza** | snapshot **estático**, sem eixo temporal |
| **Partição** | 60/20/20 estratificado — treino ajusta, validação seleciona, **teste tocado uma vez** |
| **Features usadas** | **13**, de 19 candidatas — a redução foi por **custo operacional e superfície de drift**, não por ganho de métrica (que foi nulo: 0,6883 × 0,6904) |

**Excluídas, e por quê:**

| Coluna | Motivo |
|---|---|
| `Churn Reason`, `Churn Label` | 🚨 **leakage perfeito** — só existem depois do cancelamento. `notna()` acerta 100% |
| `Churn Score` | 🚨 é o **modelo da IBM** embutido no dataset. Usá-la seria prever a IBM, não churn ⇒ virou **benchmark** (PR-AUC 0,8824, o teto de referência) |
| `CustomerID` | identificador + dado pessoal (LGPD) |
| `Latitude`, `Longitude`, `Zip Code`, `City` | proxies geográficos de renda e raça — escapariam da cardinalidade, não da objeção ética |
| `Count`, `Country`, `State` | variância zero |
| `Gender`, `Senior Citizen`, `Partner`, `Dependents` | **entram como features** (têm sinal preditivo) **e são auditadas ao lado** na §5g. ⚠️ Removê-las não removeria o viés — removeria a capacidade de medi-lo |

⚠️ **Limitação estrutural declarada:** não há **dados de uso**, de **contato com o suporte** nem de
**histórico de preço**. *Feature engineering não cria informação, só reorganiza a existente* — foi
por isso que as quatro features derivadas da Etapa 4 renderam ganho zero, e é a limitação nº 1 do
Model Card.

---

## ⚙️ Fazendo predições

| Campo | Valor |
|---|---|
| **Modo de consumo** | **híbrido, e a decisão é deliberada.** O consumo natural é **batch** (fila gerada por ciclo); a API em tempo real existe para (a) o CRM consultar o score quando o agente abre o card do cliente e (b) o requisito da fase |
| **Serviço** | FastAPI, `POST /predict` · `/v1/predict` · `/v1/predict-batch` (teto de 5.000 por lote) |
| **Latência** | p50 **254 ms** medido no cliente (Brasil→Oregon); **1,16 ms** no cronômetro da própria API. 🔑 *Latência sem ponto de medição declarado é métrica sem piso* |
| **Custo por linha** | 1.669,7 µs unitário × **2,0 µs em lote de 1.409** ⇒ **825×**. A API é intrinsecamente batch |
| **Limiar de operação** | **0,29**, derivado da curva de custo — viaja **dentro do artefato**, não no código |
| **Decisão humana** | 🚨 **sempre.** A predição é apoio; nenhuma ação é automática (LGPD Art. 20 + risco financeiro) |

---

## 🔁 Construindo e mantendo os modelos

| Gatilho de retreino | Papel | Ponto cego declarado |
|---|---|---|
| **Agendado (trimestral)** | **piso de segurança** — o gatilho principal | retreinar à toa gasta recurso e pode injetar ruído; o gate barra o modelo pior |
| **Data drift (PSI)** | reação | dispara **investigação**, nunca retreino automático |
| **Degradação de performance** | confirmação, nunca iniciativa | 🚨 **inútil sozinho:** o ground truth de churn leva ~60 dias (pressuposto declarado) ⇒ é retrovisor |
| **On demand** | evento extraordinário | depende de alguém perceber |

**Promoção:** *Continuous **Delivery***, não Deployment — o gate final é humano, porque a predição
dispara ação comercial com custo real. Nada é promovido sem passar no **gate de dois eixos**
(PR-AUC ≥ 0,66 **e** Brier ≤ 0,14, medidos na **validação**).

🎯 **E a trava já existia antes de ter esse nome:** o teste de caracterização (`|PR-AUC − ref| ≤
1e-4`) reprova **qualquer** modelo retreinado, inclusive um melhor ⇒ é impossível promover sem que
alguém edite a referência no mesmo commit, o que é um diff com autor e data.

**Rollback, em duas camadas e nesta ordem:** (1) painel da plataforma, em minutos; (2) `git revert`
do artefato + CI + push — porque o artefato é versionado junto do código, e reverter só a imagem é
uma correção que o próximo push de README desfaz **com o CI verde**; (3) conferir
`/health → artefato_sha256` **depois de cada camada**.

---

## 📡 Monitoramento

| Sinal | KPI | Gatilho | Desarme | Ação |
|---|---|---|---|---|
| **Data drift** | PSI por feature | **> 0,25** | < 0,10 | investigar → decidir retreino |
| **Prediction drift** | PSI dos scores + **taxa acima do limiar** | fila ±30% | — | 🔑 a fila enxerga antes do PSI, e já vem na unidade da decisão |
| **Categoria inédita** | taxa de **4xx** | qualquer 422 novo | — | evento de **serviço**, não de drift — o schema é derivado do artefato e recusa na porta |
| **Latência** | p95 **no cliente** | > 300 ms, excluído cold start | — | confrontar com o cronômetro interno: externo subindo com interno parado é rede |
| **Erros** | taxa de 5xx | > 1% em 10 min | — | incidente |
| **Qualidade preditiva** | PR-AUC | só após a janela cega (~60 dias) | — | reconciliação por `request_id` |

📏 **Os limiares foram calibrados contra o ruído do próprio sistema, não copiados:** o controle
"validação × treino" (onde nada aconteceu) dá **PSI 0,0128** ⇒ o gatilho de 0,25 tem **19,5×** de
folga. E a janela é de **volume (n ≥ 400)**, não de tempo, porque sem carga real "por dia" produz
decis de dois pontos.

🎯 **Linhas deliberadamente VAZIAS:** tráfego, saturação e taxa de aceitação da campanha ficam sem
limiar — não há carga real nem conversão observável. *Preencher esses campos com número plausível é
a forma mais fácil de transformar a seção de monitoramento em ficção.*

---

## ⚖️ Riscos, justiça e conformidade

| Item | Posição |
|---|---|
| **Base legal (LGPD)** | legítimo interesse (assumido); `CustomerID` fora das features |
| **Art. 20** | direito a explicação e revisão ⇒ a predição é **apoio**, com revisão humana — e o modelo linear entrega 13 coeficientes legíveis, não uma reconstrução aproximada |
| **Enquadramento AI Act** | churn com ação comercial **não** é alto risco (crédito e saúde seriam) — citado para situar, não para se eximir |
| **Disparidade medida** | 🚨 **58,89 pp** de diferença de recall em `Dependents`, **aceita e declarada**, com o preço das três alternativas medido |
| **Usos proibidos** | negar serviço · alterar preço · decisão automática sobre indivíduo · aplicar a segmento ausente do treino |
| **Feedback loop** | a campanha altera o rótulo de quem foi marcado ⇒ **grupo de controle** é necessário para medir o modelo limpo. Declarado, não implementado (não há operação real) |

---

## 🎲 O que este canvas assume — e não validou

Pressupostos são a categoria mais esquecida e a que mais derruba projeto: são as coisas que
ninguém escreveu porque pareciam óbvias.

1. **A taxa de churn histórica se mantém** — o dataset é um snapshot sem eixo temporal.
2. **As 13 features estarão disponíveis no momento da inferência**, com a mesma semântica.
3. **O ground truth chega em ~60 dias** — número **assumido**, não medido: o dataset não permite.
4. **A taxa de conversão da campanha** que sustenta os R$ 194 é premissa de negócio, não medição.
5. **A definição de churn herdada da IBM** corresponde à do negócio.

⚠️ Se qualquer um dos cinco for falso, o que muda não é o modelo — é o **enquadramento**, e o
caminho de volta é a Etapa 0.

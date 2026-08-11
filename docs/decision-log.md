# Decision Log — Tech Challenge Fase 1 (Churn Telecom)

> **Para que serve:** registrar *o que foi decidido, quais alternativas existiam e por quê*.
> Não é diário de bordo — é a matéria-prima da documentação da entrega (onde está a maior
> parte da nota). Regra: **decisão sem registro = decisão perdida**.
>
> Preencher junto com a execução, nunca depois. Cada linha deve ser rastreável até um run
> do MLflow quando envolver número.

- **Entrega:** 01/09/2026 · **Modalidade:** solo · **Peso:** 0–60 pts
- **Stack:** PyTorch (MLP) + scikit-learn + MLflow + FastAPI
- **Runbook:** skill `tech-challenge` (`.claude/skills/tech-challenge/SKILL.md`)

---

## 0. Enquadramento do problema

*Fechada em 07/08/2026.*

| Campo | Definição | Status |
|---|---|---|
| Definição exata do alvo | `Churn = Yes` — cliente cancelou o serviço. ⚠️ **Definição herdada do dataset, não construída por nós**: não sabemos se "cancelou" inclui suspensão, inadimplência ou downgrade total. Num projeto real esta seria a primeira pergunta ao negócio | ✅ |
| Janela de predição | **30 dias** — tempo hábil para o time abordar antes da decisão final, sem que o sinal comportamental enfraqueça. Padrão de go-to-market em retenção | ✅ |
| Ação de negócio disparada | **Fila priorizada de tarefas no CRM** (ex.: HubSpot) para a equipe de Retenção. O agente vê o ranking de risco, analisa o contexto e escolhe a abordagem (ligação / desconto / upgrade). **Nunca ação automática** — ver LGPD abaixo | ✅ |
| Taxa de churn base | *a confirmar na Etapa 1* — **previsão registrada antes de abrir o dado: ≈ 26,5%** | ⬜ |
| **Métrica primária** | **PR-AUC (average precision)** | ✅ |
| **Métricas operacionais** | **recall@10%** e **recall@20%** da base pontuada (curva de ganho acumulado) | ✅ |
| **Justificativa da métrica** | custo esperado **FN ≈ R$ 194** × **FP ≈ R$ 62** → razão **≈ 3:1** (conta abaixo) | ✅ |
| Métricas secundárias | recall, precision, F2, matriz de confusão, curva de ganho acumulado. **Acurácia apenas como contraste** ("o chute na majoritária já daria ~73%") | ✅ |
| Modo de execução | **Híbrido, e a decisão é deliberada:** o consumo natural é **batch** (fila gerada por ciclo); a **API em tempo real** existe para (a) o CRM consultar o score sob demanda quando o agente abre o card do cliente e (b) o requisito da Fase 1. Registrado para não fingir que a escolha foi orgânica | ✅ |

### Tabela de custo do erro — a base da métrica

> Premissas de calibração (telecom BR, ordem de grandeza): ARPU R$ 90/mês · margem de
> contribuição ~60% (R$ 54/mês) · desconto de retenção 20% por 6 meses (R$ 108) ·
> conversão da campanha ~30% · horizonte residual retido 12 meses.

| Erro | Em português | Composição do custo | Custo esperado |
|---|---|---|---|
| **Falso positivo** | demos atenção a quem ia ficar de qualquer jeito | 15 min do agente (≈ R$ 7,50) + desconto concedido em ~50% dos contatos (0,5 × R$ 108) | **≈ R$ 62** |
| **Falso negativo** | o cliente que ia sair passou batido | margem residual R$ 648 (12 × R$ 54) × **P(campanha converter) = 30%** | **≈ R$ 194** |

> 🔑 **O ajuste que muda a conclusão:** um FN **não** custa o LTV inteiro — custa só a **fatia
> recuperável**. 70% dos clientes perdidos não seriam salvos nem com a campanha. Isso derruba a
> razão de ~12:1 (intuição inicial) para **≈ 3:1**.
>
> **Consequência direta:** 3:1 **não** justifica otimizar recall puro — que tem solução trivial e
> catastrófica (marcar todo mundo → recall 100%, o time liga para a base inteira). Justifica um
> ranqueador bom, avaliado por PR-AUC, com o corte definido pela economia.

### Por que o modelo é um RANQUEADOR, não um classificador

A ação escolhida (fila priorizada no CRM) determina a natureza do artefato. Ninguém pergunta
*"esse cliente é churn: sim ou não?"* — perguntam *"quem eu ligo primeiro?"*. Logo:

- o **limiar 0,5 é irrelevante**; quem define o corte é a **capacidade operacional**;
- a métrica de seleção de modelo tem de ser **independente de limiar** → PR-AUC;
- o relatório ao negócio é a **curva de ganho acumulado** (recall@k);
- o limiar de operação é **parâmetro de negócio**, alterável sem retreinar nada.

### Cadeia KPI (métrica técnica → impacto de negócio)

> **PR-AUC mais alto → ranking melhor no topo da lista → dos 20% da base que o time consegue
> trabalhar por ciclo, capturamos uma fração maior dos churners (recall@20%) → com conversão de
> ~30% e margem residual de R$ 648 por cliente retido, cada ponto de recall@20% vale
> ≈ R$ 194 × (nº de churners no decil) → redução mensurável da taxa de churn.**

Regra de ouro registrada: **a métrica da Etapa 0, a do gate de CI (Etapa 9.5) e a do
monitoramento (Etapa 10) têm de ser a mesma.** (O PDF da Aula 08 viola isso — fala em recall no
texto e põe acurácia no gate. Não repetir.)

### Requisitos · Restrições · Pressupostos · Expectativas

| Categoria | Item | Validado? | Plano B se não se confirmar |
|---|---|---|---|
| Requisito | Entrega em **01/09/2026**, solo | ✅ | — |
| Requisito | **MLP em PyTorch** obrigatório (stack da fase) | ✅ | — |
| Requisito | **API de inferência** (FastAPI) + MLflow | ✅ | — |
| Requisito | Explicabilidade por cliente (LGPD Art. 20 — decisão automatizada que afeta o titular) | ✅ | SHAP sobre o modelo final |
| Restrição | Dataset **fixo, público e sem eixo temporal** | ✅ | — |
| Restrição | **Sem orçamento de infra** — tudo local/free tier | ✅ | MLflow local; CI no GitHub Actions |
| Restrição | **LGPD** — dados de cliente são dados pessoais | ✅ | `customerID` fora das features; máscara no log **antes** de ligar o log |
| Restrição | ~1h/dia de execução, 25 dias até a entrega | ✅ | priorizar etapas de maior retorno/nota |
| **Pressuposto** | as variáveis refletem o estado do cliente **no mês anterior** ao potencial cancelamento | ⬜ **não validável** | declarar no Model Card; se falso, a janela de 30 dias perde lastro → reenquadrar como "risco atual", sem janela |
| **Pressuposto** | a operação consegue trabalhar **10–20% da base por ciclo** | ⬜ | reportar a curva inteira de ganho, não um k único — sobrevive a qualquer tamanho de equipe |
| **Pressuposto** | campanha de retenção converte **~30%** dos abordados | ⬜ | a razão de custo 3:1 se move junto; refazer a tabela se o número real aparecer |
| **Pressuposto** | a taxa de churn histórica se mantém estável | ⬜ | é o que o monitoramento da Etapa 10 vigia (label/prior shift) |
| **Pressuposto** | todas as features estarão disponíveis **no momento da inferência** | ⬜ | testar uma a uma na Etapa 4 — é a pergunta-âncora anti-leakage |
| Expectativa | Fila priorizada consumível pelo CRM + explicação por cliente | ✅ | — |
| Expectativa | Documentação que justifique **decisões**, não só resultados (60% da nota) | ✅ | decision log preenchido durante, nunca depois |

### Limitação metodológica declarada (consequência do dataset ser um retrato)

Sem eixo temporal, **não é possível fazer split temporal** — o padrão correto em churn é treinar
no passado e testar no futuro. Usaremos **split aleatório estratificado**, o que assume
implicitamente que não há deriva temporal na base. É uma concessão real, imposta pelo dado, e
está declarada aqui em vez de escondida.

### Quem age sobre a predição

| Papel | Responsabilidade |
|---|---|
| **Equipe de Retenção** (humano) | recebe a fila, decide a abordagem, executa. **É quem age** — sem ela o modelo vale zero |
| **Gerente de Retenção** | define a capacidade operacional (o `k`) e a política de desconto; é quem deveria arbitrar a tolerância a FP |
| **TI / dono do CRM** | integração da API, e é quem conhece as limitações da base |
| **Jurídico / DPO** | base legal do tratamento e conformidade com o Art. 20 |

> ⚠️ Sendo entrega solo, **todos esses papéis são assumidos por uma pessoa** — o que é, em si,
> uma limitação de governança a declarar (ver Etapa 10.5, RACI).

---

## 1. Dados

*Fechada em 10/08/2026.*

**Fonte / versão:** `Telco_customer_churn.xlsx` — **variante estendida** da IBM (IBM Cognos
Analytics sample), não o CSV de 21 colunas mais difundido no Kaggle.
**SHA-256:** `1bcbc0ccc9b352175216979102628e579cfbde2c3ff57b005de168a433122640`
**Dimensões:** **7.043 linhas × 33 colunas**
**Taxa de churn:** **26,54%** (1.869 churners) → desbalanceado, mas **não severo**
**Baseline de chance:** chutar sempre "No" dá **73,46% de acurácia** — o piso contra o qual todo
resultado deve ser lido.

> 📌 **Por que a variante estendida, sendo que ela vem com colunas contaminadas:** precisamente
> por isso. Um dataset já limpo não permite demonstrar caça a leakage; a tabela abaixo é
> evidência de método. O enunciado exige ≥5.000 registros e ≥10 features — cumprido com folga.

### Colunas descartadas por suspeita de leakage

| Coluna | Por que suspeita | Decisão | Justificativa |
|---|---|---|---|
| **`Churn Reason`** | preenchida em **1.869** linhas, nula em **5.174** — exatamente churners × não-churners | ⛔ **descartar** | `notna()` prediz o alvo com **100% de acerto**. O motivo do cancelamento só existe *depois* do cancelamento. Leakage absoluto |
| **`Churn Score`** | não-churners: média 50 (máx **80**) · churners: média 82,5 (mín **65**) | ⛔ **descartar como feature** → 🎯 **usar como benchmark** | Duplo problema: (a) usá-la seria **prever o modelo da IBM**, não churn — destilação acidental, com teto de performance igual ao deles; (b) procedência desconhecida — se foi calculada retrospectivamente, é leakage puro |
| **`CLTV`** | valor calculado pela operadora, método desconhecido | ⛔ **fora do treino** → 🎯 **usar na ordenação da fila** | Diferença entre grupos é pequena (4.491 × 4.149), mas a procedência é opaca. Ganha uso melhor: ordenar a fila por **P(churn) × valor do cliente**, ligando a saída do modelo à conta de R$ 194 da Etapa 0 |
| `Churn Label` / `Churn Value` | — | alvo | `Churn Value` é o alvo (0/1); `Churn Label` é o mesmo em texto |
| `CustomerID` | identificador | ⛔ descartar | sem poder preditivo + **dado pessoal** (LGPD). A mesma linha serve às duas regras |
| `Count`, `Country`, `State` | **1 único valor** cada | ⛔ descartar | variância zero — todos os clientes são da Califórnia |
| `Lat Long` | string redundante | ⛔ descartar | já está em `Latitude` + `Longitude` |

> 🔑 **O critério aplicado, escrito uma vez:** uma coluna sai porque a informação **não estaria
> disponível no momento real da predição** ou porque **codifica o desfecho**. Não sai por ser
> "extra" nem por complexidade. Nada foi apagado — `data/raw` é imutável e toda decisão é
> reversível.

### Geográficas — decisão e trade-off registrado

`City` (1.129 valores) · `Zip Code` (1.652) · `Latitude`/`Longitude`.

**Decisão: fora das features na v1, dentro da auditoria de fairness.** Dois motivos independentes:

1. **Técnico (decisivo):** 1.652 CEPs em 7.043 linhas = **4,3 clientes por CEP**. One-hot geraria
   1.652 colunas; target encoding com 4 amostras por categoria vaza por construção.
2. **Ético:** CEP é o exemplo canônico de **proxy de renda e raça** (categoria socioeconômica).
   Usá-lo como feature impede distinguir *"aqui o sinal é ruim"* de *"aqui as pessoas são pobres"*.

⚠️ **Nuance registrada:** `Latitude`/`Longitude` são **contínuas** e escapam do problema de
cardinalidade — árvores cortam o espaço geográfico sem explodir dimensão. Sobrevivem à objeção
técnica, não à ética. **Backlog:** testá-las como experimento controlado, medindo performance
**e** disparidade com/sem. Vira seção de documentação de alto valor.

### Cobertura das 4 categorias de variáveis

| Categoria | Disponíveis no dataset | Faltam / seria bom ter |
|---|---|---|
| Demográficas | `Gender`, `Senior Citizen`, `Partner`, `Dependents` | renda, idade exata (só há o binário sênior) |
| **Comportamentais** ⭐ | `Tenure Months`, 9 flags de serviço (`Online Security`, `Tech Support`, `Streaming*`…), `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges` | 🚨 **consumo real (GB, minutos), nº de chamadas ao suporte, mudanças de plano** — os preditores mais fortes em churn real |
| Históricas | `Total Charges` (acumulado) | 🚨 **reclamações anteriores, atrasos de pagamento, upgrades/downgrades, histórico de reajuste** |
| Contextuais | geográficas (fora das features) | **cobertura de sinal na região, ofertas da concorrência, sazonalidade** |

> 🚨 **A maior limitação do dataset, e ela é estrutural:** não há **dados de uso** nem de
> **interação com o suporte**. Em churn de telecom real, "ligou 3 vezes no suporte no último mês"
> costuma ser o preditor mais forte que existe. O que temos são **atributos de contrato**, não de
> comportamento dinâmico. Isso limita o teto de performance alcançável e **deve constar nas
> limitações do Model Card**.

### Achados relevantes do EDA

| Achado | Número | Implicação para a modelagem |
|---|---|---|
| **`Contract` é o driver dominante** | mês-a-mês **42,7%** × 2 anos **2,8%** (amplitude **39,9 pp**) | confirma o preditor clássico de churn; flag `contrato_mensal` na Etapa 4 |
| **`Tenure` é fortemente não-linear** | 0-6m: **52,9%** → 6-12m: 35,9% → 1-2a: 28,7% → 2-4a: 20,4% → 4a+: **9,5%** | **binning ajuda muito LogReg e MLP** (entregam a não-linearidade de mão beijada); redundante para árvores |
| **Fibra ótica churna mais que DSL** | fibra **41,9%** × DSL 19,0% | contraintuitivo (é o produto premium) — hipótese: preço mais alto ou expectativa de qualidade não atendida. **Investigar antes de assumir** |
| **`Electronic check` dispara churn** | **45,3%** × ~16% nos pagamentos automáticos | pagamento manual = menos fricção para sair. Débito automático prende |
| **Serviços de valor agregado retêm** | sem `Online Security` 41,8% × com 14,6%; sem `Tech Support` 41,6% × com 15,2% | candidato a feature agregada: nº de serviços adicionais contratados |
| ⭐ **`Senior Citizen` tem prevalência muito diferente** | **41,7%** × 23,6% (**18,1 pp**) | 🎯 **crítico para a Etapa 10.5** — ver nota abaixo |
| **`Gender` não prediz nada** | 26,9% × 26,2% (**0,7 pp**) | ainda assim **deve ser auditado**: ausência de sinal ≠ ausência de disparidade nas *saídas* |
| **`Dependents` é forte** | sem 32,6% × com 6,5% (26,1 pp) | — |
| **Sem outliers univariados** | nenhuma numérica com \|z\| > 3 | não haverá etapa de tratamento de outlier; registrar que foi verificado |
| **`Total Charges` é redundante** | correlação **0,9996** com `Tenure × Monthly`, erro mediano 2% | multicolinearidade clássica — atrapalha LogReg, indiferente para árvores. Decidir na Etapa 5 |
| Correlações lineares modestas | `Tenure` −0,352 · `Monthly` +0,193 · `TC` −0,198 | nenhuma variável isolada resolve; o sinal está nas **combinações** — argumento a favor do MLP |

> ⭐ **A prevalência de 41,7% × 23,6% entre idosos e não-idosos é o achado mais consequente da
> EDA, e não é sobre performance.** Quando a **prevalência real difere entre grupos**, paridade
> demográfica, equalized odds e calibração tornam-se **matematicamente incompatíveis** (teorema de
> impossibilidade — Kleinberg, Chouldechova). Ou seja: **não existe escolher "ser justo" neste
> dataset** — só escolher *qual* definição de justiça otimizar, sabendo que isso sacrifica as
> outras. A Etapa 10.5 tem de declarar qual foi escolhida e por quê. Isto deixa de ser teoria da
> Aula 07 e passa a ser um número próprio.

### Zeros e vazios investigados (armadilha do rótulo censurado)

| Coluna | O vazio é medição ou ausência de medição? | Tratamento |
|---|---|---|
| **`Total Charges`** (11 casos, todos com `Tenure Months = 0`) | **é medição verdadeira** — o cliente pagou zero porque **não houve ciclo de faturamento**, não porque o dado se perdeu | **imputar `0`**. ⛔ **Não** imputar mediana (~R$ 1.400): inventaria histórico de pagamento para quem nunca foi faturado, e de forma plausível o bastante para passar despercebido. ⛔ **Não** remover as linhas: clientes com `tenure = 0` **existem em produção** — apagá-los do treino não elimina o caso, transfere para a API |
| `"No internet service"` / `"No phone service"` | terceira categoria legítima nas flags de serviço | **manter como categoria**, não converter em nulo — carrega informação real (churn de 7,4%, o mais baixo do dataset) |

> ℹ️ **Não é preciso criar flag `cliente_novo`:** `Tenure Months = 0` já é a flag. A informação
> não se perde com a imputação por zero.

### Hipótese testada e REJEITADA

**Hipótese:** `Total Charges / Tenure` (cobrança média histórica) versus `Monthly Charges` atual
revelaria **reajuste de preço** — cliente que sofreu aumento cancelaria mais.

**Resultado:** a razão tem mediana **1,000** e desvio-padrão **0,051**; o churn por quartil é
25,6% / 32,9% / 20,8% / 24,9% — **sem padrão monotônico**. O dataset é estático demais para
carregar histórico de reajuste. **Feature descartada antes de ser construída.**

> Registrado deliberadamente: hipótese testada e rejeitada **é resultado**, e responde à
> provocação do runbook — *"você vai medir o ganho dessa feature ou só assumir que ajudou?"*.

⚠️ **Mas a divisão sobrevive como armadilha:** qualquer feature com `Tenure Months` no
denominador produz `inf` nas 11 linhas com tenure zero. **Isso vira teste unitário obrigatório
no CI** (Etapa 9.5) — é exatamente o caso que a Aula 08 pedia (*"verificar se a engenharia de
features não gera valores infinitos"*), com ocorrência real neste dataset.

---

## 2. Preparação

*Fechada em 10/08/2026.* Código: `src/config.py` (contrato de colunas), `src/data.py`
(carregamento e partição), `src/preprocess.py` (o pipeline serializável).

| Decisão | Escolha | Alternativa considerada | Por quê |
|---|---|---|---|
| **Estratégia de split** | **três** partições **60/20/20**, aleatórias e **estratificadas**, `seed=42` | treino/teste 80/20 (o que o runbook sugere) | três coisas já decididas exigem um conjunto de **validação fixo**: early stopping do MLP (Etapa 8), gate do CI (Etapa 9.5) e seleção entre algoritmos (Etapa 6). Reparticionar depois invalidaria toda a tabela mestra, porque os números deixariam de ser comparáveis |
| **Conjunto de teste** | **intocado até o fim** — nenhuma decisão desta etapa o consultou | avaliar tudo no teste, como o enunciado da Aula 08 sugere | decidir repetidamente olhando o teste o converte em validação e ele deixa de estimar generalização. Registrado como divergência deliberada do material |
| Imputação de `Total Charges` | **constante 0** | mediana (~R$ 1.400); remover as 11 linhas | o vazio é medição verdadeira (sem ciclo de faturamento). Mediana inventaria histórico de pagamento de forma plausível demais para ser notada; remover transferiria o caso para produção, onde ele existe |
| Imputação das demais numéricas | **mediana** | média | robusta a outlier. Não há nulo no dataset — está no pipeline como defesa para o que a API receber |
| Imputação categórica | **moda** | categoria "desconhecido" | idem: defesa para produção, no-op no treino |
| **Encoding** | **`OneHotEncoder(handle_unknown="ignore")`, sem `drop`** | `get_dummies(drop_first=True)` (o que o runbook sugere) | duas divergências, a segunda com prova — ver abaixo |
| Escalonamento | `StandardScaler` nas numéricas, **parametrizável** | escalar sempre | obrigatório para LogReg e MLP; indiferente para árvores. `construir_preprocessador(escalonar=False)` já existe para a Etapa 6 |
| **Desbalanceamento** | **nenhum tratamento** | SMOTE; `class_weight="balanced"` | 26,5% não é severo, e ambos **descalibram a probabilidade** — o que quebra a ordenação da fila por `P(churn) × CLTV`. `balanced` foi medido e não agregou nada (ver §5) |
| Outliers | **nada a fazer** | winsorização, corte por z-score | a Etapa 1 verificou: nenhuma numérica com \|z\| > 3. Registrado que foi **verificado**, não ignorado |
| Tipos das numéricas | forçadas a **`float64`**, inclusive `Tenure Months` (int no arquivo) | manter o dtype original | inteiro em Python não representa nulo: um campo faltando na API viraria float e quebraria a validação de schema do MLflow. Erro que só apareceria em produção |

### As duas divergências no encoding — a segunda com evidência

**`get_dummies` → `OneHotEncoder`:** `get_dummies` não memoriza as categorias vistas no treino,
logo não sobrevive à API. O `OneHotEncoder` guarda o vocabulário dentro do artefato — é o que
transforma convenção implícita em contrato.

**`drop_first=True` → sem `drop`:** testado antes de decidir. Com `drop="if_binary"` +
`handle_unknown="ignore"`, uma categoria inédita é codificada como vetor de zeros — que é
**exatamente o vetor da categoria dropada**:

| Entrada | `drop="if_binary"` | sem `drop` |
|---|---|---|
| `Gender="Female"` | `[0.]` | `[1, 0]` |
| `Gender="Male"` | `[1.]` | `[0, 1]` |
| `Gender="Other"` (inédita) | `[0.]` ← **colide com "Female"** | `[0, 0]` ← distinto |

Ou seja: com `drop`, um `Gender="Other"` chegando na API seria silenciosamente tratado como
`"Female"`, com status 200 OK. A *dummy variable trap* que o `drop_first` evita é inofensiva sob a
regularização L2 que o sklearn já aplica por padrão; a colisão não é. Coberto por
`test_categoria_desconhecida_nao_colide_com_categoria_existente`.

### A distinção que o gate da Etapa 2 costuma embaralhar

O gate diz *"toda transformação que aprende parâmetro é ajustada só no treino"*. Correto, mas são
**duas regras diferentes**, e confundi-las leva a decidir errado:

| Regra | Contra o quê protege | Vale para |
|---|---|---|
| `fit` **só no treino** | **data leakage** | mediana do imputador, média/desvio do scaler, categorias do encoder — tudo que *aprende* dos dados |
| estar **dentro do artefato serializado** | **training-serving skew** | *tudo*, inclusive o que não aprende nada |

O `fillna(0)` do `Total Charges` é o caso que separa as duas: sendo constante, **não seria leakage
nem se fosse aplicado antes do split** — mas está dentro do `Pipeline` mesmo assim, porque precisa
viajar no artefato. Já a conversão de texto para número ficou **fora** do pipeline, em
`carregar_bruto()`: ela é defeito do `.xlsx`, e na API o campo já chega tipado pelo Pydantic.

### 🚨 O arquivo bruto está ORDENADO pelo alvo — descoberto por um teste que quebrou

As **1.869 primeiras linhas são todos os churners**; as 5.174 seguintes, todos os não-churners.
Zero mistura. Três consequências, agora fixadas em `test_arquivo_bruto_esta_ordenado_pelo_alvo`:

1. Um split com `shuffle=False` daria **treino 100% churner e teste 100% não-churner**. Escapamos
   pelo *default* do `train_test_split`, não por decisão consciente — agora é consciente e testada.
2. 🎯 **Impacto direto na Etapa 9.5:** o "CI leve" recomendado treina num *subset minúsculo* via
   `head()`. Aqui isso traz **uma classe só** e o `LogisticRegression` levanta
   `ValueError: needs samples of at least 2 classes` — uma mensagem que não aponta para a causa.
   O subset do CI tem de ser **estratificado**.
3. Explorar a base com `df.head()` dá leitura completamente enviesada.

> Foi este teste que cumpriu o critério do gate da Etapa 9.5 — *"pelo menos um dos testes já pegou
> um erro de verdade"*. Ele não foi escrito sabendo do problema; ele **encontrou** o problema.

---

## 3. Features criadas

*Fechada em 10/08/2026.* Código: `src/features.py`, medição em `src/ablacao.py`.

**Resultado da etapa: nenhuma das 4 features entra no modelo v1.** Todas foram implementadas,
medidas e descartadas por ausência de ganho. O código permanece testado e ligável por parâmetro
(`--features`) para reavaliação na Etapa 8.

| Feature | Hipótese de negócio | Disponível na predição? | Ganho medido (CV, PR-AUC) | Mantida? |
|---|---|---|---|---|
| `n_servicos_adicionais` | cada serviço a mais é um fio prendendo o cliente → **custo de troca** | ✅ sim — atributo de contrato | **+0,0001** | ❌ |
| `tenure_faixa` (binning) | o risco despenca no 1º semestre e estabiliza; a LogReg só vê a reta | ✅ sim | **+0,0001** | ❌ |
| `charge_por_servico` | pagar caro por pouco serviço gera insatisfação (custo-benefício) | ✅ sim | **−0,0010** | ❌ |
| `pagamento_automatico` | débito automático é inércia a favor da empresa; pagamento manual é decisão mensal | ✅ sim | **+0,0001** | ❌ |

*Ruído de referência: desvio entre folds = **0,0280**. Nenhum ganho chega a 1/20 disso.*

**Duas candidatas cortadas antes de escrever código**, por serem redundantes **por construção**:
`contrato_mensal` e `tem_fibra` — o `OneHotEncoder` de `Contract` e `Internet Service` já produz
exatamente essas colunas.

### Como o ganho foi medido — duas decisões metodológicas

1. **Validação cruzada estratificada (5 folds) dentro do TREINO**, não na validação. Julgar cada
   candidata olhando a validação a gastaria em cinco decisões — o mesmo problema que nos fez
   manter o teste intocado, um andar acima. A validação segue reservada para a escolha final
   entre modelos.
2. **Ablação por remoção (leave-one-out)**, não por adição isolada. Adicionar uma feature sozinha
   superestima sua contribuição quando ela é redundante com outra; medir quanto se **perde** ao
   removê-la do conjunto completo responde a pergunta certa — *"agrega algo que as outras já não
   dizem?"*.

### 🔑 Por que falharam — o diagnóstico é mais útil que o resultado

**Duas eram redundantes matematicamente, não por acaso.** `n_servicos_adicionais` é a **soma de 6
dummies já presentes** no modelo; `pagamento_automatico` é o **agrupamento de 4 dummies já
presentes**. Para um modelo linear, uma combinação linear de variáveis existentes não acrescenta
grau de liberdade nenhum — a LogReg já podia representar exatamente aquilo. Isso era previsível
antes de medir, e foi um erro de julgamento não tê-lo previsto.

**A hipótese de salvação foi testada e também rejeitada.** O princípio da Aula 03 diz que FE
depende do algoritmo alvo: uma árvore precisa de 6 splits sucessivos para "contar serviços",
enquanto a contagem entrega isso num split. Rodamos a mesma ablação com **Random Forest**:

| Modelo | base (19 features) | + as 4 novas | Δ |
|---|---|---|---|
| LogReg | 0,6868 ± 0,0236 | 0,6871 ± 0,0280 | **+0,0003** |
| Random Forest | 0,6878 ± 0,0128 | 0,6811 ± 0,0137 | **−0,0067** |

Na RF as features **pioram**, e cada remoção individual melhora. Mecanismo: features redundantes
**diluem o sorteio de colunas em cada split** — a árvore passa a considerar cópias da mesma
informação no lugar de variáveis informativas.

**A conclusão de fundo amarra com a Etapa 1:** *feature engineering não cria informação, apenas
reorganiza a existente.* A EDA já havia registrado que a maior limitação do dataset é
**estrutural** — não há dados de uso nem de interação com o suporte, só atributos de contrato.
A Etapa 4 confirma empiricamente aquela previsão: não há como derivar sinal que não está lá.
Isso vai para as limitações do Model Card.

### ⚠️ O split único teria dado a resposta errada

Rodado na validação, o conjunto com features dá **PR-AUC 0,6690 contra 0,6623** — um ganho
aparente de **+0,0067** que, olhado sozinho, levaria a incluir as quatro. Mas a CV já havia
estimado o desvio entre folds em **0,0280**: o "ganho" cabe quatro vezes dentro do ruído.

> É a provocação que o runbook reserva para a Etapa 6 (*"isso é sinal ou é variância do split?"*)
> materializada uma etapa antes. **Uma observação contra cinco: ficamos com as cinco.** Registrado
> porque é o argumento que sustenta usar CV em toda a Etapa 6.

### 🚨 Achado técnico para a Etapa 9: o `FunctionTransformer` acopla o artefato ao código-fonte

Serializar o pipeline com feature engineering falhou até declararmos
`skops_trusted_types=["src.features.adicionar_features"]`. A causa importa mais que a correção:
o artefato **carrega uma referência a uma função nossa**, logo o ambiente de inferência precisa
ter `src.features` importável **no mesmo caminho**, senão o modelo não carrega. É a versão skops
do problema clássico do pickle com funções customizadas — e é uma restrição real para o
empacotamento da API e para o Dockerfile.

---

## 4. Seleção de features

*Fechada em 10/08/2026.* Código: `src/selecao.py`.

- **Nº de features: 19 → 13** (46 → 33 colunas pós one-hot)
- **Método escolhido: ablação por remoção da feature ORIGINAL**, com CV estratificada de 5 folds
  no treino — não `SelectKBest`, não `RFE`, não L1 (todos rodados e reportados abaixo)
- **Ganho de performance: nenhum**, e isso é a conclusão, não uma falha

### O enquadramento — o que a seleção está otimizando aqui

Com **46 colunas e 4.225 amostras de treino (92 amostras por coluna)** não há pressão
dimensional: nada obriga a podar. A seleção neste projeto não busca métrica, busca **custo
operacional**, e o argumento vem da Etapa 10:

> cada feature vigiada é uma chance a mais de alerta falso de drift (80 features a 5% produzem
> ~4 alertas espúrios por rodada → fadiga de alerta), e cada feature coletada precisa ser
> obtida, validada e mantida no CRM.

### 🔑 A distinção que muda o desenho do experimento

Selecionar **depois** do one-hot não é selecionar features. Se o `SelectKBest` descartar a dummy
`Payment Method_Mailed check` e mantiver as outras três, **a coluna `Payment Method` continua
tendo que ser coletada, validada e monitorada**. Reduziu-se a dimensão do modelo, não o custo
operacional. Por isso o experimento foi feito nos **dois níveis**, e a decisão saiu do segundo.

### Nível 1 — os três tipos da Aula 03, sobre as 46 colunas

Referência: 46 colunas → **PR-AUC 0,6868 ± 0,0236**

| Tipo | Método | k | PR-AUC | Δ |
|---|---|---|---|---|
| Filtro | `SelectKBest(f_classif)` | 10 | 0,6648 | −0,0219 |
| Filtro | `SelectKBest(f_classif)` | 20 | 0,6692 | −0,0175 |
| Filtro | `SelectKBest(f_classif)` | 30 | 0,6743 | −0,0124 |
| Wrapper | `RFE` | 10 | 0,6568 | −0,0300 |
| Wrapper | `RFE` | 20 | 0,6790 | −0,0078 |
| Embedded | L1 (C=0,05) | — | 0,6858 | −0,0010 |

**Todos pioram.** O filtro é o pior justamente onde o runbook avisa: `f_classif` avalia **uma
coluna de cada vez** e é cego a interação, então descarta dummies que só fazem sentido em par.
O embedded (L1) é o único que empata — coerente, porque ele decide **durante** o treino, com o
modelo inteiro à vista.

### Nível 2 — ablação por feature original (a que decidiu)

| Feature | Perda ao remover | |
|---|---|---|
| `Dependents` | **+0,0221** | a mais importante do dataset |
| `Tenure Months` | +0,0095 | |
| `Paperless Billing` | +0,0059 | |
| `Contract` | +0,0025 | |
| *(demais)* | < 0,0025 | dentro do ruído |
| **6 features** | **≤ 0** | remover melhora ou não muda |

⚠️ **Honestidade sobre o ruído:** exceto `Dependents`, **todas as diferenças individuais estão
dentro do desvio entre folds (0,0236)**. Nenhuma feature isolada pode ser declarada descartável
com base no seu próprio número. O que decidiu foi o teste **em conjunto**:

| Conjunto | PR-AUC | ±dp | Δ |
|---|---|---|---|
| 19 features (referência) | 0,6868 | 0,0236 | — |
| **13 — sem as 6 não-sensíveis** ✅ | **0,6912** | **0,0163** | +0,0044 |
| 10 — sem as 6 + os 3 sensíveis | 0,6912 | 0,0184 | +0,0044 |
| 16 — sem só os 3 sensíveis | 0,6887 | 0,0208 | +0,0019 |

**Escolhida a de 13.** Mesma performance com 6 colunas a menos — e o **menor desvio entre folds
de todas as variantes** (0,0163 contra 0,0236), isto é, um modelo mais estável, não só mais
enxuto. Na validação: PR-AUC 0,6646 contra 0,6623 do modelo de 19 — equivalente, como a CV previu.

**Removidas:** `Phone Service`, `Online Backup`, `Device Protection`, `Streaming TV`,
`Streaming Movies`, `Payment Method`.

### 🚨 `Payment Method` sai sem custo — e a explicação é correlação espúria

A EDA havia registrado `Electronic check` com **45,3% de churn** contra ~16% dos pagamentos
automáticos: um dos sinais mais fortes do dataset. Que ele saia sem custo exigia explicação, e a
tabela cruzada dá:

| Payment Method | Mês-a-mês | 1 ano | 2 anos |
|---|---|---|---|
| **Electronic check** | **78%** | 15% | 7% |
| Bank transfer (automatic) | 38% | 25% | 37% |
| Credit card (automatic) | 36% | 26% | 38% |

**78% dos clientes de `Electronic check` estão em contrato mês-a-mês.** O que parecia efeito da
forma de pagamento é, em boa parte, o efeito do **tipo de contrato** — que já está no modelo via
`Contract`. É o caso do catálogo (*"pacientes do médico X têm mais readmissões"*, quando o médico
X atende os casos graves), agora com número próprio. A variável não deixou de correlacionar com
churn; ela deixou de **acrescentar** informação.

### Atributos sensíveis — decisão deliberadamente adiada

`Gender`, `Senior Citizen` e `Partner` também aparecem como removíveis, e **removê-los custa
exatamente zero** (as variantes de 13 e de 10 dão o mesmo 0,6912). Mesmo assim **ficam na v1**:

- a escolha entre *"dentro das features"* e *"fora, guardados ao lado para auditar"* depende do
  resultado da auditoria de fairness, que só existe na Etapa 10.5. **Decisão de governança não se
  toma sem o dado da governança** — e o runbook é explícito em que essa não é decisão do
  engenheiro sozinho;
- como o custo de mantê-los é nulo, não há pressa técnica que justifique antecipar.

> 🔑 **Achado que vale para a Etapa 10.5:** `Senior Citizen` tem sinal real e forte (churn de
> 41,7% × 23,6%), e ainda assim **removê-lo não custa performance**. A única leitura possível é
> que o modelo **reconstrói a informação por proxies** (`Contract`, `Dependents`,
> `Monthly Charges`). Isto é *fairness through unawareness* demonstrado empiricamente neste
> dataset: tirar a coluna não tiraria o viés — tiraria apenas a capacidade de medi-lo.

### ⛔ O GATE da etapa, testado em vez de afirmado

A seleção entra **como passo do `Pipeline`**, ajustada dentro de cada fold. Para medir o custo de
fazer errado, rodamos `SelectKBest` **ajustado sobre o dataset inteiro** antes da CV:

| | PR-AUC |
|---|---|
| seleção dentro do pipeline (correto) | 0,6636 |
| seleção fora do pipeline (vazada) | 0,6635 |
| **inflação** | **−0,0000** |

**Não inflou nada — e a explicação importa mais que o número.** Com 4.225 amostras e 46 colunas a
seleção é **estável**: qualquer 80% dos dados elege praticamente as mesmas colunas, então ver o
alvo inteiro não muda a escolha. O leakage do seletor só morde quando a seleção é **instável**,
isto é, quando `p >> n`.

Para não deixar a regra como dogma não verificado, reproduzimos o regime onde ela morde —
**300 amostras e 500 colunas de ruído puro**, nenhuma com qualquer relação com o alvo:

| | PR-AUC |
|---|---|
| prevalência (o piso honesto) | 0,2533 |
| seleção dentro do pipeline | 0,3402 |
| **seleção fora do pipeline** | **0,7167** |
| **inflação sobre puro ruído** | **+0,3766** |

O modelo "prevê" com PR-AUC 0,72 a partir de **colunas aleatórias**. Todo esse desempenho é
artefato de a seleção ter visto o alvo inteiro.

> ⚠️ A conclusão **não** é "pode selecionar fora do pipeline quando há muitas amostras". É que o
> custo de fazer certo é zero, a magnitude do erro depende do regime, e o pipeline é necessário de
> qualquer forma para a API. Mas medir a própria armadilha em vez de citá-la é o que separa
> conhecer a regra de entender o mecanismo.

### Efeito cascata: a Etapa 5 deixou órfãs 3 das 4 features da Etapa 4

Remover 6 colunas retirou o insumo de três features derivadas: `n_servicos_adicionais` (4 dos 6
serviços que ela somava saíram), `charge_por_servico` (dependia da contagem) e
`pagamento_automatico` (`Payment Method` saiu). Sobreviveu apenas `tenure_faixa` — não por acaso,
a única que não era combinação linear de colunas já presentes.

Código órfão removido, e um teste novo
(`test_features_derivadas_nao_dependem_de_coluna_removida`) trava a cascata. Registrado em vez de
apagado porque **uma decisão posterior pode invalidar código anterior**, e o encadeamento é o que
o decision log existe para preservar.

---

## 5. Experimentos — tabela mestra

> Toda linha corresponde a um run rastreável no MLflow (backend `sqlite:///mlflow.db`,
> experimento `churn-fase1`). **Todos os números são de VALIDAÇÃO** — o conjunto de teste
> permanece intocado até o fim do projeto.

*Etapa 3 fechada em 10/08/2026.*

| # | Run | Modelo | PR-AUC ⭐ | ROC-AUC | R@10% | R@20% | F1 | Brier | Gap PR-AUC | Custo mín. | Limiar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `66494335` | **Chute na majoritária** | **0,2654** | 0,500 | 0,088 | 0,201 | 0,000 | 0,195 | −0,000 | R$ 64.170 | — |
| 1 | `c15f8839` | **LogReg (MVP)** ✅ | **0,6623** | 0,850 | 0,289 | 0,524 | 0,618 | **0,133** | +0,033 | **R$ 31.092** | 0,22 |
| 2 | `ce070deb` | LogReg `class_weight=balanced` | 0,6638 | 0,850 | 0,286 | 0,524 | 0,637 | 0,161 | +0,030 | R$ 31.138 | 0,54 |
| 3 | `6ba7797f` | LogReg **+ 4 features** (Etapa 4) | 0,6690 | 0,853 | 0,286 | 0,529 | 0,617 | 0,133 | +0,030 | R$ 30.842 | 0,27 |
| 4 | `aedc361b` | **LogReg, 13 features** (Etapa 5) ✅ | **0,6646** | 0,847 | 0,278 | 0,516 | 0,599 | 0,134 | +0,031 | R$ 31.750 | 0,29 |
| 5 | `etapa6-finalistas` | **Random Forest** regularizada (Etapa 6) | 0,6595 | — | 0,275 | 0,505 | — | 0,134 | +0,171 | R$ 30.172 | **0,27** |
| 6 | `etapa6-finalistas` | **HistGradientBoosting** regularizado (Etapa 6) ⭐ | **0,6678** | — | **0,283** | 0,497 | — | **0,132** | +0,083 | **R$ 29.626** | **0,22** |
| — | `b1ff54a9` | *`Churn Score` da IBM (referência)* | *0,8824* | *0,949* | *0,377* | *0,655* | *0,595* | — | — | *R$ 16.802* | *0,65* |

⭐ métrica primária · ✅ **modelo de referência corrente** (run 1 é o baseline da Etapa 3;
run 4 é o mesmo modelo com 13 features, adotado na Etapa 5 por custo operacional, não por métrica)

### Leitura do baseline

**O modelo bate o chute na majoritária?** Sim, com folga: PR-AUC **0,6623 × 0,2654** — 2,5× o
piso. Um ganho dessa ordem descarta a hipótese "os dados não têm sinal".

**Gap treino-validação = +0,033.** Pequeno: não há overfitting relevante numa LogReg com 19
features e 4.225 amostras, o que era esperado. ⚠️ Vale repetir que este gap **não detectaria
leakage** — se a contaminação atingisse treino e validação igualmente, ele continuaria bonito.
Quem cobre esse flanco é a auditoria da Etapa 1 e o `test_colunas_de_leakage_fora_das_features`.

**Recall@k precisa ser lido contra o teto estrutural.** Nos k% do topo cabem no máximo `k×N`
clientes, então nenhum ranqueador pode capturar mais que `k / prevalência` dos churners. Com
prevalência 26,54%, o teto de recall@10% é **0,377** e o de recall@20% é **0,754**. Logo:

| | recall@10% | % do teto | recall@20% | % do teto |
|---|---|---|---|---|
| aleatório | 0,100 | 26,5% | 0,200 | 26,5% |
| **LogReg** | **0,289** | **76,7%** | **0,524** | **69,6%** |
| teto | 0,377 | 100% | 0,754 | 100% |

Reportar "recall@10% = 0,289" sozinho faz um ranking decente parecer fraco. **76,7% do máximo
matematicamente possível** é a leitura honesta.

### A conta de negócio — a cadeia KPI da Etapa 0 fechada com número real

Sobre os 1.409 clientes da validação, aos custos da Etapa 0 (FN R$ 194 · FP R$ 62):

| Estratégia | Custo do erro | Comentário |
|---|---|---|
| Não abordar ninguém | **R$ 72.556** | 374 churners perdidos |
| Abordar a base inteira | **R$ 64.170** | a "solução" do recall 100% — e ela é cara |
| **LogReg, limiar 0,22** | **R$ 31.092** | **−52% sobre não fazer nada** |
| *`Churn Score` da IBM* | *R$ 16.802* | *ver ressalva abaixo* |

≈ **R$ 23 economizados por cliente por ciclo** sobre a melhor estratégia trivial. É este número
que liga PR-AUC a reais, e é ele que deve aparecer no vídeo de 5 minutos.

> Note que "abordar a base inteira" custa **R$ 64 mil** — mais caro que muitos supõem. É o
> argumento concreto contra otimizar recall puro, que já havia sido antecipado na Etapa 0 e agora
> tem número.

### 🚨 O `Churn Score` da IBM não é benchmark — é gabarito vazado (confirmado)

A Etapa 1 o barrou por *procedência opaca*. A suspeita agora está confirmada, com evidência:

| Faixa de score | Não-churners | Churners |
|---|---|---|
| 0–50 | 536 | **0** |
| 50–65 | 245 | 10 |
| 65–80 | 254 | 143 |
| 80–100 | **0** | 221 |

**Nenhum não-churner passa de 80 e nenhum churner fica abaixo de 65** — zero exceções em 1.409
linhas. Um score genuinamente preditivo não separa assim; isto é a assinatura de cálculo feito
**com o desfecho já conhecido**. Consequências registradas:

- o PR-AUC de 0,88 é **inatingível por construção** — não é meta, e usá-lo como referência de
  qualidade levaria a concluir que o nosso modelo "falhou" quando ele não falhou;
- confirma retroativamente que mantê-lo fora das features foi a decisão certa: com ele, o modelo
  teria PR-AUC ~0,88 e **estaria prevendo o desfecho, não o risco**;
- fica na tabela em *itálico*, como caso documentado de leakage, não como concorrente.

### `class_weight="balanced"` foi medido e descartado

A comparação decisiva não é o PR-AUC (0,6638 × 0,6623, diferença dentro do ruído) — é esta:

| | limiar ótimo | custo mínimo | Brier (calibração) |
|---|---|---|---|
| LogReg | 0,22 | R$ 31.092 | **0,133** |
| LogReg `balanced` | 0,54 | R$ 31.138 | 0,161 |

`balanced` **chega ao mesmo custo** (R$ 46 de diferença, 0,15%) apenas **deslocando o limiar** de
0,22 para 0,54 — faz por dentro do modelo o que o limiar já faz por fora. E cobra por isso: o
Brier piora **21%**, isto é, a probabilidade fica descalibrada. Como a fila é ordenada por
`P(churn) × CLTV`, uma probabilidade descalibrada corrompe a multiplicação — deixa de significar
reais. **Descartado.**

> Este é o argumento empírico para a decisão da Etapa 2 de não tratar desbalanceamento: em um
> ranqueador, desbalanceamento é assunto de **limiar**, e limiar é **parâmetro de negócio**. Daí
> a exigência já registrada de que a API devolva **probabilidade, não classe** — o gerente muda o
> corte sem que ninguém retreine nada.

---

## 5b. Etapa 6 — comparação de algoritmos

*Fechada em 11/08/2026.* Reprodução: `python -m src.comparacao` (fase 1) e
`python -m src.finalistas` (fase 2).

**Protocolo:** CV estratificada **repetida** (5 folds × 3 = 15 fits por candidato) no
**treino**, métrica PR-AUC — a mesma partição, seed e lista de 13 features para todos.
A validação foi tocada **uma vez**, só pelos três finalistas. O teste segue intocado.
Por que CV repetida e não a validação: um split único já deu a resposta errada na Etapa 4,
e é a CV que entrega o **desvio entre folds** sem o qual não se distingue ganho de sorteio.

### Resultado — CV no treino (11 configurações)

| Modelo | Encoding | PR-AUC | ±dp | Δ vs LogReg | Δ em dp | Gap treino→CV |
|---|---|---|---|---|---|---|
| **HistGradientBoosting reg.** | ordinal | **0,6919** | 0,0188 | +0,0016 | +0,07 | 0,083 |
| HistGradientBoosting reg. | onehot | 0,6915 | 0,0192 | +0,0012 | +0,05 | 0,084 |
| **LogReg (baseline)** | onehot | **0,6904** | 0,0236 | — | — | **0,005** |
| Random Forest reg. | ordinal | 0,6886 | 0,0195 | −0,0018 | −0,08 | 0,171 |
| Random Forest reg. | onehot | 0,6868 | 0,0200 | −0,0035 | −0,15 | 0,160 |
| HistGradientBoosting | onehot | 0,6667 | 0,0168 | −0,0237 | −1,00 | 0,252 |
| Random Forest | ordinal | 0,6519 | 0,0235 | −0,0384 | −1,63 | 0,348 |
| Random Forest | onehot | 0,6437 | 0,0214 | −0,0466 | −1,98 | 0,356 |
| Árvore única | onehot | 0,3971 | 0,0161 | −0,2932 | −12,44 | **0,603** |

**A conclusão principal é um empate técnico.** Os três primeiros ficam dentro de **0,0033**
um do outro, contra um desvio entre folds de **0,0188–0,0236**. A maior diferença observada
vale **0,07 desvio-padrão** — ou seja, **nenhuma família de modelo é distinguível das outras
neste dataset**. Declarar vencedor por PR-AUC aqui seria ler ruído.

> 🔑 **O achado que vale mais que o ranking: o eixo que move o número é a REGULARIZAÇÃO, não a
> família.** A mesma Random Forest vai de 0,6519 para 0,6886 (+1,6 dp) apenas com
> `min_samples_leaf=5`; o mesmo boosting vai de 0,6667 para 0,6919 (+1,1 dp). Essas diferenças
> são **maiores que qualquer diferença entre algoritmos distintos**. Contraria a afirmação
> corrente (e o material da disciplina) de que Random Forest "não precisa de ajuste fino":
> neste dataset, não regularizar custa mais do que escolher o algoritmo errado.

### Confirmação do contrafactual da Etapa 3

A LogReg entrega, com **gap de 0,005**, o que o boosting só alcança com gap de 0,083 e a
floresta com 0,171. Isso confirma a previsão registrada antes de medir: **o sinal do Telco é
essencialmente linear no logit**. Os modelos não-lineares têm liberdade funcional para
representar interações e não-monotonicidades — e, tendo-a, **não encontram nada de relevante
para fazer com ela**. Não é fracasso dos ensembles: é uma descrição do problema.

### O papel da árvore única — o argumento viés-variância medido

A árvore sem poda **não foi incluída para competir**; ela é o controle do argumento. Três
medidas dos próprios dados, no lugar da citação bibliográfica:

| Medida | Árvore única | RF regularizada | Leitura |
|---|---|---|---|
| PR-AUC (CV) | 0,3971 | 0,6886 | a floresta vale +73% sobre a mesma família |
| PR-AUC no **treino** | **1,0000** | 0,8596 | a árvore memoriza o treino **perfeitamente** |
| Gap treino→CV | **0,603** | 0,171 | e não leva nada disso para dados novos |
| **Desvio da probabilidade prevista em 10 reamostragens** | **0,2462** | **0,0438** | **5,6×** |

A última linha é a medida direta da variância: reamostrando o treino por bootstrap e predizendo
sempre nos mesmos clientes da validação, a probabilidade que a **árvore** atribui ao mesmo
cliente oscila **±24,6 pontos percentuais** conforme o sorteio dos dados; na floresta, ±4,4 pp.
É a versão quantificada de *"dependendo do conjunto de treino, a regra da raiz mudava"* — e o
motivo de existir do ensemble, demonstrado em vez de afirmado.

> ⚠️ **Erro metodológico cometido e corrigido durante a execução:** a primeira tentativa mediu
> essa variância pelo **desvio entre folds da CV**, que deu apenas 1,1× e não mostrava nada. O
> desvio entre folds mistura a variância do modelo com o erro de estimativa de cada fold, e
> compara escalas de PR-AUC muito diferentes. A medida correta é a variação da **predição** sob
> reamostragem do treino, com o conjunto de avaliação fixo.

### Duas hipóteses testadas e REFUTADAS

**(a) Encoding por família de modelo — refutada.** A hipótese: o one-hot penalizaria as árvores
por espalhar cada feature em k dummies (gastando profundidade e diluindo o sorteio do
`max_features`). Medido nos 5 pares:

| Modelo | onehot → ordinal | Δ em dp |
|---|---|---|
| Árvore única | 0,3971 → 0,3898 | **−0,31** |
| Random Forest | 0,6437 → 0,6519 | +0,35 |
| RF regularizada | 0,6868 → 0,6886 | +0,07 |
| HistGradientBoosting | 0,6667 → 0,6658 | −0,04 |
| HGB regularizado | 0,6915 → 0,6919 | +0,02 |

Efeito **desprezível e sem sinal consistente** (muda de direção em 2 dos 5 pares). A explicação
é a **cardinalidade**: as 10 categóricas do Telco têm 2 ou 3 níveis, e com 2 níveis one-hot e
ordinal são literalmente a mesma coluna. O pedágio previsto existe, mas só se manifesta com
categóricas de cardinalidade alta — que este dataset não tem. **Hipótese boa em teoria,
irrelevante nestes dados.** O código dos dois encodings fica (`--encoding`), com custo zero.

**(b) Descalibração dos ensembles — refutada no mecanismo, confirmada no efeito.** A previsão
era que a RF comprimiria a probabilidade para o centro (por ser média de votos) e o boosting a
extremizaria, piorando o Brier. **Não aconteceu:** Brier de 0,1339 (LogReg), 0,1343 (RF) e
0,1320 (HGB) — praticamente idênticos, com desvios da distribuição de 0,249/0,251/0,255. Com
`min_samples_leaf=5` e 300 árvores, a floresta ainda produz votos variados o bastante para não
comprimir.
**Mas a conclusão prática sobrevive por outro caminho:** o limiar ótimo **de fato não transfere**
— ver abaixo.

### O limiar de operação não é transferível — medido em reais

| Modelo | Limiar ótimo próprio | Custo com o próprio | Custo se herdasse o 0,22 | Prejuízo |
|---|---|---|---|---|
| LogReg | 0,29 | R$ 31.750 | R$ 31.906 | +R$ 156 |
| **Random Forest reg.** | **0,27** | R$ 30.172 | R$ 31.262 | **+R$ 1.090 (3,6%)** |
| HistGradientBoosting reg. | 0,22 | R$ 29.626 | R$ 29.626 | +R$ 0 |

O limiar ótimo varia de **0,22 a 0,29** entre modelos igualmente bons. Herdar o corte do
baseline na Random Forest custaria **R$ 1.090 por ciclo** — sem nenhum sintoma visível: a
PR-AUC continuaria a mesma, e só o volume da fila de retenção mudaria. **O limiar é parâmetro
do par (modelo, custo), não do projeto.** Confirma a decisão da Etapa 0 de que a API devolve
probabilidade, não classe.

### `permutation_importance` × `feature_importances_` — o viés do MDI provado nos dados

| Posição | Permutação (validação) — RF | MDI (`feature_importances_`, treino) — RF | EDA (Etapa 1) |
|---|---|---|---|
| 1 | Tenure Months | Tenure Months | Contract (39,9 pp) |
| 2 | **Contract** | Total Charges | Tenure |
| 3 | Monthly Charges | Monthly Charges | Internet Service |
| 4 | Dependents | **Contract** | … |
| 7 | **Total Charges** | *(2º lugar)* | — |

**`Total Charges` é 2ª pelo MDI e 7ª pela permutação.** É exatamente o viés previsto: variável
contínua de altíssima cardinalidade oferece muitos pontos de corte candidatos, e o MDI premia a
frequência de uso em vez da contribuição real. Reforça o que já se sabia: `Total Charges`
correlaciona **0,9996** com `Tenure × Monthly` — é redundante, e a permutação enxerga isso
enquanto o MDI não. Simetricamente, **`Contract` sobe de 4ª para 2ª** na permutação, que é o que
a EDA mediu de forma independente.
→ **Decisão: a documentação usa `permutation_importance`.** O MDI fica registrado apenas como
demonstração do viés.

> 📌 **Achado lateral relevante para a Etapa 10 (monitoramento):** a permutação mostra que a
> LogReg depende de `Tenure Months` de forma esmagadora (+0,2514, quatro vezes a segunda
> colocada), enquanto a RF distribui o peso (+0,0623 no topo). Modelos com dependência
> concentrada são **muito mais sensíveis a drift de uma única feature** — se a distribuição de
> `tenure` mudar, a LogReg degrada primeiro. Isso é critério de escolha que não aparece na
> métrica, e entra na política de monitoramento.

### Decisão da Etapa 6

1. **Eliminados:** árvore única (não ranqueia — folhas puras produzem probabilidades degeneradas,
   PR-AUC 0,3971) e Random Forest (**dominada**: menor PR-AUC, maior custo e ~5× mais lenta que
   o boosting, sem compensação em nenhum critério).
2. **Seguem para a Etapa 7 (tuning):** **LogReg** e **HistGradientBoosting regularizado**, que
   representam as duas pontas do trade-off — interpretabilidade máxima com gap de 0,005 × melhor
   custo de operação (−R$ 2.124/ciclo, −6,7%).
3. **Nenhum modelo é declarado vencedor nesta etapa.** O empate é o resultado, e o desempate,
   se a Etapa 7 não separar os dois, **será por interpretabilidade e custo operacional, não por
   métrica** — com a tabela de odds ratios da LogReg sustentando o argumento.
4. O encoding ordinal permanece no código como opção medida, não como decisão adotada.

---

## 6. Decisão do modelo final

- **Modelo escolhido:**
- **Concorrentes e seus números:**
- **Por que este, além da métrica** (interpretabilidade, latência, robustez, custo):
- **O MLP superou os modelos clássicos?** ⬜ sim ⬜ não — *(um "não" documentado com honestidade
  vale mais que um resultado forçado)*

---

## 7. Limitações conhecidas

| Limitação | Impacto | Mitigação possível |
|---|---|---|
| | | |

---

## 8. Reprodutibilidade

- **Seeds fixadas:**
- **Versões de biblioteca:** (`requirements.txt` / `uv.lock`)
- **Versão/hash do dataset:**
- **Comando para reproduzir do zero:**

---

## Histórico de decisões (cronológico)

| Data | Etapa | Decisão | Por quê |
|---|---|---|---|
| 2026-08-07 | 0 | Métrica primária = **PR-AUC**, não recall | recall puro tem solução degenerada (marcar todos); PR-AUC avalia o ranking sem depender de limiar, e o artefato é um ranqueador |
| 2026-08-07 | 0 | Métrica de negócio = **recall@10% / @20%**, não um `k` absoluto | capacidade do time é dado de stakeholder que não temos; o percentil sobrevive a qualquer tamanho de equipe |
| 2026-08-07 | 0 | Custo FN/FP calculado como **valor esperado** (× taxa de conversão), não valor nominal | um FN só custa quando a campanha teria funcionado; ignorar isso inflaria a assimetria de 3:1 para 12:1 e distorceria a métrica |
| 2026-08-07 | 0 | Predição é **apoio à decisão**, com humano no meio | risco financeiro de ação automática + LGPD Art. 20 (direito a revisão de decisão automatizada) |
| 2026-08-07 | 0 | Split será **aleatório estratificado**, não temporal | imposição do dataset (retrato sem eixo temporal); declarado como limitação em vez de omitido |
| 2026-08-10 | 1 | Manter a **variante estendida** (33 col) em vez do CSV clássico (21 col) | a caça a leakage documentada é evidência de método; dataset já limpo não permite demonstrá-la |
| 2026-08-10 | 1 | `Churn Score` e `CLTV` **fora do treino**, com uso na avaliação | usá-las seria prever o modelo da IBM, não churn (destilação acidental) + procedência desconhecida. Viram benchmark competidor e critério de ordenação da fila |
| 2026-08-10 | 1 | Geográficas **fora das features**, dentro da auditoria | cardinalidade inviável (4,3 clientes/CEP) **e** proxy de renda/raça. `Lat`/`Long` ficam em backlog como experimento controlado |
| 2026-08-10 | 1 | `Total Charges` vazio → **imputar 0**, não mediana nem remoção | o vazio é medição verdadeira (sem ciclo de faturamento); mediana inventaria histórico; remoção transferiria o caso para produção |
| 2026-08-10 | 1 | Feature de "reajuste de preço" **rejeitada antes de construir** | testada: razão com mediana 1,000 e sd 0,051, churn por quartil sem padrão. O dataset não carrega histórico de reajuste |
| 2026-08-10 | 2 | Split em **três** partições (60/20/20), não duas | early stopping do MLP, gate do CI e seleção de algoritmos exigem validação fixa; reparticionar depois invalidaria a tabela mestra inteira |
| 2026-08-10 | 2 | Conjunto de **teste intocado** até o fim; toda seleção na validação | divergência deliberada do enunciado da Aula 08: decidir repetidamente olhando o teste o converte em validação |
| 2026-08-10 | 2 | `OneHotEncoder` **sem `drop`**, contra a recomendação usual de `drop_first` | com `drop`, categoria inédita vira o mesmo vetor da categoria dropada (`Gender="Other"` → `"Female"`), silenciosamente. A dummy trap é inofensiva sob L2; a colisão não é |
| 2026-08-10 | 2 | **Nenhum tratamento de desbalanceamento** | 26,5% não é severo, e SMOTE/`class_weight` descalibram a probabilidade, quebrando a ordenação por `P(churn) × CLTV`. `balanced` foi medido: mesmo custo, Brier 21% pior |
| 2026-08-10 | 2 | Numéricas forçadas a `float64`, inclusive `Tenure Months` | inteiro não representa nulo: campo faltando na API quebraria o schema enforcement do MLflow só em produção |
| 2026-08-10 | 2 | MLflow com backend **SQLite**, não file store | o MLflow 3 pôs `./mlruns` em modo de manutenção e o Model Registry (Etapa 9) nunca funcionou sobre arquivos. Migrar agora evita mexer nos experimentos depois |
| 2026-08-10 | 2 | Serialização via **skops** (padrão do MLflow 3), não pickle | skops não executa código arbitrário ao carregar — importa quando o artefato vem de um registry. Custo: declarar `skops_trusted_types=["numpy.dtype"]` |
| 2026-08-10 | 3 | **Baseline oficial = LogReg simples**, PR-AUC 0,6623 na validação | 2,5× o piso de 0,2654; gap de 0,033 sem overfitting relevante. É o número que toda etapa seguinte precisa bater |
| 2026-08-10 | 3 | `Churn Score` da IBM reclassificado de "benchmark" para **caso de leakage documentado** | nenhum não-churner acima de 80, nenhum churner abaixo de 65, zero exceções em 1.409 linhas: score calculado com o desfecho conhecido. PR-AUC 0,88 é inatingível por construção |
| 2026-08-10 | 3 | Reportar recall@k **como % do teto estrutural** (`k / prevalência`) | recall@10% jamais passa de 0,377 nesta base; o número cru faz um ranking de 76,7% do máximo parecer fraco |
| 2026-08-10 | 4 | Ganho de feature medido por **CV no treino**, não na validação | julgar 4 candidatas olhando a validação a gastaria, do mesmo modo que decidir no teste o gastaria. E a CV entrega o desvio entre folds, sem o qual não se distingue ganho de sorteio |
| 2026-08-10 | 4 | Ablação por **remoção**, não por adição isolada | adicionar sozinha superestima a contribuição de feature redundante; a pergunta certa é "agrega algo que as outras não dizem?" |
| 2026-08-10 | 4 | **Nenhuma das 4 features entra no modelo v1** | ganho de +0,0003 contra desvio de 0,0280 na LogReg, e **−0,0067 na Random Forest**. Duas delas eram combinações lineares de dummies já presentes — redundantes por construção, não por acaso |
| 2026-08-10 | 4 | Código das features **mantido**, ligável por `--features` | é evidência de método para a banca e permite reavaliação barata na Etapa 8 (MLP). Caminho de código coberto por 7 testes |
| 2026-08-10 | 4 | Ignorar o ganho de +0,0067 que aparece **na validação** | uma observação contra cinco da CV; o valor cabe 4× dentro do desvio entre folds. É o caso-teste do "isso é sinal ou variância do split?" |
| 2026-08-10 | 5 | Seleção por **ablação de feature original**, não por `SelectKBest`/`RFE` sobre as dummies | podar dummy não reduz custo de coleta: a coluna-mãe continua tendo de ser obtida, validada e monitorada. Os 3 métodos clássicos foram rodados e todos pioraram |
| 2026-08-10 | 5 | **19 → 13 features**, removendo 6 | performance equivalente (0,6912 × 0,6868 na CV) com o **menor desvio entre folds** (0,0163). A justificativa é custo operacional e menos superfície de drift, não métrica — e está escrita assim |
| 2026-08-10 | 5 | `Payment Method` removida apesar do sinal forte na EDA | 78% dos `Electronic check` são mês-a-mês: o sinal já está em `Contract`. Correlação espúria confirmada com tabela cruzada |
| 2026-08-10 | 5 | Atributos sensíveis **mantidos**, decisão adiada para a Etapa 10.5 | removê-los custa zero, mas a escolha depende do resultado da auditoria de fairness — decisão de governança não se toma sem o dado da governança |
| 2026-08-10 | 5 | Gate de leakage do seletor **medido**, não só afirmado | inflação de 0,0000 neste dataset (seleção estável com 4.225 amostras) × **+0,3766 sobre puro ruído** em regime `p >> n`. A regra continua valendo; agora o mecanismo está entendido |
| 2026-08-10 | 5 | Features órfãs da Etapa 4 **removidas do código** | a remoção de 6 colunas tirou o insumo de 3 das 4 derivadas. Registrado o encadeamento em vez de apagado, com teste que trava a cascata |
| 2026-08-11 | 6 | Protocolo da comparação = **CV estratificada repetida (5×3) no treino**, validação tocada só pelos 3 finalistas | um split único já deu a resposta errada na Etapa 4; e é o desvio entre folds que distingue ganho de sorteio |
| 2026-08-11 | 6 | **Nenhum vencedor declarado por métrica** — LogReg 0,6904, HGB 0,6919, RF 0,6886 | os três cabem em 0,0033 contra desvio de 0,0188–0,0236: a maior diferença vale 0,07 dp. Escolher por PR-AUC aqui seria ler ruído |
| 2026-08-11 | 6 | Incluir variantes **reguladas** (`min_samples_leaf=5`) dos ensembles na comparação | declarar vencedor sobre adversário sabidamente mal configurado (árvores até a pureza em 4.225 amostras) não é comparação, é armar o resultado |
| 2026-08-11 | 6 | 🔑 Registrar que **a regularização move mais o número que a família do modelo** | RF: 0,6519 → 0,6886 só com `min_samples_leaf=5` (+1,6 dp) — maior que qualquer diferença ENTRE algoritmos. Contraria a alegação de que RF dispensa ajuste |
| 2026-08-11 | 6 | Árvore única mantida no experimento **como controle**, não como candidata | é ela que mede a variância que justifica o ensemble: probabilidade oscila ±0,2462 sob reamostragem contra ±0,0438 da floresta (5,6×) |
| 2026-08-11 | 6 | Medir variância por **reamostragem da predição**, não pelo desvio entre folds | erro cometido e corrigido: o desvio entre folds mistura variância do modelo com erro de estimativa e deu 1,1×, escondendo o efeito real de 5,6× |
| 2026-08-11 | 6 | **Hipótese do encoding por família REFUTADA** — ordinal não ajuda árvores aqui | efeito ≤0,35 dp e muda de sinal em 2 dos 5 pares. Causa: as 10 categóricas têm 2–3 níveis, e com 2 níveis one-hot e ordinal são a mesma coluna. Código mantido, decisão não adotada |
| 2026-08-11 | 6 | **Hipótese da descalibração REFUTADA no mecanismo** — Brier idêntico (0,132–0,134) | com `min_samples_leaf=5` e 300 árvores a RF não comprime a probabilidade como previsto. A conclusão prática (limiar não transfere) sobrevive por outra via |
| 2026-08-11 | 6 | **Limiar re-derivado por modelo** (0,29 / 0,27 / 0,22), nunca herdado | herdar o 0,22 na RF custaria **+R$ 1.090 por ciclo** sem sintoma visível — a PR-AUC não muda, só o volume da fila. Limiar é parâmetro do par (modelo, custo) |
| 2026-08-11 | 6 | Documentação usa **`permutation_importance`**, não `feature_importances_` | o MDI põe `Total Charges` em 2º e a permutação em 7º: viés de cardinalidade confirmado nos próprios dados. A permutação concorda com a EDA sobre `Contract`, o MDI não |
| 2026-08-11 | 6 | **Random Forest eliminada; seguem LogReg e HistGradientBoosting** | a RF é dominada — menor PR-AUC, maior custo e ~5× mais lenta que o boosting, sem vantagem em nenhum critério |
| 2026-08-11 | 6 | Confirmado o contrafactual registrado na Etapa 3: **o sinal é essencialmente linear no logit** | a LogReg alcança o mesmo resultado com gap de 0,005 contra 0,083 do boosting. Os não-lineares têm liberdade funcional e não acham o que fazer com ela |

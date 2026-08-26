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

## 5c. Etapa 7 — tuning de hiperparâmetros

### ⚠️ PRÉ-REGISTRO — escrito ANTES de rodar (11/08/2026)

Esta subseção foi commitada **antes** da primeira execução de `src/tuning.py`, de propósito.
Previsão registrada e depois confirmada é evidência de método; previsão escrita depois do
resultado é narrativa. As quatro expectativas:

1. **O ganho do tuning será pequeno.** A Etapa 6 mediu três famílias dentro de 0,0033 contra
   desvio entre folds de 0,0188–0,0236 — o teto aparenta ser **dos dados**, não do modelo.
   Esperado: ganho **abaixo de 1 desvio**, ou seja, não distinguível de ruído.
2. **O `C` ótimo da LogReg sairá alto.** Com `n >> p` (4.225 amostras × 24 colunas após o
   one-hot) há amostra de sobra por parâmetro, então não há muito o que encolher. `C` muito
   **baixo** seria sinal de investigar, não de comemorar. ⚠️ `C = 1/λ` — **escala invertida**
   em relação ao `alpha` do `Ridge`/`Lasso`; montar a grade com a intuição de `alpha` inverteria
   o experimento inteiro.
3. **O `l1` não deve ganhar do `l2`.** L1 zera coeficientes, e a Etapa 5 já mediu que **podar
   features não melhora a métrica** neste dataset (19 → 13 sem custo, mas sem ganho). Se o L1
   vencer, o ganho virá de estabilidade, não de seleção.
4. **O HGB não deve abrir vantagem decisiva.** Se abrir, o desempate da Etapa 6 muda de critério;
   se não abrir, vale o argumento do Netflix Prize (item 43 do revisita): empate dentro do ruído
   ⇒ escolhe-se o mais simples e interpretável.

### 🚨 Disciplina de relato — o viés de seleção que esta etapa introduz (item 46)

> *"Se agora reportamos o erro médio de 5-fold obtido, estaremos ligeiramente otimistas, pois
> aquele erro médio foi usado na seleção"* — M02-A07, pág. 11–12.

O `max` sobre muitas medições ruidosas captura **ruído favorável**, e o viés **cresce com o
tamanho da grade**. É o mesmo mecanismo do gate medido no teste (§9.5) e de decidir feature na
validação (§3), em terceiro disfarce — e **não tem sintoma**: o gap treino–validação continua
bonito, porque não é o modelo que aprendeu mal, é o instrumento que foi gasto pelo uso.

Solução formal seria **nested CV**. A prática que a própria aula indica é **train/val/test**, que
o split 60/20/20 da Etapa 2 já entrega. Logo, o que falta aqui não é protocolo — é disciplina:

1. tunar por **CV no treino**;
2. **nunca** publicar o melhor score da grade como estimativa de generalização;
3. o número da documentação sai da **validação**; o **teste continua intocado**, para ser tocado
   uma vez só, no fim.

*Pergunta-âncora:* **"esse número foi usado para escolher alguma coisa?"** Se foi, ele não é
estimativa imparcial daquilo que ele escolheu.

### Estratégia de busca — escolhida pelo tamanho da grade (item 47)

| Finalista | Busca | Por quê |
|---|---|---|
| **LogReg** | `GridSearchCV` | `C` (6) × `penalty` (2) = **12 configs**, ambos os eixos sabidamente relevantes. Exaustivo custa 180 fits — barato, e garante o ótimo dentro da grade |
| **HGB** | `RandomizedSearchCV` | espaço de **5 eixos** com centenas de combinações, e com **suspeita de eixo inerte**. Bergstra & Bengio (2012) — bibliografia oficial da aula, pág. 25: num grid, um eixo que não importa desperdiça `k×` o orçamento medindo a mesma coisa; na busca aleatória cada amostra testa um valor **novo de todos**, então a resolução no eixo que importa é o total de amostras, não a raiz dele |

⚠️ **As duas grades não são montadas do mesmo jeito, e isso é a assimetria da M02-A04:**
`n_estimators` é **seguro na RF** (mais árvores não sobreajusta) e é **regularizador no
boosting** (mais árvores sobreajusta). No HGB o eixo entra como `max_iter` **com
`early_stopping` interno**, que é a forma correta de deixar a validação escolher onde parar.

### 🔑 A regra 1-SE aplicada DENTRO da grade (uso original de Hastie)

Em vez do `argmax`, o `refit` é um *callable* que escolhe **o candidato mais regularizado cujo
score esteja dentro de 1 erro padrão do melhor**. Motivo medido no hands-on da aula (300 seeds):
a 1-SE recupera a complexidade verdadeira em **98%** das amostras contra **66%** do `argmin` —
o pico da grade é, em boa medida, sorteio favorável.

Isso exige uma **ordem de complexidade declarada por família** (a regra foi formulada para um
eixo de complexidade *dentro* de uma família):
- **LogReg:** menor `C` = mais regularizado = mais simples. Desempate: `l1` antes de `l2`
  (zera coeficientes ⇒ modelo com menos parâmetros efetivos).
- **HGB:** menos capacidade primeiro — menor `max_leaf_nodes`, depois menor `learning_rate`,
  depois maior `min_samples_leaf`, depois maior `l2_regularization`.

⚠️ **Ressalva registrada:** com CV **repetida**, as `K` dobras **não são independentes** (as
repetições reusam os mesmos dados), então `dp/√K` **superestima** a precisão — é a correção de
**Nadeau–Bengio**, ausente do material da disciplina. Por isso o envelope 1-SE é calculado em
**três leituras da dispersão** (erro padrão, versão conservadora e desvio cheio): se as três
concordam, a conclusão não depende de qual foi escolhida, e essa robustez é o argumento.

---

### Resultado — CV no treino (58 configurações, protocolo idêntico ao da Etapa 6)

*Executada em 11/08/2026.* Reprodução: `make tuning` (37 segundos no total).

| Família | Busca | Configs | Pico da grade | Escolha 1-SE | Δ (custo da 1-SE) |
|---|---|---|---|---|---|
| **LogReg** | grid | 18 | 0,6904 ± 0,0236 (`C=1,0`, `l1_ratio=0,5`) | 0,6861 ± 0,0222 (`C=0,1`, L1 pura) | −0,0044 (−0,18 dp) |
| **HGB** | random | 40 | 0,7007 ± 0,0208 | 0,6973 ± 0,0218 (`max_leaf_nodes=2`) | −0,0034 (−0,16 dp) |

> 🔑 **O achado da LogReg é a ausência de achado, e ele é forte: o default JÁ ERA o pico da
> grade.** `C=1,0` com L2 pura dá **0,6904** — exatamente o número que a Etapa 6 mediu com o
> LBFGS, e o mesmo valor do melhor ponto entre as 18 configurações. Dezoito ajustes de
> regularização não encontraram **nada** acima do que a biblioteca já entregava. A expectativa
> pré-registrada era "`C` alto e ganho pequeno"; o resultado medido é **ganho exatamente zero**.
>
> Isso é a contraparte necessária do achado da Etapa 6 (*"a regularização move mais o número que a
> família"*): mover a regularização importou **onde ela estava ausente** — nos ensembles crescidos
> até a pureza —, e não importa num modelo que já vinha regularizado por default. As duas frases
> juntas descrevem o mesmo fenômeno, e só a segunda estava escrita.

**Sobre o eixo `penalty`:** ele não existe mais. O scikit-learn **1.8 depreciou `penalty`**
(remoção em 1.10) e unificou tudo em `l1_ratio` — `0,0` é L2 pura, `1,0` é L1 pura, e o meio é
elastic net. A grade ganhou um terceiro ponto de graça, e o `solver='saga'` foi usado nos três
para manter a função objetivo constante ao longo do eixo (o `liblinear` penalizaria também o
intercepto, alterando o experimento justamente na variável medida).

**Envelopes 1-SE, nas três leituras:**

| Família | `dp/√15` (material) | `dp/√5` (Nadeau–Bengio) | desvio cheio |
|---|---|---|---|
| LogReg | 0,6843 → **12 de 18** dentro | 0,6799 → 13 de 18 | 0,6668 → 15 de 18 |
| HGB | 0,6953 → **9 de 40** dentro | 0,6914 → 15 de 40 | 0,6799 → 29 de 40 |

As três leituras concordam no que importa: **a maioria da grade é indistinguível do pico**. Com
12 de 18 candidatos empatados na LogReg, o `argmax` estaria escolhendo entre doze modelos
equivalentes — e "escolher o melhor de doze empates" é a definição operacional de ler ruído.

### O aviso de borda produziu uma AÇÃO, não só um alerta

A primeira execução do HGB bateu no extremo de três eixos — `max_leaf_nodes=4` (mínimo),
`min_samples_leaf=160` (máximo) e `l2_regularization=0,0` — **todos na direção de menos
capacidade**. A grade foi estendida para onde o aviso apontava (`max_leaf_nodes` até **2**, o
toco de decisão; `min_samples_leaf` até 320), e o resultado foi que a 1-SE **desceu ainda mais**:
escolheu `max_leaf_nodes=2` e continua na borda.

Ou seja: o boosting, neste dataset, quer ser o mais fraco que a grade permitir. Não é sinal de
que faltou tunar — é a **terceira medição independente** de que o sinal do Telco é simples,
depois do gap de 0,005 da LogReg (Etapa 6) e do empate entre famílias. Um modelo-base com **duas
folhas** não representa interação nenhuma: é um aditivo de decisões univariadas, ou seja, um
parente próximo do modelo linear que já estava lá.

### 🎯 O teste pareado diz que o HGB ganha. A validação diz que não. Os dois estão certos.

Com os 15 scores por dobra agora preservados (item 45), o teste pareado é imediato — e o
pareamento não custou nada, porque o `random_state` fixo no `RepeatedStratifiedKFold` já fazia
todos os candidatos verem exatamente as mesmas partições:

| Medida | Valor |
|---|---|
| Δ médio (HGB − LogReg) por dobra | **+0,0113** |
| Dobras em que o HGB vence | **13 de 15** |
| t pareado | p < 0,001 |
| Wilcoxon (não supõe normalidade) | p < 0,001 |

E, na validação, **o HGB é o pior dos três** (0,6589 contra 0,6620 da LogReg tunada e 0,6646 do
campeão). Não há contradição: o teste pareado responde *"a diferença é consistente?"* — e é —,
não *"a diferença importa?"*. Ele remove a variância da partição, o que lhe dá poder para detectar
**+0,011 de PR-AUC**, uma diferença que a operação não distingue de zero (R$ 278 de custo por
ciclo, 0,9%). Somem-se as duas ressalvas: os p-valores são **anticonservadores** (Nadeau–Bengio),
e o número da CV **foi usado para escolher**.

> **Formulação para a defesa:** significância estatística e relevância prática são perguntas
> diferentes, e com 15 dobras pareadas a primeira é fácil de conseguir. A decisão continua sendo
> da 1-SE e da validação — o teste pareado entra como evidência corroborante, exatamente o papel
> que lhe foi atribuído **antes** de ele ser calculado.

### 🚨 O viés de seleção previsto no pré-registro foi MEDIDO — com grupo de controle

O campeão serve de controle perfeito: ele **não foi escolhido por grade nenhuma**, então a queda
dele da CV para a validação mede o que a troca de conjunto custa por si só.

| Modelo | Candidatos | CV (treino) | Validação | Queda | **Excedente sobre o controle** |
|---|---|---|---|---|---|
| **Campeão** (controle) | **0** | 0,6904 | 0,6646 | −0,0258 | — |
| LogReg 1-SE | 18 | 0,6861 | 0,6620 | −0,0240 | **+0,0018** |
| HGB 1-SE | 40 | 0,6973 | 0,6589 | −0,0384 | **−0,0126** |

Duas leituras, ambas previstas pela aula e nenhuma delas óbvia sem o controle:

1. **A maior parte da queda não é viés de seleção** — é diferença entre CV no treino e um holdout
   único, e aparece igual no modelo que nunca passou por busca nenhuma. Atribuir os 0,026 inteiros
   à seleção seria o erro simétrico de ignorá-la.
2. **O excedente cresce com o tamanho da grade, exatamente como a aula prevê.** Com 18 candidatos
   e a 1-SE protegendo, o excedente é **nulo** (+0,0018, ou seja, a LogReg tunada generalizou tão
   bem quanto o controle). Com 40 candidatos, o excedente é **−0,0126** — metade de uma dobra de
   desvio, e material o bastante para inverter o ranking entre os dois finalistas.

> É a razão de a disciplina de relato ter sido escrita **antes**: se o número reportado fosse o
> pico da grade do HGB (**0,7007**), a documentação anunciaria um ganho de +0,0088 sobre o
> campeão. O número honesto, na validação, é **0,6589** — ou seja, **−0,0057**. A diferença entre
> a versão otimista e a real é de **0,042 de PR-AUC**, e nenhum sintoma a denunciaria: o gap
> treino–validação continua bonito, porque não é o modelo que aprendeu mal, é o instrumento que
> foi gasto pelo uso.

### Validação — o número honesto (tocada uma vez, com os modelos já escolhidos)

| Modelo | PR-AUC | Brier | Limiar ótimo | Custo/ciclo | recall@10% |
|---|---|---|---|---|---|
| **Campeão — LogReg default, 13 features** | **0,6646** | 0,1339 | 0,29 | R$ 31.750 | 0,278 |
| LogReg 1-SE (`C=0,1`, L1) | 0,6620 | 0,1342 | 0,33 | **R$ 31.326** | 0,283 |
| HGB 1-SE | 0,6589 | 0,1351 | 0,32 | R$ 31.604 | 0,281 |

*(Piso da PR-AUC = prevalência = 0,2654. Teto estrutural do recall@10% = 0,377.)*

Os três cabem em **0,0057**, contra desvio entre folds de 0,022. E as métricas **discordam entre
si**: a LogReg tunada tem a **pior** PR-AUC das duas lineares e o **menor custo em reais**
(−R$ 424, −1,3%) — porque o limiar re-derivado a coloca num ponto de operação um pouco melhor.
Discordância dessa magnitude entre métricas é o que empate dentro do ruído produz; ler qualquer
uma delas como desempate seria escolher a métrica depois de ver o resultado.

### 🔑 A L1 eliminou duas colunas ORIGINAIS — e isso é a distinção da Etapa 5 outra vez

A configuração 1-SE zerou **8 de 28** coeficientes (29% das colunas pós-encoding). Mas zerar
dummy não é zerar feature: a coluna original só sai do custo de coleta, validação e monitoramento
se **todas** as suas dummies zerarem. Medido: **duas** saem inteiras.

| Coluna eliminada pela L1 | O que isso confirma |
|---|---|
| **`Total Charges`** | **terceira evidência independente** de que é redundante: correlaciona **0,9996** com `Tenure × Monthly` (EDA, Etapa 1), a `permutation_importance` a põe em **7º** contra o 2º do MDI (Etapa 6), e agora a L1 lhe atribui coeficiente **zero**. Três métodos que não compartilham mecanismo, mesma conclusão |
| **`Gender`** | insumo para a Etapa 10.5, **não** a decisão dela. Que a L1 não encontre sinal preditivo não responde à pergunta de fairness — a coluna sai do modelo por não predizer, e a decisão de governança precisa do dado da auditoria |

**Sondagem do contrato reduzido** (11 features, `Total Charges` e `Gender` fora): CV **0,6883**,
validação **0,6642** — indistinguível das 13. Confirma pelo terceiro caminho o que a Etapa 5 já
tinha medido: neste dataset, podar não custa e não ganha em métrica.

### Decisão da Etapa 7

1. **O campeão permanece: LogReg nos defaults, 13 features, PR-AUC 0,6646 na validação.**
   Nenhuma das 58 configurações o superou lá. O tuning é reportado como **ganho zero**, que era a
   previsão pré-registrada — e uma previsão confirmada é resultado, não etapa desperdiçada.
2. **A configuração 1-SE (`C=0,1`, L1) NÃO é adotada**, apesar de a regra apontá-la. Motivo
   escrito: na validação ela empata (−0,0026, 0,11 dp) e a simplicidade que ela compra é real mas
   pequena (2 colunas de 13), enquanto o custo é concreto — a margem do gate do CI cairia de
   0,0046 para **0,0020**, menos que a folga de 0,005 reservada para variação numérica entre
   plataformas. Trocar o modelo para ficar com menos folga do que a variação esperada é comprar
   um CI intermitente em troca de nada.
3. **O gate do CI NÃO sobe** (item 35 do revisita). Ele acompanha o campeão, e o campeão não
   mudou. Elevar o piso sem um modelo melhor seria transformar o gate em obstáculo arbitrário;
   mantê-lo em 0,66 com a justificativa escrita é o oposto de afrouxá-lo para ficar verde.
4. **`Total Charges` e `Gender` permanecem no contrato v1**, com a evidência registrada. A
   remoção passou a ter três confirmações independentes, mas não muda métrica nenhuma — e a de
   `Gender` é decisão da Etapa 10.5, que precisa dele disponível para auditar.
5. **A Etapa 8 recebe o HGB tunado como adversário do MLP**, e não a configuração de referência
   da Etapa 6: comparar a rede com um boosting mal ajustado repetiria o erro que o gate de justiça
   da Etapa 6 existe para impedir.

---

## 5d. Etapa 8 — MLP em PyTorch

### ⚠️ PRÉ-REGISTRO — escrito ANTES de rodar (11/08/2026)

Mesmo protocolo de honestidade da Etapa 7: as previsões abaixo estão no repositório **antes** da
primeira execução. Previsão confirmada é evidência de método; previsão escrita depois do resultado
é narrativa, e a diferença entre as duas some no texto final se ninguém marcar a data.

| # | Previsão | Por que | Como seria refutada |
|---|---|---|---|
| 1 | **O MLP não supera o campeão** (0,6646 na validação) nem o HGB tunado. O delta cabe dentro de 1 desvio entre folds (0,019–0,024) | o material da disciplina (M02-A04, pág. 23) afirma que ensembles de árvores são estado da arte em tabular, *"superando inclusive redes neurais profundas"*. Somam-se **três** medições próprias na mesma direção: gap 0,005 (LogReg) × 0,083 (boosting) na Etapa 6; a FE da Etapa 4 sem efeito; e a 1-SE do HGB descendo até o **toco de 2 folhas** na Etapa 7 | MLP acima do campeão por **mais de 1 dp** na CV **e** na validação. Isso derrubaria o achado *"o sinal é essencialmente linear no logit"* que hoje tem três confirmações |
| 2 | **O MLP de ZERO camadas ocultas reproduz a LogReg** (CV ~0,690, dentro de ±0,005) | não é analogia: com uma camada linear e `BCEWithLogitsLoss`, o objeto **é** uma regressão logística (M02-A05, pág. 12). Mesma perda, mesma família de funções, mesmo pré-processamento | se divergir mais que isso, o problema é a **minha** implementação (otimizador, épocas, escala), não um achado sobre os dados. Esta previsão é o **teste de sanidade do experimento inteiro** |
| 3 | **O desvio entre seeds será menor que o desvio entre folds** (0,0188–0,0236), mas não nulo | a superfície de erro é não-convexa e os pesos partem de inicialização aleatória — variância que nenhum modelo do repo tinha (LogReg é convexa; árvores têm `random_state`). Com 1 camada e um problema fácil, espera-se que os mínimos sejam parecidos | se o desvio entre seeds for **da mesma ordem** do desvio entre folds, então nenhum número de seed única sustenta conclusão nesta etapa — e o protocolo da Etapa 6 teria de ganhar a dimensão seed (item 36) |
| 4 | **A arquitetura escolhida pela 1-SE será pequena** (1 camada, ≤16 neurônios) e o `weight_decay` não será o menor da grade | 4.225 amostras de treino contra 28 colunas: uma camada de 32 já são ~950 pesos, **4 amostras por parâmetro**. O teorema da aproximação universal garante que a capacidade *existe*, e é silencioso sobre ela ser *encontrável* e *necessária* — com o sinal medidamente simples, `hidden_layer_sizes` é eixo de **regularização** | arquitetura grande vencendo por mais de 1 dp indicaria interação que nem a árvore nem a FE encontraram |
| 5 | **As 4 features da Etapa 4 continuam sem ajudar** (item 16) | elas não criaram informação — duas eram combinação linear de dummies já presentes. E o argumento "a rede aprende interação sozinha" corta nos dois sentidos: se a interação valesse algo, o MLP a acharia sem a feature | ganho > 1 dp com `--features` ligadas seria a primeira evidência a favor delas em três medições |
| 6 | **Brier do MLP na mesma faixa dos demais** (0,133–0,140) | os finalistas da Etapa 7 ficaram em 0,1339–0,1351 e PR-AUC e Brier andaram juntos. Risco conhecido: o early stopping por PR-AUC otimiza **ordenação**, e nada garante calibração | Brier acima de 0,14 com PR-AUC igual seria o caso do *gate de um eixo só* (item 41) acontecendo de verdade — e é exatamente por isso que o segundo eixo entra **nesta** etapa, antes de o MLP ser avaliado |

### Decisões de protocolo tomadas antes da execução

1. **O MLP entra como estimador do MESMO `Pipeline`** (`escalonar=True`, `encoding="onehot"`, 13
   features), sob a **mesma CV estratificada repetida 5×3** com a **mesma seed** das Etapas 6 e 7.
   Não é conveniência de código: é a condição para o número ser comparável com a tabela existente
   e para o teste pareado nas mesmas 15 dobras vir de graça. Um MLP avaliado sob outro protocolo
   produziria uma linha que não conversa com nenhuma das anteriores.
2. **Isto torna a Etapa 8 um experimento controlado, e é o argumento mais forte da etapa.** Entre a
   LogReg e o MLP não muda a perda (log-verossimilhança da logística **é** a entropia cruzada), nem
   a forma da saída, nem o encoding, nem o pré-processamento — muda **uma variável só: a
   profundidade**. Logo o delta LogReg → MLP mede, isolada, quanta não-linearidade existe no Telco.
   RF e HGB trocam a família inteira de uma vez, e é por isso que o empate da Etapa 6 é ambíguo
   sobre *o que* empatou. O MLP de profundidade 0 entra na grade como **controle desse controle**.
3. **Early stopping por PR-AUC numa validação interna estratificada, extraída do treino** (15%),
   nunca na validação da Etapa 2. Duas razões: (a) `MLPClassifier(early_stopping=True)` pontua com
   `accuracy_score` no fonte e **não expõe `scoring`** — seria *otimizar a métrica errada dentro do
   laço de treino*, sem log nem warning, num problema de prevalência 26,5% e custo 3:1; (b) parar o
   treino olhando a validação faria do número de épocas mais um parâmetro escolhido nela, que é
   precisamente o viés que a Etapa 7 mediu com grupo de controle. Custo aceito: cada fit vê 15%
   menos dados que os demais candidatos — desvantagem **contra** o MLP, declarada.
4. **Os pesos restaurados são os da melhor época, não os da última.** Parar por paciência e ficar
   com o estado final entregaria um modelo pior que o que o próprio critério de parada elegeu.
5. **`random_state` é fator do experimento** (item 36): o finalista roda com **5 seeds**, e o desvio
   entre elas é reportado ao lado do desvio entre folds. Comparar uma média (LogReg, determinística)
   com um sorteio (MLP de seed única) seria comparar réguas diferentes.
6. **CPU, `torch.set_num_threads(1)`.** Não é limitação: com 4.225×28 o dado não paga a transferência
   para a GPU, e single-thread remove a não-determinismo da ordem de redução — reprodutibilidade é
   requisito registrado, não preferência.
7. **O adversário é o HGB tunado da Etapa 7** (`max_leaf_nodes=2`, `learning_rate≈0,120`,
   `min_samples_leaf=80`, `l2=1,0`), não a configuração de referência da Etapa 6. Comparar a rede
   com um boosting mal ajustado repetiria o erro que o gate de justiça da Etapa 6 existe para impedir.

### Resultado — CV no treino (15 arquiteturas, protocolo idêntico ao das Etapas 6 e 7)

`hidden` × `weight_decay`, 15 dobras por configuração, 42 segundos no total.

| `hidden` | `weight_decay` | PR-AUC | ±dp | treino | gap | |
|---|---|---|---|---|---|---|
| `(32,)` | 0,010 | **0,6858** | 0,0211 | 0,6924 | 0,0066 | ← pico |
| `(16,)` | 0,001 | 0,6857 | 0,0230 | 0,6959 | 0,0102 | |
| `(16,16)` | 0,010 | 0,6857 | 0,0221 | 0,6914 | 0,0057 | |
| `(8,)` | 0,001 | 0,6846 | 0,0255 | 0,6944 | 0,0098 | |
| `()` | 0,000 | 0,6837 | 0,0217 | 0,6877 | 0,0040 | ← controle |
| `()` | 0,010 | 0,6816 | 0,0236 | 0,6840 | 0,0024 | ← **escolha 1-SE** |
| `(16,16)` | 0,000 | 0,6724 | 0,0306 | 0,6834 | 0,0110 | pior da grade |

**A grade inteira cabe em 0,0134**, contra desvio entre folds de 0,021–0,031. **14 das 15
configurações entram no envelope 1-SE** nas duas leituras estreitas, e **as 15** no envelope do
desvio cheio. Traduzindo: arquitetura nenhuma se distingue de outra neste dataset.

### 🎯 Achado nº 1 — a regra 1-SE, aplicada DENTRO da família das redes, escolheu profundidade zero

A 1-SE elegeu `hidden=()` — que não é uma rede: é a regressão logística que está no repositório
desde a Etapa 3. Não é falha da regra nem do código; é a regra funcionando e dizendo que **nenhuma
quantidade de profundidade se paga aqui**.

É a **quarta medição independente** da mesma conclusão, e a mais forte, porque desta vez quem
falhou foi a família de funções mais flexível de todas:

| # | Etapa | Medição | O que disse |
|---|---|---|---|
| 1 | 4 | FE com razões e binning: +0,0003 contra ruído de 0,0280 | não há não-linearidade a entregar de mão beijada |
| 2 | 6 | gap treino→CV: LogReg 0,005 × boosting 0,083 | os não-lineares têm liberdade funcional e não acham o que fazer com ela |
| 3 | 7 | a 1-SE do HGB desce ao **toco de 2 folhas** | o modelo-base quer ser o mais fraco possível: o sinal é aditivo |
| 4 | **8** | a 1-SE do MLP desce à **profundidade 0** | com capacidade universal disponível, a escolha ótima é não usá-la |

> **Consequência para o relatório e para o vídeo:** *"o sinal do Telco é essencialmente linear no
> logit"* deixou de ser interpretação de um resultado e passou a ser previsão confirmada por quatro
> instrumentos que não compartilham mecanismo — engenharia de features, decomposição viés-variância,
> busca em boosting e busca em rede neural.

**Consequência prática:** o entregável e o resultado metodológico divergem. A fase exige uma rede,
e um MLP sem camada oculta é o baseline com outro nome. Por isso os **dois** seguem avaliados: a
escolha da 1-SE (o achado) e a melhor configuração **com** camada oculta, `(32,)` com
`weight_decay=0,01` (o entregável).

### 🚨 Achado nº 2 — o controle de profundidade zero NÃO reproduziu a LogReg, e a decomposição é o resultado

A previsão nº 2 dizia ±0,005. Medido: **−0,0067**. Fora da previsão, e a tentação seria escolher
uma explicação. Mas **duas** coisas diferem entre a LogReg do sklearn e o `MLPTorch(hidden=())`,
não uma — e um terceiro modelo separa as duas:

| | Modelo | CV | Leitura |
|---|---|---|---|
| (a) | LogReg sklearn, **100%** do fold | 0,6904 | a referência das Etapas 6 e 7 |
| (b) | LogReg sklearn, **mesmos 85%** | 0,6890 | **−0,0014 = o preço do early stopping em dados** |
| (c) | `MLPTorch hidden=()`, mesmos 85% | 0,6837 | **−0,0053 = o efeito do otimizador** |

O early stopping **não é grátis**: ele reserva 15% do treino para uma validação interna que a
LogReg não precisa ter. Isso é uma desvantagem **estrutural** do MLP neste desenho, e está declarada
em vez de embutida no resultado. O restante (−0,0053, ainda dentro do desvio entre folds) é
Adam com parada por PR-AUC contra LBFGS rodando até convergir com L2 — otimizadores diferentes
sobre a **mesma** perda e a **mesma** família de funções.

> A lição é de método, não de MLP: **quando dois fatores mudam juntos, dois números não decidem
> nada.** O terceiro modelo custou dez linhas e transformou uma previsão errada num achado com
> mecanismo identificado.

### 🔑 Achado nº 3 — a variância de seed existe, é minúscula, e o modelo CONVEXO variou mais

O item 36 perguntava se o protocolo da Etapa 6 precisaria da dimensão seed. Medido, com o mesmo
protocolo, cinco seeds cada:

| Regime | Configuração | PR-AUC (média entre seeds) | desvio entre **seeds** | × desvio entre **folds** (0,0215) |
|---|---|---|---|---|
| não-convexo | `(32,)`, wd 0,01 | 0,6858 | **0,0008** | **0,04×** |
| convexo | `()`, wd 0,01 | 0,6760 | 0,0035 | 0,13× |

**Resposta ao item 36: não.** A variância de inicialização é **uma ordem de grandeza menor** que a
variância de partição — mas isso é uma *medição*, não era a intuição: o hands-on do XOR falhava em
**14%** das inicializações. A diferença é o problema: numa superfície com mínimos locais ruins, a
seed decide; num problema quase linearmente separável no logit, todas as inicializações convergem
para soluções equivalentes. **A conclusão do XOR não transferia, e só se sabe qual dos dois casos é
o nosso medindo.**

⚠️ **E o resultado contraintuitivo é o mais instrutivo:** a rede **com** camada oculta variou
**menos** que a configuração sem camada nenhuma. A explicação é que "variância de seed" agrega
**três** fontes — inicialização dos pesos, ordem dos minibatches e o split da validação interna — e
sem camada oculta o problema volta a ser **convexo**, então não sobra mínimo local: o que resta é
justamente o split interno e o ponto de parada. Chamar os 0,0035 de "variância de inicialização"
seria nomear errado o que foi medido.

Na mesma unidade da Etapa 6 (desvio da probabilidade prevista por cliente): **±0,0094** entre seeds,
contra ±0,0438 da floresta e ±0,2462 da árvore única sob reamostragem do treino.

### Achado nº 4 — a profundidade não ganha nem *consistentemente*

Teste pareado nas mesmas 15 dobras, rede `(32,)` − LogReg: **Δ −0,0046**, a rede vence em **5 de
15** dobras, t pareado **p = 0,064**, Wilcoxon **p = 0,095**.

O contraste com a Etapa 7 é o que dá o significado. Lá o HGB vencia em **13/15 com p < 0,001** e
perdia na validação — *diferença consistente que não importa*. Aqui não há nem consistência: a
profundidade produz uma diferença que o teste pareado, com todo o poder que o pareamento lhe dá,
não distingue de zero. **É a medição direta da não-linearidade do Telco, e ela é indistinguível de
zero.**

### Validação — o número honesto (tocada uma vez, com as arquiteturas já escolhidas)

| Modelo | PR-AUC | ±dp (seeds) | Brier | limiar | custo/ciclo | recall@10% |
|---|---|---|---|---|---|---|
| **CAMPEÃO** (LogReg, Etapa 3) | **0,6646** | — | **0,1339** | 0,29 | R$ 31.750 | 0,278 |
| MLP `(32,)`, wd 0,01 | 0,6615 | 0,0040 | 0,1344 | 0,29–0,34 | **R$ 31.271** | 0,278 |
| MLP 1-SE `()`, wd 0,01 | 0,6600 | 0,0026 | 0,1354 | 0,27–0,35 | R$ 31.499 | 0,278 |
| HGB tunado (Etapa 7) | 0,6589 | — | 0,1351 | 0,32 | R$ 31.604 | 0,281 |

**961 parâmetros para 4.225 amostras de treino: 4,4 amostras por parâmetro.** O número que explica
por que os dois eixos da grade eram de regularização, e não de capacidade.

### 🚨 Achado nº 5 — PR-AUC e custo em reais apontaram para lados OPOSTOS

A rede fica **0,0031 abaixo** do campeão em PR-AUC e **R$ 479 abaixo** em custo por ciclo (−1,5%).
Não é ruído: a dispersão do custo **entre as próprias seeds** da rede é R$ 203, menor que a
diferença. É o cenário que a M02-A06 descreve (métrica agregada × regime operacional) acontecendo
no projeto, **e na direção inversa da esperada** — o modelo pior na integral é melhor no ponto onde
a operação vive.

**A decisão segue a hierarquia declarada na Etapa 0, não a métrica que favorece o resultado:**

| nível | métrica | quem ganha |
|---|---|---|
| 1 · seleção | PR-AUC | **campeão** (+0,0031, que é 0,15 dp — dentro do ruído) |
| 2 · operação | recall@10% | **empate exato** (0,278 × 0,278) |
| 3 · decisão | custo em R$ no limiar ótimo | rede (−R$ 479, −1,5%) |

Escolher agora o nível 3 porque ele favorece a rede seria trocar de régua depois de ver o
resultado — o mesmo pecado do `argmin` que a regra 1-SE existe para evitar. Duas ressalvas
reforçam: o custo mínimo é obtido num **limiar escolhido na própria validação** (otimista por
construção, para os dois modelos), e o nível 2 — o número que o gerente de retenção lê — empata na
terceira casa. **Registrado como pendência com número medido**, não descartado: se a Etapa 9 ou 10
mostrar que a decisão de negócio é custo, a rede volta com o número pronto.

### Decisão da Etapa 8

1. **O campeão permanece: LogReg nos defaults, 13 features, PR-AUC 0,6646 na validação.** O MLP foi
   implementado, avaliado sob protocolo idêntico e **não superou** — nem a rede `(32,)` (−0,0031)
   nem a configuração que a própria regra 1-SE elegeu (−0,0046). Previsão nº 1 **confirmada**.
2. **A rede entregue como artefato da fase é a `(32,)` com `weight_decay=0,01`**, treinada com
   `BCEWithLogitsLoss` + `Adam(1e-3)` e early stopping por PR-AUC. Ela é o entregável exigido; o
   modelo em produção é o campeão, e a documentação diz as duas coisas com a mesma clareza.
3. **O gate do CI não sobe** (item 35): ele acompanha o campeão, e o campeão não mudou. Terceira
   etapa seguida em que o piso 0,66 se mantém por ausência de modelo melhor, não por inércia.
4. **O gate ganhou o SEGUNDO EIXO** (item 41): `Brier ≤ 0,14` ao lado de `PR-AUC ≥ 0,66`, com o
   novo eixo **verificado reprovando** (teto a 0,10 → `exit 1`). Entrou agora justamente porque a
   calibração do MLP era desconhecida quando o eixo foi proposto — e um gate de eixo único aprovaria
   um modelo que ordena igual e calibra pior, com o CI verde e o custo subindo.
5. **Item 16 fechado: as 4 features da Etapa 4 continuam sem efeito** também no MLP (+0,0006, 0,03
   dp). Terceira família de modelos a rejeitá-las. O argumento *"a rede aprende interação sozinha"*
   corta nos dois sentidos: se a interação valesse algo, a rede a acharia **sem** a feature pronta.
6. **O MLP não é dependência da API.** Como o modelo servido é a LogReg, a Etapa 9 não precisa de
   `torch` em runtime — o que também reabre a discussão do item 33 (o CI instala ~800 MB de torch
   em dois jobs para um modelo que não vai a produção).

### Placar do pré-registro: 5 confirmadas, 1 refutada com mecanismo identificado

| # | Previsão | Resultado |
|---|---|---|
| 1 | o MLP não supera o campeão | ✅ −0,0031 na validação; delta dentro de 1 dp |
| 2 | profundidade 0 reproduz a LogReg em ±0,005 | ❌ **−0,0067** — refutada, e decomposta em −0,0014 (orçamento de dados) + −0,0053 (otimizador) |
| 3 | desvio entre seeds < desvio entre folds | ✅ 0,04× (rede) e 0,13× (convexo) — confirmada com folga |
| 4 | arquitetura pequena, `weight_decay` não mínimo | ✅ a 1-SE foi a **profundidade zero** com o maior `weight_decay` da grade |
| 5 | as 4 features seguem inúteis | ✅ +0,0006 (0,03 dp) |
| 6 | Brier em 0,133–0,140 | ✅ 0,1344 (rede) e 0,1354 (1-SE) |

A refutada é a mais útil das seis: ela é a única que produziu um experimento novo.

---

## 5d-bis. Etapa 8-bis — o `MLPClassifier` do scikit-learn (25/08/2026)

*Módulo: `src/mlp_sklearn.py` · testes: `tests/test_mlp_sklearn.py` · run MLflow
`mlp-sklearn-etapa8bis`.*

**Por que este experimento existe.** O enunciado da fase nomeia a implementação:
*"treinar uma Rede Neural simples utilizando o **MLPClassifier** do Scikit-Learn"*. A Etapa 8 foi
escrita em PyTorch, e o motivo estava registrado — o `early_stopping=True` do `MLPClassifier`
pontua a validação interna com `accuracy_score`, está no fonte e a classe **não expõe `scoring`**.
Era uma afirmação sobre código de terceiro, defendida por leitura. Este módulo a **mede**, e de
quebra cumpre o requisito com um número ao lado dos demais candidatos em vez de com uma
justificativa.

**Protocolo idêntico**, e isso não é retórica: mesmas 13 features, mesmo `construir_pipeline`
(one-hot + padronização — verificado por teste que compara os transformadores um a um com os da
LogReg), mesma CV estratificada repetida 5×3, mesma seed, `scoring="average_precision"`, mesma
grade de arquitetura da Etapa 8 e mesmas 5 inicializações. `alpha` (a L2 do sklearn) espelha o
`weight_decay` do Adam, com **1e-4 acrescentado por ser o default da biblioteca** — isto é, a
configuração que o experimento teria se ninguém tivesse escolhido nada.

### A grade — 20 configurações, 15 dobras cada

| hidden | melhor PR-AUC (CV) | ±dp | gap treino−CV |
|---|---|---|---|
| `(8,)` | **0,6874** (α=0) | 0,0231 | **0,0241** |
| `()` | 0,6845 | 0,0231 | **0,0033** |
| `(16,)` | 0,6800 | 0,0203 | 0,0499 |
| `(32,)` | 0,6696 | 0,0164 | 0,0885 |
| `(16, 16)` | 0,6641 | 0,0227 | 0,0879 |
| *LogReg (Etapa 6, mesma CV)* | *0,6904* | — | *0,0033* |

🔑 **A coluna que vale mais é a do gap, e ela é monotônica na capacidade: 0,0033 → 0,0241 → 0,0499
→ 0,0885.** Cada neurônio acrescentado foi inteiramente para dentro do treino. Não é que a rede
tenha "falhado em aprender": ela aprendeu mais, e nada do que aprendeu a mais generalizou. É a
mesma conclusão das Etapas 4, 6, 7 e 8 chegando por um quinto caminho — e o único em que o
overfitting aparece **graduado**, não como um sim/não.

🎯 **A regra 1-SE elegeu `hidden=()` outra vez.** Exatamente como na Etapa 8, com outra biblioteca,
outro otimizador e outra inicialização: dentro do envelope de 1 erro padrão (0,6814), a
configuração mais simples é a que não tem camada oculta nenhuma — a regressão logística. Duas
implementações independentes desistiram da profundidade pelo mesmo critério.

### O controle de profundidade zero — e uma réplica que não estava prevista

| | PR-AUC (CV, 15 dobras) |
|---|---|
| LogReg do sklearn (LBFGS) | 0,6904 |
| `MLPClassifier` `hidden=()`, α=0 (Adam) | 0,6845 |
| **diferença** | **−0,0059** |

🔑 **Isto é uma réplica do achado nº 2 da Etapa 8, com implementação diferente.** Lá, o controle de
profundidade zero em PyTorch ficou −0,0067 abaixo do baseline, e o terceiro modelo decompôs a
diferença em **−0,0014 de orçamento de dados + −0,0053 de otimizador**. Aqui não há desconto de
orçamento (o early stopping está desligado), e o que sobra é o otimizador: **−0,0059**, contra os
−0,0053 previstos pela decomposição. *Duas implementações de Adam, escritas por equipes diferentes,
custam a mesma coisa em relação ao LBFGS.* A decomposição da Etapa 8 era uma conta feita com um
modelo construído para o fim; esta é uma medição independente que chega a 0,0006 dela.

### 🚨 O custo do `early_stopping` que pontua por acurácia — medido, enfim

A configuração é a mesma (`(8,)`, α=0), as seeds são as mesmas, os dados são os mesmos. Muda uma
palavra-chave:

| | PR-AUC (CV) | PR-AUC (validação) | Brier | custo/ciclo | dispersão entre seeds |
|---|---|---|---|---|---|
| `early_stopping=False` | **0,6846** ± 0,0042 | **0,6655** ± 0,0054 | 0,1330 | **R$ 30.793** | ±0,0054 |
| `early_stopping=True` | 0,6542 ± 0,0127 | 0,6378 ± 0,0191 | 0,1398 | R$ 32.631 | ±0,0191 |
| **efeito** | **−0,0304** | **−0,0277** | +0,0068 | **+R$ 1.838** | **3,5×** |

**−0,0304 de PR-AUC é mais de um desvio entre folds (0,0231).** E o número que decidiu a parada
está gravado no próprio objeto: `best_validation_score_ = 0,8175` — que **é acurácia**, num
problema cujo piso de acurácia é 73,46%. Duas réguas no mesmo treino: uma escolhe a época, a outra
avalia o resultado.

🔑 **O que a tabela acrescenta à leitura do fonte não é o sinal, é a ordem de grandeza — e a
terceira coluna.** A dispersão entre seeds **triplica** quando o early stopping entra. Faz sentido
no mecanismo: o critério de parada passa a depender de um split interno aleatório *e* de uma
métrica de degrau (acurácia muda em saltos, e várias épocas empatam), então a época escolhida vira
sorteio. *O default não só piora a média: ele torna o resultado menos reproduzível, que é o
oposto do que se pede a um critério de parada.*

⚠️ Registrar o que **não** se mediu: nada disso diz que `early_stopping` seja ruim em geral. Diz
que **parar por uma métrica que não é a do projeto** custa isto aqui, neste problema. A versão em
PyTorch para por PR-AUC e restaura os pesos da melhor época — foi para poder fazer as duas coisas
que ela foi escrita.

### 🚨 O resultado desconfortável: na validação, a rede do sklearn ficou marginalmente acima do campeão

| | PR-AUC | Brier | custo/ciclo | R@10% |
|---|---|---|---|---|
| **Campeão (LogReg, 13 features)** | 0,6646 | 0,1339 | R$ 31.750 | 0,278 |
| `MLPClassifier (8,)`, média de 5 seeds | **0,6655** ± 0,0054 | 0,1330 | R$ 30.793 | 0,283 |
| diferença | **+0,0009** | −0,0009 | −R$ 957 | +0,005 |

Quatro métricas apontando para o mesmo lado é o tipo de resultado que merece ser escrito **antes**
de ser explicado. **A decisão é manter o campeão**, e os motivos são numéricos, não de preferência:

1. 🔑 **Na métrica de seleção, o campeão ganha.** A hierarquia da Etapa 0 diz onde cada número
   decide: a **CV no treino** seleciona (15 dobras, 4.225 amostras), a **validação** reporta. Na CV
   é **0,6904 × 0,6874** — a favor da LogReg. Trocar a régua depois de ver a validação é o mesmo
   pecado que a regra 1-SE existe para evitar, um andar acima.
2. 📏 **A vantagem é menor que o sorteio que a produziu.** O desvio entre as 5 seeds é **±0,0054**,
   seis vezes a diferença de 0,0009 — e a seed 2024 sozinha deu **0,6586**, isto é, 0,0060 **abaixo**
   do campeão. *Qual dos dois "ganha" depende de qual inicialização for promovida*, e essa é a
   definição de ruído.
3. 📏 **O IC95 do conjunto de teste tem 0,1056 de largura** (Etapa 11) — **117 vezes** a diferença.
   Nenhum conjunto de teste deste tamanho poderia distinguir os dois modelos.
4. ⚖️ **O preço de trocar é alto e o ganho não é mensurável.** Promover exigiria re-tocar o teste
   (lido **uma única vez**, com o registro versionado), refazer a auditoria de fairness, regravar o
   baseline de drift dentro do artefato e reescrever a caracterização — tudo para servir um modelo
   **não determinístico** (5 seeds = 5 modelos) e sem os 13 coeficientes que a LGPD Art. 20 torna
   úteis.

🎯 **E é isto que o teste de caracterização já impedia sem ter sido projetado para isso:**
`|PR-AUC − ref| ≤ 1e-4` reprova qualquer troca de modelo, **inclusive uma que melhore o número**.
A trava funcionou na primeira vez em que houve um candidato melhor — e o candidato apareceu de um
experimento que existia para cumprir um requisito de enunciado.

### Convergência

**0 avisos de `ConvergenceWarning` em 300 ajustes** da grade, com `max_iter=800`. O default do
sklearn é 200, e com ele parte das configurações pararia por **orçamento**, não por convergência —
uma tabela em que um candidato foi interrompido no meio do treino mede o corte, não o modelo. O
contador está no relatório do módulo de propósito: *um warning que ninguém lê é um warning que não
existe*.

### Decisão registrada

| Campo | Valor |
|---|---|
| **Requisito atendido** | rede neural com `MLPClassifier` do scikit-learn, sob protocolo idêntico ao dos demais candidatos |
| **Melhor configuração** | `hidden_layer_sizes=(8,)`, `alpha=0`, `early_stopping=False`, `max_iter=800` |
| **Escolha da 1-SE** | `hidden_layer_sizes=()` — profundidade zero, pela segunda vez |
| **Campeão** | **inalterado** (LogReg, 13 features) — motivos 1 a 4 acima |
| **Toca o teste?** | **não.** Treino (CV) e validação apenas |
| **Promove?** | **não.** `make promover` continua sendo o único caminho, e o gate continua decidindo |

---

## 5e. Etapa 9 — pipeline serializado e API

### 9c · A promoção do artefato (17/08/2026)

Até esta data o repositório **não tinha um modelo promovido**: `models/` continha apenas
`.gitkeep`, `grep joblib.dump src/` não retornava nada, e todas as medições das Etapas 6 a 8 — e
das quatro aulas do Módulo 04 — foram feitas contra um pipeline **treinado na hora**. Isso é
suficiente para comparar modelos e é insuficiente para servir um: um serviço que retreina no
startup serve *um* modelo com o mesmo código, não *o* modelo que foi avaliado.

**O que foi construído:** `src/artefato.py` (empacotar/carregar/verificar), `src/promover.py`
(o executável, `make promover`), `tests/test_artefato.py` (5 testes) e o alvo `make artefato`
de inspeção.

**Artefato promovido:** `models/campeao.joblib` · sha256 `5f2b62dcd5ab7ef2…` · LogReg v1.0.0 ·
13 features · limiar 0,29 · PR-AUC 0,6646 · Brier 0,1339 — os mesmos números que o gate mede e
que o teste de caracterização (item 85) fixa.

**As sete decisões, com o modo de falha que cada uma fecha:**

| # | Decisão | Contra o quê |
|---|---|---|
| 1 | **`gate.treinar_campeao()` extraída**: promoção e gate treinam pelo mesmo caminho | duas definições do mesmo modelo mantidas iguais pela memória de quem edita. O gate **retreina** em vez de conferir o artefato ⇒ a divergência não quebraria nada: o CI aprovaria um modelo e a API serviria outro, os dois verdes |
| 2 | **UM arquivo**: pipeline e metadados serializados juntos | modelo e metadados em arquivos separados, sem verificação de que combinam, produzem **rótulo errado com HTTP 200** — o modo de falha mais caro de uma API de ML, porque não falha |
| 3 | **`dict` puro no disco, não uma dataclass nossa** | serializar uma classe de `src.artefato` obrigaria o ambiente de inferência a ter `src` importável — o mesmo acoplamento do `FunctionTransformer` (item 17), destruído por comodidade de tipagem. O campeão carrega sem `src` no `sys.path`, e há teste que impede que ele saia desse lado |
| 4 | **Versão do sklearn gravada à mão e verificada na carga** (item 96) | `BaseEstimator.__setstate__` faz `state.pop("_sklearn_version")`: o objeto carregado não sabe mais com que versão foi treinado. E versão divergente **não impede a carga** — `InconsistentVersionWarning` é subclasse de `UserWarning`, o modelo prediz com número diferente (0,6601 × 0,6646 medidos) e ninguém lê o aviso |
| 5 | **Falhar na CARGA** (`ArtefatoIncompativel`), não avisar no log | subir degradado é escolher a falha silenciosa: a API responderia 200 com a probabilidade de outro modelo. Falhar na inicialização troca um erro invisível de predição por um erro visível de deploy — é o servidor nº 8 da Knight Capital |
| 6 | **Limiar de operação (0,29) viaja DENTRO do artefato** | um `0.29` literal na API é a segunda cópia que ninguém atualiza junto — e a que fica para trás não dá erro, só corta a fila no lugar errado. O limiar é propriedade do **modelo servido**: muda quando a distribuição de probabilidades muda |
| 7 | **Features declaradas a partir de `pipe.feature_names_in_`**, não de `config.FEATURES` | o config acompanha o **código**, o artefato acompanha o **modelo servido**; os dois só coincidem enquanto ninguém promove um modelo treinado com outro config. É a fonte de verdade do contrato Pydantic da 9d (item 94) |

**Nada é promovido sem passar no gate.** `promover.main()` aplica `gate.aprovado()` nos dois eixos
antes de gravar; reprovado, **o artefato anterior continua servindo** e o processo sai com 1.
Verificado reprovando (piso forçado a 0,99): nada gravado e o arquivo anterior **idêntico byte a
byte**. Promoção que grava mesmo reprovada não é promoção, é sobrescrita.

**Round-trip verificado no ato da promoção**, além de no teste: grava, recarrega, compara as 1.409
probabilidades da validação com `np.array_equal` — **idênticas**, não "próximas". Serialização não
é aproximação numérica; diferença na décima casa significaria objeto reconstruído em vez de
restaurado.

**Os 5 testes novos, e o que cada um pega** (a suíte anterior tinha 56 e **nenhum tocava o objeto
serializado** — ela exercitava o pipeline em memória, que é justamente o que a API não usa):
round-trip bit a bit · metadados ⇄ pipeline não podem divergir · versão de sklearn forjada é
**barrada** · formato desconhecido é barrado · **carga num subprocesso sem a raiz do repo no
`sys.path`** (o ambiente do container: dos 9 artefatos do `mlruns/`, 8 carregam e 1 não — o
`logreg+feat`, por causa do `FunctionTransformer`).

**Item 96(a) aplicado:** `InconsistentVersionWarning` entrou no `filterwarnings` do
`pyproject.toml`, ao lado do `ConvergenceWarning` que já estava lá pelo argumento idêntico e
escrito — *"um warning que ninguém lê é um warning que não existe"*. O `requirements.in` tinha a
razão do pin em comentário desde a Etapa 3 e nenhum mecanismo que a aplicasse; agora tem.

**Duas limitações declaradas:**
- **O artefato não é versionado no Git** (`models/*` no `.gitignore`, decisão da Etapa 9.5: *o
  registry é a fonte de verdade, não o repositório*). Consequência a resolver na 9f: a imagem
  precisa do artefato, e ele não vem do clone — ou o build roda `make promover` (e aí o dado bruto
  LGPD entraria na imagem, o que a M03-A04 proíbe), ou o artefato é injetado como camada/volume.
  **É decisão da 9f, e está registrada aqui como pendência, não como esquecimento.**
- **Promover duas vezes gera sha256 diferentes**, porque `promovido_em` é um timestamp. O modelo é
  bit-determinístico (três execuções dão o mesmo `repr` de PR-AUC); a identidade do **arquivo**,
  não. É o comportamento certo para um carimbo de deploy — mas quem quiser comparar dois artefatos
  compara as métricas e o commit, não o sha do arquivo.

**Dependências:** `httpx2` e `joblib` declarados no `requirements.in` com o motivo. O `httpx2` era
**bloqueio** e não planejamento — sem ele `from fastapi.testclient import TestClient` levanta
`RuntimeError` e nenhum teste de rota da 9d rodaria. ⚠️ Starlette 1.6 pede `httpx2`, não `httpx`:
o segundo ainda funciona emitindo `DeprecationWarning` a cada import. O `joblib` chegava de carona
como transitiva do scikit-learn e agora é importado **diretamente** por `src/artefato.py` — se um
sklearn futuro trocar de serializador, o lock sairia sem joblib e nenhum artefato do histórico
carregaria.

### 9d · A API (17/08/2026)

`src/api/` em três camadas — `schema.py` (contrato), `servico.py` (pontuar), `app.py` (HTTP) —,
separadas por **uma razão para mudar**. O critério que decidiu a separação não foi estético:
*isto fica testável sem subir servidor?* — os testes de limiar e de contrato de erro rodam contra
o serviço e contra dublês, sem `TestClient` e sem tocar o disco.

**Rotas:** `GET /health` · `POST /v1/predict` · `POST /v1/predict-batch`.

#### O que o contrato de domínio fechou — os quatro casos que davam 200

Com validação apenas de esquema (campo presente + castável para float), medido em 17/08/2026
contra o campeão real, estes payloads eram **aceitos e prediziam normalmente**:

| payload | antes | efeito medido na validação | agora |
|---|---|---|---|
| `Contract = "Vitalicio"` | 200 | 153 clientes (10,9%) trocam de lado no limiar | **422** |
| `Tenure Months = -999` | 200 | P(churn) = 1,0000 nas 1.409 linhas; 878 (62,3%) cruzam o limiar | **422** |
| `Monthly Charges = -999` | 200 | P(churn) = 0,0000 nas 1.409 linhas | **422** |
| `Total Charges = 1e9` | 200 | P(churn) = 1,0000 nas 1.409 linhas | **422** |

As 10 categóricas aceitavam lixo porque `handle_unknown="ignore"` faz o valor desconhecido virar
**linha de zeros** — que para o modelo significa *"a categoria de referência"*, não *"não sei"*.
Correto no treino, perigoso na inferência. `-999` não é hipótese: é a sentinela de nulo mais comum
em sistema legado, que é quem consome uma API interna de churn.

🔑 **O contrato é derivado do artefato, não de `config.py`** — 13 nomes e a ordem de
`pipe.feature_names_in_`, 25 valores de `ohe.categories_` viram `Literal`, e as 3 numéricas ganham
`Field(ge, le)`. Zero constante de coluna escrita à mão. ⚠️ As **faixas** são a exceção declarada:
são decisão de negócio (0–120 meses · 0–200 · 0–12.000), folgadas de propósito, porque colar o `le`
no máximo do treino (72 meses) rejeitaria um cliente legítimo de 73 — o alvo é `-999` e `1e9`, não
a cauda. Um teste exige que **toda numérica do artefato tenha faixa declarada**, senão uma coluna
nova entraria sem limite e em silêncio.

#### As decisões de mecanismo, e uma incompatibilidade que só aparece implementando

🚨 **Derivar o contrato do artefato (item 94) e carregar por `lifespan` (item 98a) são
incompatíveis — e quem cede é o mecanismo.** O schema precisa do artefato **antes de as rotas
existirem**; o `lifespan` roda depois. A solução foi a factory `criar_app(artefato)`, que carrega
na **construção** do app: mais cedo que o `lifespan` e servindo ao mesmo propósito medido — o
**deploy** paga os ~715 ms, não o primeiro cliente (383× contra `@lru_cache`, que não é
carregamento antecipado e sim carregamento preguiçoso com memória). Ganho colateral: a factory
recebe o artefato, então o teste monta o app sobre um artefato de fixture, e fica explícito que a
API serve **um objeto específico**, não "o modelo" genericamente.

| Decisão | Motivo medido |
|---|---|
| **`def`, nunca `async def`** em todas as rotas | vazão idêntica (203,1 × 206,4 ms — o GIL serializa), mas o atraso máximo do event loop vai de 32,58 ms para 206,28 ms = o lote inteiro. Quem fica preso na fila junto é o `GET /health` do orquestrador ⇒ healthcheck expira ⇒ container reiniciado no meio da campanha |
| **`/predict-batch` é o endpoint principal** | 825×: 1.409 linhas custam 2,853 ms em lote e 2.353 ms uma a uma. O custo do sklearn é fixo por chamada (~1,67 ms); a linha marginal custa 2,0 µs. `/predict` é o caso de lote 1 |
| **`max_length = 5.000`** no lote | lista sem teto é vetor de negação de serviço: o trabalho é síncrono e segura um worker. 5.000 ≈ 12 ms e cobre a carteira (7.043) em duas chamadas |
| **`response_model` em todas as rotas** | o campo não declarado **não sai**, mesmo que o `return` o inclua (36 bytes × 374 bytes medidos). É a única defesa contra overexposure que não depende de alguém lembrar |
| **`pd.DataFrame(linhas)`, com `by_alias=True`** | o `ColumnTransformer` seleciona por **nome**; `np.array([[...]])` fixaria a ordem no corpo da função, onde nada a verifica. Reordenar dois campos do schema — refatoração que nenhum linter barra — daria 200 OK com número errado. Há teste que embaralha as chaves do JSON |
| **`predict_proba` + limiar do artefato** | `.predict()` aplica 0,5 implícito: R$ 39.296 × R$ 31.750 por ciclo, **R$ 7.546 e 83 churners**. A resposta leva probabilidade, decisão **e o limiar** — sem ele a decisão não é auditável depois |
| **`Annotated[...]` em vez de `= Depends(...)`** | as duas formas são equivalentes para o FastAPI e só a primeira não dispara `B008`. Suprimir a regra seria trocar uma correção de duas palavras por uma afirmação sobre o linter que ninguém verifica |

#### O vazamento de LGPD, fechado em duas peças (nenhuma basta sozinha)

`extra="forbid"` impede que `CustomerID` entre em silêncio, **e concentra o vazamento**: o erro
`extra_forbidden` devolve `loc: ["CustomerID"]` **e** `input: "3668-QPYBK"`. Quem fecha é o handler
de `RequestValidationError`, que remove `input` e `ctx` e mantém `loc` (nome de campo não é dado
pessoal). O mesmo handler cobre o 500: `str(e)` do sklearn cita o dado do cliente
(`could not convert string to float: 'setenta reais'`). **Verificado nos 8 payloads inválidos:
nenhum devolve valor do payload.** A resposta de sucesso também não ecoa a entrada — a correlação
é pelo `request_id` gerado no servidor, que vai no corpo e no header `X-Request-ID`.

#### `/health` que afirma algo

Declara `status` (prontidão, lida do **objeto que serve**), `versao_modelo`, `artefato_sha256`,
`n_features`, `limiar_operacao` e as versões de biblioteca. 🎯 **Verificado reprovando:** com o
artefato apagado do disco, o processo **morre na inicialização** com mensagem acionável —
ele não sobe e responde `healthy`, que é o comportamento que a composição das figuras da M04-A04
produz. Subir degradado é escolher a falha silenciosa.

#### 🚨 Achado novo: a predição não é bit-idêntica entre lote e unitário

Pontuar as 1.409 linhas da validação **em lote** e **uma a uma** dá resultados diferentes em
**495 linhas (35%)**, com diferença máxima de **2,2e-16** (um ulp) — é o BLAS tomando caminhos de
vetorização diferentes conforme o número de linhas. **Zero clientes mudam de lado no limiar de
0,29**, então não há efeito de negócio; mas a consequência de projeto fica registrada: a resposta
para o mesmo cliente **não é reproduzível byte a byte** entre `/predict` e `/predict-batch`, logo
nada rio abaixo pode comparar predições por igualdade exata (cache por hash, deduplicação,
reconciliação do log da Etapa 10 — todos por tolerância). O teste de caracterização do artefato
não é afetado, porque pontua sempre o mesmo lote.

#### 🚨 O erro que só a máquina limpa podia encontrar (cometido e corrigido em 17/08/2026)

A primeira versão terminava com `app = criar_app()` no nível do módulo — o idioma que todo tutorial
usa, e que o `uvicorn src.api.app:app` pede. Com ele, **carregar o artefato virou efeito colateral
do `import`**: qualquer `import src.api.app` passou a exigir `models/campeao.joblib` no disco.

`make ci` local passou (o artefato está aqui) e o CI falhou **na coleta**, no runner limpo, onde
`models/` está vazio **por decisão nossa** — 81 testes derrubados por uma linha, com a mensagem
de erro certa no lugar errado.

🔑 **A lição não é sobre a linha, é sobre o instrumento: este defeito só existe onde o arquivo não
existe.** Nenhuma execução local podia encontrá-lo, e é exatamente o "funciona na minha máquina"
que o CI existe para pegar — desta vez com a raiz correta, porque a diferença entre as duas
máquinas *é o objeto da etapa*.

**Correção:** o módulo não exporta objeto de app; o uvicorn recebe a **factory**
(`uvicorn src.api.app:criar_app --factory`), o que preserva a propriedade que importava — a carga
acontece antes de a primeira conexão ser aceita, e o processo morre na inicialização se o artefato
não bater. Junto vieram duas coisas que trazem o defeito para dentro do alcance do desenvolvimento:
`config.ARTEFATO` passou a ser configurável por `TC_ARTEFATO` (que o container vai querer de
qualquer forma, para artefato em volume), e **um teste roda `import src.api.app` num subprocesso
com um caminho inexistente**, exigindo que o import passe e que `criar_app()` falhe.

#### Escopo declarado (o que NÃO foi feito, e por quê)

- **Sem autenticação** — limitação declarada na descrição da própria API e na documentação, não
  omissão. A M04-A01 nomeia *"API interna sem proteção"* como anti-padrão, e a banca procura.
- **`/docs` e `/openapi.json` abertos** — publicam 13 nomes de feature, 25 valores e 3 faixas.
  Não é dado pessoal: é a descrição do modelo. Mantidos por ser API interna; `openapi_url=None` é
  o que se faz quando a API sair da rede interna.
- **Sem Strategy/Factory/Observer** — um endpoint de predição não passa no teste do próprio
  padrão. Recusar o padrão é o conteúdo da aula sobre padrões.
- **Sem log de inferência em JSONL** — é a Etapa 10. O `request_id` e o middleware que cronometra
  (inclusive o 422) já são o gancho, e custam +13%.

**20 testes novos** (81 na suíte, `make ci` verde) e a API verificada com **uvicorn real**, não só
com `TestClient`: `/health` responde, `/v1/predict` devolve 0,3700 para o cliente de exemplo, e a
latência local ficou em mediana 3,5 ms / p95 4,9 ms com 20 requisições sequenciais — número de
desenvolvimento, não SLA (medida honesta exige carga concorrente).

---

### 9f · A imagem de serviço (18/08/2026)

A API existia; o que faltava era ela existir **empacotada**. A imagem é o artefato que vai para
qualquer destino — VM, Kubernetes ou PaaS — e é ela, não o Dockerfile, que é reprodutível por
construção. 🔑 *O Dockerfile é a receita; a imagem é o artefato.* A receita só é repetível se
fixar tudo o que resolve em tempo de build: **base por digest, lockfile com pins e plataforma
explícita** — as três estão no arquivo, e a ausência de qualquer uma produziria imagens
diferentes em meses diferentes com o mesmo `git checkout`.

#### O `.dockerignore` veio antes do Dockerfile, e é uma allowlist

🚨 **O Docker não lê o `.gitignore`.** O repositório já excluía `mlruns/`, `mlflow.db`,
`data/processed`, `logs/*.jsonl`, `.env` e `*.pem` — **do Git**. Medido construindo de propósito
uma imagem com `COPY . .` e sem exclusão nenhuma:

| | com `.dockerignore` (allowlist) | sem, o `COPY . .` de tutorial |
|---|---|---|
| contexto transferido | **2,61 kB** | **1,5 GB** |
| imagem resultante | 510 MB, funcional | **2,03 GB**, e sem Python instalado |
| `data/raw/Telco_customer_churn.xlsx` | fora | **dentro** (1.368.250 bytes, 7.043 clientes reais) |
| `.venv` | fora | **dentro** (1,5 GB de binários **arm64 de macOS** numa imagem Linux) |
| `.git`, `mlruns/`, `mlflow.db` | fora | dentro (4,2 MB + 3,4 MB + 1,2 MB) |

**Camada Docker é imutável:** um `RUN rm` posterior não apaga o arquivo da camada anterior — quem
puxar a imagem tem o dado. A decisão precisa ser tomada **antes** do `COPY`, porque depois não
existe "apagar", e isso transforma o assunto de peso em **exposição de dado pessoal por imagem**.

🔑 **Por que allowlist e não lista de exclusões:** uma denylist protege contra o que já existe; a
allowlist protege contra o que ainda vai existir. No dia em que alguém criar
`data/exportacoes/clientes.csv`, a denylist deixa passar em silêncio — é assim que dado pessoal
entra em imagem publicada — e a allowlist exige uma linha nova, escrita por quem teve de pensar
no assunto.

#### A imagem de serviço ≠ a imagem de experimentação

Medido no `site-packages` com a mesma régua (`du`):

| | tamanho |
|---|---|
| venv completo (experimentação) | **1.400 MB**, 118 distribuições |
| imagem de serviço, filesystem inteiro | **510 MB** |
| `site-packages` dentro da imagem | 394 MB, **30 distribuições** |
| `torch` sozinho, no venv | **1.057 MB** — *o dobro da imagem de serviço inteira* |
| o modelo servido | **8.306 bytes** |

O campeão é uma regressão logística e **não importa `torch` em lugar nenhum**; ele está no repo
porque a Etapa 8 é exigência da fase, e a Etapa 8 é treino. 🔑 É a distinção **código ≠
dependência**: o perdedor continua versionado (reprodutibilidade e aprendizado organizacional) e
mesmo assim fora do runtime. Dentro da imagem, `scipy` (111 MB) + `pandas` (77 MB) + `sklearn`
(49 MB) + `numpy` (42 MB) são **~70%** do peso — *o que não cabe numa função serverless não é o
modelo, é o que ele precisa para existir*.

⚠️ O `requirements-serve.txt` é derivado do `requirements.txt` **com as mesmas versões**, não
resolvido de forma independente: `src/artefato.py` compara a versão do scikit-learn gravada no
artefato com a do ambiente e mata o processo na carga se divergirem. Derivar do lock existente
faz a divergência ser **impossível** em vez de detectável.

#### As decisões do Dockerfile que um tutorial não tem

| decisão | motivo medido |
|---|---|
| `FROM python:3.12-slim@sha256:2c941e…` | tag é mutável, digest não. E **3.9 não constrói este projeto**: `scikit-learn 1.9.0` e `numpy 2.4.6` exigem `>=3.11` — falha alta que **só existe porque há pins**; sem eles o pip acharia versões antigas, a imagem subiria e o artefato quebraria na carga |
| lockfile copiado **antes** do código | camada de dependências reaproveitada: alterar uma linha de `app.py` não reinstala scipy |
| `COPY src/` + `COPY models/campeao.joblib` explícitos | `COPY . .` dependeria de o `.dockerignore` estar certo para não vazar dado pessoal; cópia explícita não depende de nada |
| `USER appuser` (uid 999) | verificado dentro do container |
| `HEALTHCHECK` que exige `status == "pronto"` | **verificado reprovando**, três cenários: serviço real → exit 0; porta morta → exit 1; **um servidor que responde 200 com `status: degradado` → exit 1**. Um healthcheck que só pergunta "responde?" aprova exatamente o cenário de falha que importa |
| `criar_app --factory`, sem `--reload`, `--workers 1` | a factory impede que carregar o artefato vire efeito colateral do import (erro real de 17/08); o nº de workers sai da **RAM**, e o container mede **171,2 MiB** — 4 workers ≈ 685 MB, acima do teto de 512 MB de um plano gratuito |
| `${PORT:-8000}` com `exec` | um PaaS reivindica a porta; o `exec` faz o uvicorn ser PID 1 e receber o `SIGTERM` do `docker stop` |

🚨 **Plataforma.** Um contêiner Linux no macOS roda numa VM **arm64**: `docker build` no Apple
Silicon produz imagem arm64, e o runner do Actions e a maioria das clouds são `x86_64` ⇒
`exec format error`, **no deploy, não no build local que passou**. O `make docker-build` usa
`--platform linux/amd64` por padrão. Verificado: a imagem amd64 reporta `linux/amd64`, sobe e
responde `pronto` (emulada neste Mac, com `platform.machine() == "x86_64"` dentro dela).

#### Etapa 9e — o teste de integração contra o container

`make docker-teste` constrói, sobe, espera o healthcheck e roda `scripts/integracao_container.py`
contra a imagem. **Ele encontrou um defeito na primeira execução**, e encontrou porque manda as
**1.409 linhas reais da validação** em vez de um payload sintético (ver o achado abaixo).

Resultados na imagem `linux/amd64` (emulada — as latências não são de produção):

| verificação | resultado |
|---|---|
| identidade das predições | `max\|dif\| = 2,22e-16` (**4 ulps**), **40%** das linhas diferem, **0 decisões trocadas** no limiar 0,29, e **PR-AUC idêntico nos 10 dígitos** (0,6646020519 dos dois lados) |
| unitário × lote | dif = 0,000e+00 na linha testada |
| contrato de erro | categoria inédita · `-999` · campo extra · campo faltando ⇒ **422**, nenhum devolvendo valor do payload; vazio declarado ⇒ **200** |
| latência (amd64 emulada) | unitário p50 6,82 / p95 9,54 ms · 8 concorrentes p50 44,4 ms · lote de 1.409 p50 26,3 ms |
| latência (arm64 nativa) | unitário p50 4,37 / p95 13,30 ms · lote de 1.409 p50 15,6 ms |
| `/health` sob carga | com **8 lotes de 1.409 em voo** (pior lote 279 ms), a probe respondeu p50 73,9 ms e máx 178,7 ms — folgado contra o timeout de 5 s. É o `def` × `async def` verificado no ambiente real |

🔑 **A diferença numérica entre macOS e Linux é real e irrelevante — e as duas metades importam.**
40% das linhas mudam no último bit porque o BLAS muda de caminho entre plataformas; **nenhuma**
decisão de negócio muda, e a métrica reportada não se move na décima casa. Consequência de
projeto, não de negócio: **nada rio abaixo pode comparar predições por igualdade exata** (cache
por hash de resposta, deduplicação, reconciliação do log da Etapa 10).

#### 🚨 Achado — o contrato ficou mais estreito que o pipeline que ele protege

A linha 487 da validação foi **rejeitada com 422** pela própria API. Não é dado inválido: é um
dos **11 clientes do Telco com `Total Charges` vazio, todos com `Tenure Months = 0`** — quem
ainda não teve ciclo de faturamento. A Etapa 2 decidiu que esse vazio é *medição verdadeira* e o
imputa com 0 (não com a mediana), e o pipeline sempre soube tratá-los: `predict_proba` com `NaN`
devolve **0,2449336585**. Quem não sabia era o schema, porque todo campo numérico era `float`
obrigatório com `ge`/`le` — e `NaN` não satisfaz `le`.

**Consequência de negócio:** a API recusava exatamente a população que uma campanha de retenção
mais quer pontuar — o cliente do primeiro mês.

🔑 *É o erro simétrico do que o módulo de schema foi escrito para impedir.* Lá o risco era
**aceitar lixo** (`-999`, categoria inexistente); aqui era **recusar dado legítimo**. Mesma raiz:
contrato escrito à parte do objeto que ele descreve. A correção mantém a regra — quem decide é o
artefato: o grupo `zero` do `ColumnTransformer` **é** a lista de colunas cujo vazio tem tratamento
declarado, e o schema o lê de lá.

⚠️ **Duas peças, e nenhuma serve sozinha.** Aceitar `null` no schema **sem** convertê-lo para
`NaN` no serviço trocaria o 422 por um **500**: `pd.DataFrame` com `None` produz `dtype=object`,
o imputador não reconhece a ausência e o `LogisticRegression` levanta `ValueError: Input X
contains NaN`. Medido.

⚠️ **E a assimetria é deliberada:** `Tenure Months: null` continua **422**, embora o grupo `num`
também tenha imputador. Em `Total Charges` o vazio *significa* algo e o valor imputado **é** essa
informação; em `Tenure Months` um vazio é dado faltando do integrador, e imputar a mediana do
treino em silêncio transformaria uma falha de integração numa predição plausível — a sentinela de
nulo entrando pela porta da frente. *O pipeline sabe imputar as duas; a API aceita o vazio só onde
ele quer dizer algo.*

📌 Nota para quem integra: `json.dumps(float("nan"))` do Python emite `NaN` literal, que **não é
JSON válido**; o servidor responde 422 com `less_than_equal`, mensagem que aponta para o lugar
errado. O contrato pede `null`.

#### 🚨 Achado — o `/health` declarava o ambiente errado, e só o container mostrou

O endpoint tinha um campo `versoes`, alimentado pelos metadados do artefato. Dentro da imagem ele
dizia **Python 3.12.5** enquanto o processo rodava **3.12.14** (`python:3.12-slim`). Nada estava
errado no artefato: o carimbo do treino estava correto. **O rótulo é que respondia a outra
pergunta** — num endpoint de prontidão, cuja pergunta é *"o que este serviço tem?"*.

É a família de erro que atravessa este repositório — *uma afirmação sobre um sistema que ninguém
confronta com o sistema* — e no `/health` ela é a pior variante, porque este endpoint existe
justamente para declarar identidade. **Um campo ambíguo é pior que um campo ausente: parece
resposta.** Agora são dois campos, `versoes_treino` e `versoes_runtime`, e um teste exige que o
**scikit-learn** coincida (é onde a serialização mora; divergir ali mata o processo no boot).

#### O artefato passou a ser versionado — exceção nomeada, não revogação da regra

`models/*` continua no `.gitignore` com o motivo escrito (*"o Registry é a fonte de verdade, não
o Git"*), e a regra continua certa. O que mudou é o **destino**: a plataforma de deploy escolhida
constrói a imagem a partir do **clone do Git** — sem máquina nossa no caminho e sem volume no
plano gratuito. Ou o artefato está versionado, ou a imagem sobe sem modelo.

É `!models/campeao.joblib` e **nunca** `!models/*`: nomear o arquivo é o que impede o próximo
`.joblib` experimental de entrar de carona. São **8.306 bytes**, e o ganho colateral é que o
artefato passa a ter sha256 rastreável pelo Git — que `src/artefato.py` já verifica na carga.
🔑 *Exceção declarada com motivo é decisão; exceção silenciosa é a regra apodrecendo.*

#### Escopo declarado

- **O deploy em nuvem não foi feito nesta etapa** — a imagem está pronta e portátil, e escolher o
  destino é a decisão seguinte. A portabilidade é o argumento: *a mesma imagem roda local, em VM,
  em Kubernetes ou num PaaS* — trocar de destino não reescreve a aplicação.
- **O CI não constrói a imagem.** O risco que isso deixa aberto (Dockerfile quebrado descoberto
  tarde) é pequeno **porque a plataforma escolhida builda ela mesma e falha o deploy sem
  publicar** — ela é, de fato, o CI da imagem. Acrescentar um job de build ao Actions custaria
  ~2 min por push para antecipar um erro que já não chega a produção.
- **O pacote não é instalável ainda** (`pip install .` é a 9g). Enquanto não for, `PYTHONPATH`
  aparece em três lugares — no Dockerfile, no alvo `docker-teste` e em qualquer script fora da
  raiz. É o item 17 pedindo a correção estrutural em vez de mais uma variável por chamador.
- **Sem registry de imagem, sem multi-arch publicado, sem volume para o artefato.** A relação
  artefato 8,3 KB × imagem 510 MB = **1 : 61.000** quantifica o acoplamento que resta: promover um
  modelo novo republica 61 mil vezes mais bytes do que a mudança real. É limitação a declarar e
  caminho de evolução, não obra a fazer agora.

**5 testes novos** (87 na suíte) e o fluxo `make docker-build && make docker-teste` verde de ponta
a ponta na imagem `linux/amd64`.

---

### 9f-quater · O deploy (18/08/2026) — e o defeito que só a nuvem revelou

**URL:** https://tc-churn-api.onrender.com · plano gratuito (512 MB · 0,1 CPU, números da
documentação oficial).

A escolha de destino é PaaS, e a defesa é por medição, não por preferência: **FaaS foi descartado
com número** (o fechamento de serviço tem 253,2 MB contra o limite de 250 MB de função por zip —
estoura por 1,3% antes mesmo do adaptador e do artefato) e IaaS exigiria administrar máquina. A
portabilidade da imagem é o que impede isso de virar aprisionamento: **a mesma imagem roda local,
em VM, em Kubernetes ou noutro PaaS** — trocar de destino não reescreve a aplicação.

#### O deploy é código, não memória de quem clicou

`render.yaml` na raiz. Tudo ali poderia ter sido preenchido no painel; a diferença é que painel não
entra no `git log`, não é revisável e não responde *"por que o serviço está assim?"* seis meses
depois. Mesmo argumento do `Makefile` contra "os comandos que eu sei de cabeça".

⚠️ **Os campos foram conferidos contra a documentação vigente, e o principal havia mudado de
nome:** não existe mais `autoDeploy: true|false`, e sim **`autoDeployTrigger`**. Escrever de
memória produziria um YAML que a plataforma aceita **ignorando o campo** — a configuração que você
acha que fez, e não fez. É a regra da Etapa 7 (*conferir a assinatura antes de montar a grade*)
valendo para infraestrutura.

#### 🎯 O último elo do CI/CD, por uma linha

`autoDeployTrigger: checksPass` faz o Render **esperar os checks do GitHub Actions** e só implantar
se todos passarem. O default (`commit`) publicaria a cada push — **entrega contínua sem integração
contínua**, o exato anti-padrão que o backlog previa. O plano era desligar o auto-deploy e disparar
por webhook no fim do job, com um secret a gerenciar; a plataforma já resolvia isso nativamente.
**Verificado funcionando:** o commit da correção de porta só foi implantado depois de o CI fechar
verde.

⚠️ Efeito colateral a conhecer: o Render **não implanta** commit sem check algum. Aqui o
`ci.yml` dispara em todo push na `main`, mas se um dia ele ganhar filtro de `paths:`, commits fora
deles deixariam de implantar **em silêncio**.

#### 🚨 O defeito: a porta declarada em dois lugares com valores diferentes

Logo após o primeiro deploy, o serviço passou a responder **`x-render-routing: no-server` em 48%
das requisições**. O diagnóstico dependeu de separar camadas, e todas as pistas apontavam para
fora da aplicação:

| evidência | o que elimina |
|---|---|
| os 404 **não têm** `x-render-origin-server: uvicorn` | nunca chegaram à aplicação |
| nos logs do serviço, **todos** os `/health` são 200 | a aplicação não errou uma requisição |
| **zero restarts**, nenhum OOM nos eventos | não era crash loop nem memória |
| quando respondia, o `sha256` era **o correto** | não era artefato errado |
| local, `--cpus 0.1` **nativo** sobe em 1-2 s | não era a CPU do plano gratuito |

🔑 **A causa estava numa linha do log que passa despercebida:** `==> Detected service running on
port 10000`, aparecendo **centenas de requisições depois** de o serviço já estar de pé. O Render
estava *procurando* a porta. E procurava porque nós declaramos duas: o `EXPOSE` do Dockerfile dizia
**8000**, o processo escutava em `${PORT:-8000}` = **10000** (o default da plataforma). A
documentação usa o advérbio que descreve exatamente o sintoma: *"Render is **usually** able to
detect and use it"*.

✅ **Correção: `PORT: 8000` no `render.yaml`** ⇒ `EXPOSE`, `CMD` e plataforma passam a declarar o
mesmo número, e não há mais nada a detectar. Medido depois: **120 requisições, 120 respostas 200**.

🚨 **E o raciocínio que produziu o defeito era bom** — está escrito na primeira versão do arquivo:
*"não fixar a porta, porque é um parâmetro que a plataforma reivindica"*. O erro foi confundir
**não declarar** com **delegar**: não declarar não devolve a decisão à plataforma, devolve a uma
**heurística de detecção**. Delegar de verdade seria declarar o valor que ela usa.

🔑 *O `EXPOSE` é documentação — e documentação que discorda do processo é uma afirmação que alguém
vai ler; aqui, uma máquina.* É a mesma família do campo `versoes` do `/health` corrigido horas
antes: nada estava "errado", o **rótulo** é que respondia a outra pergunta. A diferença é que ali o
leitor enganado seria humano, e aqui foi o roteador.

⚠️ Ressalva honesta: a documentação **não** confirma que o Render lê o `EXPOSE` para decidir a
porta, então o mecanismo exato permanece hipótese. A correção não depende dela — declarar a porta
nos dois lados remove a ambiguidade que qualquer heurística teria de resolver.

#### Verificação em produção — o mesmo script da 9e, contra a URL pública

| verificação | resultado |
|---|---|
| identidade | **PR-AUC 0,6646020519 local × 0,6646020519 na nuvem** — idêntico nos 10 dígitos; 37% das linhas diferindo em 1 ulp e **0 decisões trocadas** |
| contrato de erro | categoria inédita · `-999` · campo extra · campo faltando ⇒ **422**; vazio declarado ⇒ **200**; nenhum devolvendo valor do payload |
| latência (Brasil → Oregon, 0,1 CPU) | unitário p50 **254 ms** / p95 670 ms · lote de 1.409 p50 **911 ms** |
| `/health` sob carga | com **8 lotes de 1.409 em voo**, a probe respondeu p50 968 ms e máx 1.576 ms — o `def` em vez de `async def` segurando a probe também na nuvem |

⚠️ A latência é dominada por rede e por 0,1 CPU, não pelo modelo: o `x-response-time-ms` que a
própria API devolve marca **1,16 ms** para o mesmo `/health` que leva 254 ms de round-trip.
*Medir latência sem dizer onde o cronômetro estava é métrica sem piso.*

#### Cold start: risco de ENTREGA, não de operação

O plano gratuito **dorme após 15 minutos sem tráfego** e o spin-up leva cerca de um minuto (o
painel avisa "50 segundos ou mais"). Medido: a primeira requisição após o deploy levou **29,8 s**.

🚨 **A consequência é sobre o vídeo da Etapa 11, não sobre produção:** gravar com a API dormindo
trava a demonstração no primeiro `curl`. **Mitigação, que não é técnica: acordar o serviço alguns
minutos antes de gravar.** Não há prazo — o serviço acorda sempre que chamado.

#### 🚨 Os DOIS 404 do dia, e por que a distinção é o método

Horas depois de publicar, abrir a URL no navegador devolvia `{"detail":"Not Found"}`. Parecia o
problema de roteamento voltando; era outra coisa inteiramente, e os headers separam as duas em
segundos:

| | roteamento (48% das requisições) | raiz inexistente |
|---|---|---|
| `x-render-routing` | **`no-server`** | ausente |
| `x-render-origin-server` | **ausente** | `uvicorn` |
| `x-request-id` (nosso middleware) | ausente | **presente** |
| corpo | `Not Found` em texto puro | `{"detail":"Not Found"}` |
| camada | o roteador, sem rota registrada | a aplicação, respondendo corretamente |

🔑 *Mesmo status HTTP, camadas diferentes, correções diferentes.* O primeiro era um defeito de
configuração; o segundo era a aplicação **certa**, dizendo que `GET /` não existe — o que era
verdade, porque nunca escrevemos essa rota.

✅ **A correção é de ENTREGA, não de arquitetura:** a raiz agora redireciona para `/docs`. O
motivo é que aquele endereço é o que vai na documentação e no vídeo, e é a primeira coisa que um
avaliador abre — receber um 404 tecnicamente correto ali é a pior primeira impressão possível por
uma rota de três linhas. ⚠️ Duas exceções deliberadas às regras da casa, ambas por não haver o que
proteger: **sem `response_model`** (um redirecionamento não tem corpo, logo não há campo a vazar) e
**fora do schema OpenAPI**, porque "a raiz redireciona" é conveniência de navegador, não contrato
de API. 📌 Acoplamento declarado: se o `/docs` for desligado (o que se faz ao tirar a API da rede
interna), esta rota muda junto.

⚠️ **E o teste verifica o redirecionamento SEM segui-lo**, além de conferir que o destino existe:
seguir mediria o `/docs`, que é outra rota, e **redirecionar para um 404 seria pior que o 404
direto**.

#### Nota de medição: o cold start de spin-down NÃO foi medido

O número que temos é **29,8 s** para a primeira requisição **após um deploy**, que é fenômeno
diferente do spin-down por inatividade. A medição do spin-down foi tentada e **invalidou-se**: o
serviço precisa de 15 minutos sem tráfego nenhum, e nesse intervalo houve polling de verificação e
dois redeploys — o relógio de inatividade zerou a cada um, e a "primeira requisição após o
spin-down" saiu em 0,247 s, ou seja, mediu um serviço acordado.

🔑 *Registrar a medição inválida vale mais que substituí-la pelo número que a plataforma declara* —
e a ação que dela decorre não muda: **acordar o serviço antes de gravar**. Para medir de verdade é
preciso 15 minutos de silêncio deliberado, o que só faz sentido fazer quando ninguém estiver
trabalhando no repositório. Referência da plataforma: spin-down em 15 min, spin-up de "50 segundos
ou mais".

#### Limitações declaradas do deploy

- **Instância única, sem redundância.** Plano gratuito: se ela cair, não há para onde rotear.
- **Sem domínio próprio e sem autenticação** — a segunda é decisão registrada, não esquecimento.
- **HTTPS termina no proxy da plataforma**; a aplicação fala HTTP na rede interna.
- **Build minutes são limitados** e cada build consome de 3 a 5 minutos. Com `checksPass`, só
  commits aprovados pelo CI consomem cota — o que é mais uma vantagem dele.

---

## 5f. Etapa 10 — monitoramento e manutenção

### 10a · O log estruturado de inferência (18/08/2026)

O `request_id` existia desde a 9d, era gerado no servidor, devolvido no header e
**não era escrito em lugar nenhum**: `grep -rn "logging\|logger" src/` dava zero. Ele
correlacionava a resposta com um log que não existia. Este bloco cria o log.

**Uma linha JSON por requisição, em stdout, com 7 campos.** Os 6 canônicos
(`timestamp`, `request_id`, `metodo`+`rota`, `status_code`, `latency_ms`) mais o
**`artefato_sha256`**. O sétimo não é redundante com o que a resposta já devolve: a
resposta é efêmera e o log é o rastro de auditoria. Quando o PSI cruzar o limiar daqui
a três semanas, a primeira hipótese a descartar é *"trocaram o modelo no meio da
janela"* — e sem o hash na linha ela não é descartável.

#### 🚨 A decisão de máscara LGPD, e por que ela não é "tudo ou nada"

A regra da casa é decidir a máscara **antes** de ligar o log, por assimetria: aumentar
o que se loga é uma linha de configuração; desfazer o que já foi emitido não é operação
nenhuma. **Log emitido não volta.**

O que torna esta decisão diferente das anteriores é que é a **terceira camada** do
mesmo problema, e a única sem solução por exclusão:

| camada | protegida por | o que a regra faz |
|---|---|---|
| repositório | `.gitignore` (`logs/*.jsonl`, motivo LGPD escrito) | exclui |
| imagem | `.dockerignore` (allowlist, 9f) | exclui |
| **stdout do container** | 🚨 **nada** | — |

Em container se loga em stdout (o filesystem é efêmero) e, no Render, **stdout é
coletado pela plataforma** — um terceiro, cuja retenção não é nossa. Não há regra de
exclusão possível, porque logar é a finalidade do arquivo. *Onde não se pode excluir,
decide-se o que se escreve.*

E o conflito é **interno à Etapa 10**: o `request_payload` é o insumo do data drift
**e** é o dado pessoal — pior aqui do que no caso genérico, porque a Etapa 5 manteve
`Gender`, `Senior Citizen`, `Partner` e `Dependents` nas features **de propósito**,
para que a auditoria de fairness da 10.5 possa medir o viés em vez de ficar cega a ele.
*A decisão certa numa etapa cobra o preço em outra.*

✅ **A saída: as duas famílias de drift têm custos de privacidade opostos, então são
tratadas separadamente.**

| | precisa de | é dado pessoal? | quando |
|---|---|---|---|
| **prediction drift** (`P(ŷ)`) | as probabilidades de saída | **não** — número sem atributo ao lado não identifica ninguém | **sempre**, inclusive em produção |
| **data drift** (`P(X)`) | as 13 features | **sim** | só com `TC_LOG_FEATURES=1` |

⇒ A vigilância que roda **de graça e sem exposição** fica ligada na nuvem; a que custa
privacidade é ligada por quem roda, na prática localmente para o `simulate_drift.py`.
Prediction drift é sintoma e não causa — mas é sintoma de graça.

⚠️ **E este `getenv` com default é legítimo onde o do item 100 não era.** Lá o default
transformava "esqueci de configurar" em "configurei com o valor público": a ausência da
variável levava ao estado **inseguro**, e o código funcionava. Aqui a ausência leva ao
estado **conservador**. A regra não é *"nunca use default"* — é ***o default tem de ser
a direção segura***.

#### 🚨 O achado que mudou o código: o handler de `Exception` roda POR FORA do middleware

Medido em 18/08/2026, com um app mínimo:

```
middleware viu: [('RESPOSTA', 200), ('EXCECAO', 'ValueError')]
```

No Starlette, um handler registrado para `Exception` vive no `ServerErrorMiddleware`,
que é o **mais externo de todos**. Logo o nosso middleware **nunca vê a resposta 500** —
vê a exceção crua subindo. Um log escrito a partir de `resposta.status_code` perderia
**toda falha interna**, que é justamente o evento que o log existe para registrar, e o
painel de taxa de erro mostraria zero para sempre.

✅ Correção: `status = 500` como valor inicial + `finally`. O 200, o 422 e o 500 saem
com uma linha cada. É a regra *"instrumentação que só mede o caminho feliz mede o que
não precisa de medição"*, agora com o mecanismo específico do framework medido.

#### 🚨 O defeito que só o teste encontrou — e ele quase passou despercebido

O `logging.StreamHandler` da biblioteca padrão **congela o objeto de stream na
construção**. Como `configurar()` roda dentro de `criar_app()`, o handler ficava preso
ao `sys.stdout` daquele instante. Sob o pytest, isso significou que a linha saía num
descritor que **nem `capsys` nem `capfd` liam**.

🔑 **O que salva aqui é a forma do teste, não a esperteza:** os testes *parseiam* a
linha e comparam conteúdo. Um teste escrito como *"a requisição respondeu 200, logo
logou"* teria ficado **verde sem nunca ter visto uma linha** — e a Etapa 10 inteira
seria construída sobre um log que ninguém verificou.
✅ Correção: um handler que resolve `sys.stdout` **na hora de emitir**. É a família de
*"alguém guardou uma referência e o recurso mudou por baixo"*, e a saída é a mesma de
sempre: não guardar, perguntar toda vez.

#### `--no-access-log` no `CMD` — e o número que o justifica

O uvicorn tem log de acesso próprio, em texto puro. Medido, 3 requisições ao `/health`:

| | linhas JSON | linhas de texto |
|---|---|---|
| sem a flag | 3 | 3 (+ banner) |
| **com a flag** | **3** | **0** (só o banner de startup) |

Duas linhas por requisição, uma delas não-JSON — o parser da Etapa 10 quebraria na
primeira. ⚠️ Não é duplicação dentro da hierarquia do `logging` (isso o
`propagate=False` já resolve): **é outro emissor**, e por isso o antídoto é outro.
Nada se perde: a linha JSON tem método, rota, status e latência — tudo o que o access
log tinha — mais o `request_id` e o `sha256`, que ele não tinha.

#### Verificação (a disciplina de sempre: reprovando)

9 testes novos (88 → **97**). Três sabotagens, cada uma derrubando exatamente o teste
que devia:

| sabotagem | teste que caiu |
|---|---|
| prefixo no `Formatter` (a linha deixa de ser JSON) | os 9, com `json.decoder.JSONDecodeError` |
| `LOGAR_FEATURES = True` (a máscara cai) | `test_o_dado_pessoal_NAO_vai_no_log_por_default` |
| `status` inicial 200 (o 500 some do log) | `test_500_tambem_e_logado` — `assert 200 == 500` |

E a verificação no servidor real, que os testes não fazem: `GET /health` 200, `/v1/predict`
200 e `/v1/predict` 422 produziram **exatamente 3 linhas JSON e nenhuma linha de texto**
além do banner de startup.

#### O que fica declarado como limitação

- **`rota` usa o template**, não a URL. Hoje coincidem (nenhuma rota tem path param) —
  o teste existe para o dia em que uma delas ganhar `/{id}` e a cardinalidade explodir
  sem ninguém perceber.
- **Sem rotação nem retenção do `.jsonl`.** Em produção quem retém é a plataforma, com
  a política dela; localmente o arquivo está no `.gitignore` desde sempre, com o motivo
  LGPD escrito. Uma política de retenção própria é fora do escopo desta entrega, e é
  exatamente o que a opção (a) da máscara exigiria se o payload fosse logado por default.
- **`scores` são arredondados em 6 casas.** É muito mais resolução do que o PSI usa (ele
  bina em ~10 faixas) e descarta de propósito o ruído de última casa já medido entre
  plataformas (40% das linhas diferem em até 4 ulps, com **zero** decisões trocadas).
  Guardar essas casas convidaria alguém a comparar predições por igualdade exata — que é
  precisamente o que aquela medição proibiu.

### 10a-2 · As estatísticas de referência do treino (18/08/2026)

> *Sem baseline não existe drift para detectar, só um gráfico comparando nada.*

**A decisão pendente era de formato** (item 109 do revisita, aberto no ato e fechado no dia
seguinte): as estatísticas moram **dentro dos metadados do artefato** ou num
`models/referencia.json` ao lado?

**Escolhido: dentro do artefato.** O argumento é o mesmo que já decidiu o limiar de operação na
9c — a distribuição de referência é propriedade **daquele modelo**, não do repositório: ela muda
quando o modelo muda. A alternativa criaria exatamente o par que a regra nº 1 do empacotamento
existe para proibir ("dois arquivos que podem não combinar"), e nesta variante a falha é pior que
rótulo errado, porque **não falha**: baseline de um modelo comparado com predições de outro mede
drift fantasma, ou deixa de ver o real. Ninguém recebe erro.

🔑 **O argumento que fechou a decisão só apareceu ao escrever o código: o baseline é
auto-suficiente.** Ele guarda as **proporções esperadas por bin**, não os dados. Quem calcula PSI
em produção não precisa do dataset de treino — que é justamente o que **não pode** estar lá
(`data/raw` é LGPD e o `.dockerignore` o barra). Uma referência que exigisse reler o Excel seria
uma referência que não roda onde o drift acontece.

#### A verificação antes de escolher — e o risco que ela inverteu

O item 109 mandava checar se `carregar(estrito=True)` valida o conjunto de chaves dos metadados,
porque acrescentar uma chave nova poderia **recusar artefatos antigos**. Medido: **não valida** —
todos os acessos usam `metadados.get(...)`. Ou seja, o risco previsto não existia, e o risco real
era o oposto: um artefato sem baseline **carregaria em silêncio**, subiria, responderia 200, e a
falha só apareceria semanas depois como **ausência de alerta** — que é indistinguível de ausência
de problema.

⇒ A carga ganhou a **quarta checagem**: exige a `referencia` e exige que ela **cubra as features
declaradas**. Baseline de 13 colunas para um modelo de 12 mede drift de uma coluna que o modelo
não usa mais, e não mede o da que entrou.

#### O furo encontrado no meio da execução: `P(X)` congelado, `P(ŷ)` não

A primeira versão congelou só a distribuição de **entrada**. O erro aparece quando se cruza com a
decisão de máscara da 10a:

| | vai para o log | tem baseline (1ª versão) |
|---|---|---|
| `features` (data drift) | **só com `TC_LOG_FEATURES=1`** | ✅ |
| `scores` (prediction drift) | **em toda linha, sempre** | ❌ |

Isto é, o baseline existia para a metade que quase nunca é coletada e faltava para a que **sempre**
é. Corrigido: a referência passou a ter um bloco `scores`.

**As duas metades vêm de partições diferentes, de propósito.** `X` sai do **treino**; os `scores`
saem da **validação** — in-sample o modelo é otimista por construção, e um baseline otimista faria
produção parecer deslocada para sempre (drift fantasma permanente, que treina o time a ignorar o
alerta). ⚠️ Medido neste campeão: PSI treino × validação nos scores = **0,0022**, ou seja, a
escolha custaria quase nada aqui — LogReg regularizada com 13 features mal overfitta. A decisão é
por princípio e a medição diz que ela é **barata**, não dispensável: com um HGB profundo no lugar,
o mesmo erro não sairia de graça.

#### O que foi congelado

- **3 numéricas:** média, desvio, min/max, 7 quantis, **bordas de bin** (decis) e proporções.
- **10 categóricas:** frequências relativas, ordenadas por nome (ordem que depende do dado faz o
  mesmo baseline produzir bytes diferentes conforme empates).
- **scores:** quantis, bordas, proporções, o limiar e a **taxa acima do limiar**.

🔑 **As bordas dos bins viajam junto, e isso não é detalhe.** PSI com bins recalculados na janela
de produção compara duas escalas diferentes: o número deixa de significar o que a regra de bolso
(`<0,10` estável · `0,10–0,25` investigar · `>0,25` agir) diz que significa. Congelar as bordas é o
que torna a série comparável ao longo do tempo.

🔑 **A `taxa_acima_do_limiar` é o número que dispensa analista.** "PSI 0,30" exige interpretação;
"a fila de retenção encolheu 37%" já é a conversa. Métrica de monitoramento que só existe em
unidade estatística é métrica em que ninguém age.

#### Controles (a disciplina de sempre: verificar nos dois sentidos)

| controle | esperado | medido |
|---|---|---|
| treino × treino | 0 | **0,0000** |
| validação × treino (ruído de partição) | « 0,10 | **0,0128** (pior coluna) |
| `Tenure Months` +12 | agir | **3,55** — demais colunas quietas |
| `Monthly Charges` ×1,15 | agir | **1,46** |
| categoria inédita em ⅓ da janela | agir | **4,37**, com a categoria **nomeada** |
| scores da própria validação | 0 | **0,0000**, fila 1,00× |

**4 sabotagens verificadas reprovando:** bins fechados nas pontas, bordas sem deduplicação, PSI sem
o termo logarítmico, e a quarta checagem desligada.

⚠️ **Uma proteção que o dado real não exercita.** Remover o `np.unique` das bordas **não quebrava
nenhum teste**: as 3 numéricas do repo têm 9 decis distintos cada. O teste que a cobre passou a ser
**sintético e explícito** (coluna com 70% de zeros) — proteção que o dataset não exercita é
proteção que ninguém verifica.

#### 🎯 Achado: o PSI de scores é pouco sensível, e a fila não é

Dobrando o risco de toda a carteira por um fator:

| fator | PSI dos scores | veredito | fila de retenção |
|---|---|---|---|
| ×1,2 | 0,068 | estável | +12% |
| ×1,4 | 0,151 | investigar | +25% |
| ×1,6 | 0,224 | **ainda** investigar | +32% |
| ×2,0 | 0,386 | agir | +42% |

O motivo é estrutural, não um defeito: os bins são **decis**, e transformação monotônica move pouca
massa entre decis — quase todo mundo continua no mesmo lugar da fila, mesmo com todos os números
subindo. É o aviso do 10c no sentido inverso: **nem toda mudança que importa aparece no PSI**.
⇒ O painel não pode ter só PSI; a taxa acima do limiar dispara antes e já vem na unidade da decisão.

#### Efeitos colaterais assumidos

- **O artefato muda:** 8.306 → **10.398 B**, novo `sha256`. Como ele é versionado (exceção nomeada
  da 9f) e o Render tem `autoDeployTrigger: checksPass`, promover **re-implanta o serviço**.
  Verificado que o objeto servido é o mesmo: **PR-AUC `0.6646020518504109`**, idêntico nos 16
  dígitos ao da 9e — só os metadados cresceram.
- **O round-trip do `promover.py` passou a comparar os metadados também.** O anterior comparava só
  as 1.409 probabilidades, e probabilidade não passa por `referencia`: ~2,5 KB que **nada** no
  caminho feliz da API lê, e cujo erro só apareceria na primeira análise de drift.
- **A fixture do `conftest.py` deixou de montar os metadados à mão** e passou a chamar
  `promover.montar_metadados()`. Eram duas construções paralelas do mesmo objeto — e um teste que
  reconstrói o objeto sob teste testa a reconstrução. Mesmo movimento do item 85, quando
  `gate.medir()` foi extraída para que o CI e o teste de caracterização medissem **o mesmo objeto**:
  o teste puxa o desenho.

---

### 10c-bis · Drift fabricado — provar que o detector dispara (18/08/2026)

> **O problema é de calendário, não de estatística.** Drift real leva meses; um Tech Challenge de
> seis semanas nunca vê um, e a Etapa 10 termina declarativa ("eu monitoraria…"). A inversão: em
> vez de esperar, **fabricar** o drift e mostrar o alarme nascer. O que se testa **não é o
> modelo** — é o sistema de vigilância.

    make simular-drift          # ~20 s, três cenários

O script sobe a própria API num subprocesso com o stdout redirecionado, manda uma janela **base**
(400 clientes reais da validação) e a mesma janela **deslocada**, e separa as duas metades do log
**pelo `request_id` que cada resposta devolveu** — que é exatamente para isso que ele existe:
correlacionar resposta e log sem devolver dado pessoal ao cliente.

#### Os três cenários e o que cada um respondeu

| cenário | data drift | prediction drift | fila de retenção |
|---|---|---|---|
| **base envelhecendo** (`tenure` +12) | `Tenure Months` **3,59** (agir) | **0,30** (agir) | **0,63×** — encolhe |
| **reajuste** (`Monthly Charges` ×1,15) | `Monthly Charges` **1,49** (agir) | **0,03** (estável) | 1,01× |
| **plano novo** (categoria inédita) | — | — | **HTTP 422** |

**🔑 As duas famílias se movem de forma independente, e os dois primeiros cenários provam isso em
direções opostas.** No reajuste, a entrada muda muito e a saída quase nada — é o aviso do 10c em
números (*nem todo data drift significa que o modelo piorou*), e um painel que alertasse por PSI de
entrada teria acordado alguém de madrugada por nada.

**🚨 E no envelhecimento o alerta chega vestido de boa notícia.** A saída se desloca (PSI 0,30) e a
fila de retenção **encolhe 37%**, porque tenure alto significa menos churn. Um painel de negócio
olhando só volume comemora — "a campanha vai ligar para menos gente"; um painel de drift pergunta
por quê. É o caso que justifica manter as duas leituras lado a lado.

#### 🎯 O achado que inverteu a premissa: categoria inédita não é drift silencioso aqui

A expectativa — do enunciado e do catálogo de armadilhas — era que o `OneHotEncoder` tratasse a
categoria desconhecida como tudo-zero e **predissesse mesmo assim, com HTTP 200**, com o
monitoramento pegando dias depois via PSI. **Medido: a API devolve 422 e a linha nunca chega ao
modelo.**

O motivo é uma decisão da 9d: o schema é **derivado do artefato**
(`Literal[tuple(ohe.categories_)]`), não escrito à mão como `str`. O contrato HTTP herdou as 25
categorias que o encoder viu, então categoria inédita vira erro de **validação**, na borda.

**Consequência para o desenho do monitoramento:** nesta API, categoria nova **não é evento de
drift — é evento de serviço**. Aparece como pico de 4xx (10b, família "serviço"), não como PSI
subindo, e aparece **na primeira requisição**, não ao fim de uma janela. Vigiar isso com PSI seria
vigiar o lugar errado; um alerta de taxa de erro, que qualquer plataforma dá de graça, é o detector
mais rápido para esta família.

⚠️ **O preço está declarado:** o serviço **recusa** o cliente novo em vez de arriscar um palpite.
É a escolha certa para um ranqueador de campanha (a fila fica menor e ninguém é pontuado errado) e
seria a errada para um serviço obrigado a responder sempre — nesse caso o schema teria de aceitar a
categoria, e o monitoramento é que assumiria a carga.

#### Limitações — declaradas de propósito

1. **É covariate shift** (`P(X)` mudou) e é detectável **por construção**, porque foi construído. O
   script **não** simula concept drift (`P(y|X)`), que é invisível às estatísticas de entrada: o
   objeto que mudou não existe sem `y`.
2. **Detectar não é corrigir.** O alerta abre um caminho (investigar → decidir se retreina → passar
   pelo gate), não fecha um. Pipeline que retreina sozinho ao ver PSI alto é uma máquina de pôr
   modelo pior em produção mais rápido.
3. **A régua é PSI, não p-valor** — com amostra grande KS acusa qualquer coisa. E KS é para
   contínuas; categórica pede PSI ou qui-quadrado.
4. 🔑 **O deslocamento é univariado, e o detector também.** Somar 12 meses a `Tenure Months` sem
   mexer em `Total Charges` cria clientes com dois anos de casa e a fatura acumulada de um —
   combinação que **não existe no mundo**. Medido: `Tenure Months` vai a 3,59 e `Total Charges`
   fica em **0,034** (estável), apesar de metade da história daqueles clientes ter deixado de fazer
   sentido. É a demonstração acidental do ponto do 10c: KS/PSI olham **uma coluna por vez** e são
   cegos à combinação inédita. O upgrade natural, se sobrar tempo, é um `IsolationForest` sobre o
   treino como indicador multivariado ao lado do PSI.
5. ⚠️ **O script não roda contra a API em produção** — e não por falta de endereço. Lá o stdout
   pertence à plataforma e não há como lê-lo de volta. É a terceira camada do "quem lê este
   arquivo?" cobrando o preço na direção contrária: o log que é fácil de emitir é o mesmo que é
   difícil de recuperar.

#### O leitor de logs (`src/monitoring.py`)

`make monitorar LOG=…` transforma o `.jsonl` nas famílias do 10b — latência em **percentis**
(P50/P95/P99, porque a média mente), status e taxa de erro, prediction drift sempre e data drift
quando as features estiverem no log.

🚨 **A primeira coisa que ele faz não é estatística, é identidade.** Conta quantos
`artefato_sha256` distintos há na janela e os confronta com o artefato cujo baseline está sendo
usado — é o que torna aquele campo do log **verificável em vez de decorativo**: PSI alto tem duas
explicações ("a população mudou" e "trocaram o modelo no meio da janela") e sem essa checagem a
segunda é indistinguível da primeira.

Três silêncios que o painel se recusa a produzir, cada um com teste próprio:

- **features ausentes ⇒ "não medido"**, nunca "estável" — não medir e medir zero produzem o mesmo
  silêncio, e só um deles é informação;
- **janela sem predição ⇒ "sem dados"**, nunca PSI 0 — volume zero é sintoma próprio (falha
  upstream), e "sem erro" ≠ "tudo bem";
- **linhas não-JSON são contadas**, não descartadas — é o sintoma de que outro emissor entrou no
  stream (o access log do servidor é o caso conhecido).

**Qualidade preditiva fica de fora, e a ausência é a informação:** acurácia e PR-AUC exigem o
rótulo, e o churn só se confirma ao fim do ciclo de faturamento. É a janela cega do ground truth —
e a razão de o drift ser o que se vigia em tempo real.

---

### 10b · As quatro famílias de métricas, instanciadas em churn (19/08/2026)

> *Alerta sem ação definida é ruído; e limiar sem denominador não escala.*

A forma de cada linha é **sinal → KPI → gatilho → desarme → janela → ação**. As duas
colunas do meio existem porque **um limiar oscila e dois não**: dispara a 5,1%, silencia
a 4,9%, dispara de novo. Sem o limiar de desarme, cada oscilação em torno do gatilho
consome um ciclo de investigação — que é a versão cara da fadiga de alerta.

| família | sinal (KPI) | gatilho | desarme | janela | ação |
|---|---|---|---|---|---|
| **serviço** | p95 de `/v1/predict`, **medido no cliente** | > 1.500 ms | < 800 ms | 3 janelas | investigar rede × app confrontando com o `x-response-time-ms` da própria resposta |
| **serviço** | 5xx ÷ **total de requisições da janela** | > 1% | < 0,2% | 2 janelas | **é bug, não drift** — ler o log pelo `request_id`, não retreinar |
| **serviço** | 4xx ÷ total | > 5% | < 1% | **imediata** | contrato quebrado a montante: categoria nova, campo faltando, sentinela de nulo |
| **drift** | PSI por coluna, bordas congeladas | > 0,25 | < 0,10 | 3 janelas | validar integridade **antes** de culpar o mundo; se for mundo, abrir a decisão de retreino (10d) |
| **drift** | PSI dos `scores` (`P(ŷ)`) | > 0,25 | < 0,10 | 3 janelas | idem, e é o único que roda sem custo de privacidade |
| **drift** | **taxa acima do limiar** (tamanho da fila), base **0,3769** | desvio > ±25% | dentro de ±10% | 3 janelas | conversar com a operação **antes** do PSI: já vem na unidade da decisão |
| **negócio** | volume de predições ÷ mediana das 4 janelas anteriores | < 0,5× | > 0,8× | 2 janelas | falha upstream — *"sem erro" não é "tudo bem"* |
| **qualidade** | PR-AUC no lote reconciliado | < 0,66 | ≥ 0,68 | por lote de rótulo | reavaliar; retreino passa pelo gate |
| **qualidade** | Brier no lote reconciliado | > 0,14 | ≤ 0,13 | por lote de rótulo | recalibrar / re-derivar o limiar |
| **fairness** | disparidade de recall entre grupos | > 10 pp | ≤ 7 pp | por lote de rótulo | Etapa 10.5 — o limite é escrito **antes** de medir |

**Três linhas que ficam vazias de propósito, e a ausência é a informação:** *tráfego*
(RPS) e *saturação* (CPU/memória) não têm base — não há carga real, e limiar inventado
sobre tráfego que não existe é decoração; *taxa de aceitação da campanha* não é
observável no escopo, porque todas as métricas do TC são offline e o dataset não carrega
a conversão da retenção. Um ❌ justificado vale mais que um ✅ inventado.

#### 🔑 A régua do PSI já é uma banda morta — e ninguém a lê assim

`<0,10 estável · 0,10–0,25 investigar · >0,25 agir` é citada em todo material como uma
**escala de severidade**. Ela é melhor que isso: com os dois extremos usados como
gatilho e desarme, a faixa do meio vira exatamente a **zona morta de um termostato**, e
o estado do alerta só muda quando o sinal atravessa a banda inteira. Não foi preciso
inventar um segundo número — ele já estava na regra, sem uso.

🔗 É a mesma histerese da regra 1-SE das Etapas 6 e 7, aplicada ao **tempo** em vez da
complexidade: não trocar de estado por movimento que cabe dentro da variação esperada.

#### 📏 O limiar de drift foi calibrado contra o ruído MEDIDO, não contra a regra de bolso

A prática manda *observar a variabilidade normal antes de fixar o threshold*. Isso
costuma ficar como intenção; aqui há número, e ele veio de graça dos controles da 10a-2:

| medição | PSI | leitura |
|---|---|---|
| treino × treino | 0,0000 | o instrumento não inventa sinal |
| **validação × treino** (só ruído de partição) | **0,0128** | é o piso do que "nada aconteceu" produz |
| `Monthly Charges` ×1,15 | 1,46 | 114× o ruído |
| `Tenure Months` +12 | 3,55 | 277× o ruído |

⇒ O gatilho de 0,25 está a **19,5×** o ruído de partição e o de investigação a **7,8×**.
A folga não é confortável por acidente: é o que separa este limiar de um número copiado.

#### 🚨 Toda taxa tem denominador, e a janela é de VOLUME, não de tempo

`rate(5xx[5m]) > 0.05` — a forma que o material da M04-A07 usa — **não** é "5% das
requisições": é **5xx por segundo**. Como está, dispara com 3 erros por minuto
independentemente do volume, e **piora conforme o produto cresce**. Toda linha da tabela
acima é escrita como razão com o denominador visível.

E a **janela é `n ≥ 400` predições**, não "por dia". Sem carga real, janela temporal é
chute: um dia com 12 requisições produziria decis com dois pontos cada e um PSI que
oscila por conta própria. 400 é o tamanho de janela que o `simulate_drift.py` usa e no
qual os controles acima foram medidos — o limiar e a janela em que ele foi calibrado
viajam juntos, ou o número não significa o que diz.

#### 🚨 O ponto de medição faz parte do limiar

Medido em produção, Brasil → Oregon, 0,1 CPU:

| cronômetro | p50 | p95 |
|---|---|---|
| cliente (`curl`, unitário) | 254 ms | 670 ms |
| **a própria API** (`x-response-time-ms`) | **1,16 ms** | — |

**219× de diferença** na mesma requisição. Um SLA de "p95 < 300 ms" seria violado o
tempo todo medindo do Brasil e nunca medindo de dentro — e as duas leituras estão
certas. Por isso o limiar de 1.500 ms diz **"medido no cliente"**, e a ação manda
confrontar as duas: p95 externo subindo com o interno parado é rede ou plataforma, e
retreinar modelo não conserta latitude. *Latência sem o ponto de medição declarado é
métrica sem piso.*
⚠️ **O cold start fica fora por construção** — o plano gratuito dorme em 15 min e a
primeira requisição custa ~30 s. Contá-lo no p95 faria o alerta disparar por causa da
característica que torna o plano gratuito.

#### Quem age — e por que a coluna existe mesmo solo

O projeto é individual, então todos os papéis são a mesma pessoa. Escrever o RACI mesmo
assim não é formalidade: **alerta sem dono é o alerta que ninguém desliga e ninguém
atende**. No mundo real, a família de *serviço* é do time de plataforma (dispara paging),
a de *drift* e *qualidade* é do dono do modelo (abre investigação, não incidente), e a de
*negócio* é de quem opera a campanha. A distinção prática que sobrevive ao solo é o
**canal**: 5xx acorda alguém; PSI abre um item de backlog com prazo. Detalhamento na
Etapa 10.5.

---

### 10d · Política de retreino (19/08/2026)

> *O gargalo nunca foi saber que precisa atualizar — é o custo de atualizar. E a decisão
> de retreinar é uma decisão de negócio disfarçada de tarefa técnica.*

**Pressuposto declarado, porque a resposta não está no dataset:** o Telco é um retrato
sem eixo temporal (limitação registrada na Etapa 0), então **não temos como medir** em
quantos dias o rótulo se confirma. Assume-se **~60 dias** — dois ciclos de faturamento
mais carência — e a política inteira é dimensionada sobre esse número. Se ele estiver
errado, o que muda é a frequência, não a estrutura.

| gatilho | vale aqui? | ponto cego | decisão |
|---|---|---|---|
| **agendado** (trimestral) | ✅ **piso de segurança** | retreinar à toa gasta recurso e pode injetar ruído | adotado — com o gate barrando o modelo pior |
| **degradação de performance** | ⚠️ **inútil sozinho** | reage com 60 dias de atraso; até lá o dano já ocorreu | mantido como confirmação, nunca como detector |
| **data / prediction drift** | ✅ **é o detector real** | falso positivo | adotado **com banda morta + 3 janelas** (10b) |
| **on demand** | ✅ | depende de alguém perceber | reajuste tarifário anunciado, mudança regulatória, viés achado na 10.5 |
| **online / contínuo** | ❌ | instabilidade, e exige rótulo rápido | descartado: a janela cega o torna impossível aqui |

**O padrão é a combinação:** agendado como piso, drift como reação. E a ordem importa —
**drift não dispara retreino, dispara investigação**. Pipeline que retreina sozinho ao
ver PSI alto é uma máquina de pôr modelo pior em produção mais rápido.

#### 🚨 A janela cega é o que estrutura a política, não um detalhe

Concept drift é mudança em `P(y|X)` e **só é detectável com rótulo** — não é limitação de
ferramenta, é que o objeto que mudou não existe sem `y`. Com ~60 dias de atraso, um
gatilho por performance é um retrovisor: quando ele acender, dois meses de campanha já
foram priorizados com um modelo pior. É exatamente por isso que o drift de `P(X)` e de
`P(ŷ)`, que roda em tempo real e sem rótulo, é o **proxy** — e é também por isso que ele
não pode ser tratado como prova: ele responde *"quem chega mudou?"*, nunca *"a regra
mudou?"*.

#### 🚨 O log de inferência NÃO é fonte de rótulo

*"Retreine com os dados novos capturados"* soa razoável e é impossível aqui: o log
guarda a **predição**, não o desfecho. Retreinar com ele é self-training degenerativo —
o erro da versão anterior vira alvo da próxima e o modelo aprende a concordar consigo
mesmo, com as métricas melhorando enquanto a realidade se afasta.

⇒ O rótulo vem da base transacional ao fim do ciclo, reconciliado pelo **`request_id`**
que a resposta devolve e o log grava. Sem essa chave não existe retreino supervisionado,
só a ilusão dele — e é a razão de o `request_id` ter sido gerado no servidor desde a 9d,
antes de existir log para escrevê-lo.

#### 🚨 E o rótulo reconciliado vem contaminado pela própria campanha

O modelo marca o cliente, a operação liga e dá desconto, **o cliente fica**. O desfecho
registrado é "não churnou" e o modelo é **penalizado por ter acertado**. Pior: os dados
do próximo treino já não descrevem o mundo sem intervenção.

⇒ **Grupo de controle**: uma fatia dos preditos como risco que *não* recebe a campanha.
É a única forma de medir o modelo limpo **e** o valor real da campanha — que é a pergunta
que paga a conta. Fora do escopo de implementação (não há campanha real), mas a política
o exige e a Etapa 11 declara.

#### 🔑 O CI já é a trava do retreino — e não foi projetado para isso

O teste de caracterização exige `|PR-AUC − 0,6646| ≤ 1e-4`. Um modelo retreinado
**sempre** viola isso, mesmo melhor: a menor mudança já medida (remover `Contract`)
desloca 0,0019, dezenove vezes a tolerância. Logo:

> **é impossível promover um retreino sem alguém editar `PR_AUC_REF` e `BRIER_REF` no
> mesmo commit** — e isso é um diff, revisável, com data e autor.

Aquilo que a Etapa 9.5 escolheu por escrito — **Continuous Delivery, não Deployment**,
com humano no último passo — já estava implementado em código antes de ter esse nome. A
combinação é o mecanismo completo: o **gate de dois eixos** (`PR-AUC ≥ 0,66` **e**
`Brier ≤ 0,14`) impede promover um modelo pior; o **teste de caracterização** impede
promover um modelo *diferente* **em silêncio**. Piso e contrato fazem perguntas
diferentes, e o retreino precisa das duas respostas.

#### O que fazer quando o alerta acende — e retreinar não é a primeira opção

1. **Integridade antes de drift.** *Mudou o mundo ou mudou o meu pipeline?* Um join que
   duplicou linhas, uma unidade que trocou de escala, um campo que passou a vir nulo:
   retreinar sobre dado quebrado **assa o bug dentro do modelo novo**. O painel já ajuda —
   o `simulate_drift.py` mediu que `Tenure Months` a 3,59 com `Total Charges` em 0,034 é
   a assinatura de uma **coluna** deslocada, não de uma população que envelheceu.
2. **Identidade antes de estatística.** O `monitoring.py` conta os `artefato_sha256`
   distintos na janela: PSI alto tem duas explicações, e "trocaram o modelo no meio da
   janela" é a que se descarta primeiro porque custa uma contagem.
3. **Só então a decisão de retreinar**, que passa pelo gate e pelo diff acima.

**Contingência (degradar com graça):** se o artefato não carregar, o processo **morre na
inicialização** por decisão — não existe modo degradado servindo predição de origem
desconhecida. O fallback de negócio não é outro modelo: é a fila ordenada pela regra
simples que a Etapa 6 já validou como forte (contrato mês-a-mês), no espírito do Uber
trocando o modelo de ETA por uma fórmula de distância. Pior que o modelo, muito melhor
que nada, e explicável por telefone.

---

### 10e · Rollback e troca segura de versão (19/08/2026)

> *A segurança do rollback é o que encoraja atualizar com frequência. Quem não sabe
> voltar atrás congela — e modelo congelado apodrece em produção.*

**O estado de partida é bom e foi construído sem este objetivo:** o artefato é versionado
no Git (exceção nomeada de 8,1 KB da 9f), o `/health` declara o `artefato_sha256`
carregado, e o Render implanta atrelado ao CI (`autoDeployTrigger: checksPass`). Três
versões do campeão são recuperáveis por Git; os nove candidatos das Etapas 6–8 seguem no
`mlruns/`. *Backup off-site do modelo versionado* sai de graça.

#### O procedimento, em duas camadas — e a ordem não é opcional

| # | passo | mecanismo | tempo |
|---|---|---|---|
| 1 | **parar o sangramento** | painel do Render → *Rollback* para o deploy anterior (re-implanta a **imagem** inteira) | minutos |
| 2 | **conferir a identidade** | `GET /health` → `artefato_sha256` **é o esperado** | segundos |
| 3 | **tornar definitivo** | `git revert` do commit que promoveu (ou `git checkout <sha> -- models/campeao.joblib`) + `make ci` + push | uma hora |
| 4 | **reconferir** | `/health` de novo, depois que o `checksPass` re-implantar | segundos |
| 5 | **registrar** | linha no decision log: o que caiu, qual sha voltou, por quê | — |

#### 🚨 Rollback só no painel é reversão que o próximo push desfaz

Esta é a armadilha específica desta arquitetura, e ela é consequência direta de uma
decisão que estava certa. Como o **artefato viaja no repositório** e o Render implanta a
cada commit aprovado pelo CI, o passo 1 sozinho é temporário: **o próximo commit de
qualquer natureza — um ajuste de README — reconstrói a imagem a partir da `main` e traz
o artefato ruim de volta**, com o CI verde e sem nada acusando.

🔑 *Onde o artefato é versionado junto do código, o rollback também tem de ser versionado
junto do código.* O passo 1 é anestesia; o passo 3 é a cirurgia. Fazer só o primeiro é
programar a recaída para a próxima sexta-feira.

#### 🎯 A verificação de identidade é o que separa rollback de esperança

O passo 2 não é zelo: reverter e **não conferir** é exatamente o cenário da Knight
Capital — sete servidores com o código novo e um com o velho, e nada no sistema
perguntando qual estava servindo. Aqui a pergunta custa um `curl`, porque o `/health`
foi construído na 9d para respondê-la:

    curl -s https://tc-churn-api.onrender.com/health | jq '.artefato_sha256, .versao_modelo'

⚠️ **E há um preço declarado:** só **um** artefato entra na imagem (o `.dockerignore` é
allowlist e a exceção do `.gitignore` nomeia um arquivo). Logo `TC_ARTEFATO` **não**
permite trocar de modelo sem reconstruir — o rollback custa um deploy, não uma variável
de ambiente. Foi o preço de não ter registry acessível ao build no plano gratuito, e a
evolução natural é o Model Registry com alias `Production`, onde reverter é trocar um
ponteiro.

#### As estratégias de release: qual está feita, quais não cabem e por quê

| estratégia | mecanismo | mede antes de decidir? | estado |
|---|---|---|---|
| **endpoint versionado** (`/v1` → `/v2`) | o cliente escolhe a URL | não — depende de o cliente migrar | ✅ **já feito na 9d**, antes de ter esse nome |
| **blue-green** | dois ambientes idênticos, comuta o roteamento | **não** — o Green nunca viu tráfego real até ser 100% | ❌ dobra o custo; o plano é gratuito e único |
| **canary** | 5% → 20% → 50% → 100% | ✅ **o único que mede** | ❌ exige roteador de tráfego |

🔑 **A distinção que vale a linha:** blue-green tem risco **zero antes** da comutação e
**total depois**; canary tem risco pequeno e contínuo, e é o único que mede antes de
decidir. E o corolário que amarra na Etapa 7: canary é teste A/B em produção, 5% de
tráfego é uma **amostra**, amostra tem incerteza ⇒ promover pelo resultado da primeira
janela é **o mesmo pecado do pico da grade**. *Canary sem teste de significância é 1% de
tráfego produzindo uma decisão de 100%.*

⚠️ **O que não foi feito, dito com todas as letras:** o procedimento acima está escrito e
os mecanismos existem, mas **um rollback real nunca foi executado** — não há versão ruim
em produção para reverter, e fabricar uma para testar custaria um ciclo de deploy sem
ninguém do outro lado para observar. É a linha 🟡 do checklist de manutenção: *rollback
declarado, não testado*. Declará-lo assim vale mais que um ✅ que a banca não pode
verificar — e é o mesmo critério pelo qual o gate, o healthcheck e o baseline **foram**
verificados reprovando: quando o teste é possível, ele é obrigatório; quando não é, a
ausência é escrita.

---

## 5g. Etapa 10.5 — governança, fairness e conformidade

### ⚠️ PRÉ-REGISTRO — escrito ANTES de rodar a auditoria (19/08/2026)

> Este bloco foi commitado **antes** de a primeira linha de `MetricFrame` existir, e o `git log`
> é a prova. O motivo é o de sempre nas etapas anteriores (7, 8): um limite escrito **depois** do
> resultado não é limite, é negociação — *"12 pontos tá ruim? ah, dá para viver com isso"*. A
> diferença aqui é que o objeto negociado não é uma métrica, são pessoas.

#### 1. A métrica que carrega o dano — e por que não é a acurácia

Desagregar a métrica errada produz uma tabela tranquilizadora sobre um problema que continua lá.
Em churn o dano **não** mora na acurácia: mora no **falso negativo**. O cliente que ia cancelar e
o modelo não marcou **não entra na fila de retenção**, não recebe oferta, e vai embora — e o
prejuízo nunca vira linha de planilha, porque ninguém registra a campanha que não foi feita.

⇒ A métrica pré-registrada é o **recall por grupo** (equivalentemente, a **FNR**: `FNR = 1 −
recall`). `selection_rate` entra como **diagnóstico**, não como critério — ver o item 3.

#### 2. O limite, escrito antes: **disparidade de recall ≤ 10 pontos percentuais**

| item | valor |
|---|---|
| métrica | recall (por grupo) |
| **limite de disparidade** (max − min) | **≤ 10 pp** |
| desarme (10b) | ≤ 7 pp |
| partição | **validação** (o teste segue intocado até a Etapa 11) |
| limiar | **0,29 — o de operação**, não o 0,5 implícito |
| atributos auditados | `Gender`, `Senior Citizen`, `Partner`, `Dependents` |
| exploratório (sem gate) | `Zip Code`/`City` agregados, se houver grupo com n suficiente |

⚠️ **O limiar da auditoria é o de operação, e isto não é detalhe.** Auditar em 0,5 mediria um
modelo que este projeto não usa: a fila real é cortada em 0,29, e é nesse ponto que se decide quem
recebe campanha. Uma auditoria no limiar errado é tecnicamente correta e operacionalmente ficção.

⚠️ **Quem decidiu os 10 pp.** O projeto é solo, então quem escreveu o número é quem o cumpre — o
que é exatamente a situação que a literatura desaconselha. O registro honesto é este: **10 pp é uma
decisão de negócio e jurídica que um engenheiro não deveria tomar sozinho**, adotada aqui por
ausência de contraparte, com o valor declarado antes da medição para que ao menos não seja
*post hoc*. Num contexto real, quem assina é o dono do produto com o jurídico. Está no RACI.

#### 3. A definição de fairness escolhida — e a que foi recusada, com o motivo

**Escolhida: paridade de erro** (família *equalized odds*) — mesma taxa de falso negativo entre
grupos. **Recusada: paridade demográfica** (mesma taxa de seleção entre grupos).

O motivo é substantivo, não estético. A prevalência real de churn **difere entre grupos** nesta
base. Exigir a mesma taxa de seleção forçaria o modelo a **subatender o grupo de maior risco** para
igualar a estatística — ou seja, deixaria de ligar para quem tem mais chance de cancelar, em nome
da paridade. Isso prejudicaria justamente quem a métrica pretendia proteger.

🔑 **E a escolha é obrigatória, não opcional:** paridade demográfica e equalized odds são
**matematicamente incompatíveis** quando a prevalência difere entre grupos (Kleinberg et al.;
Chouldechova). Não é limitação de ferramenta, é aritmética — logo *alguma* delas tem de ser
abandonada, e a única resposta errada é não escolher.

⇒ `selection_rate` **será medido e reportado**, porque é o diagnóstico que explica a disparidade de
recall. Mas um `selection_rate` maior para um grupo **não** é achado de viés se a prevalência dele
for maior: é o modelo funcionando.

#### 4. Previsões falseáveis — o que espero encontrar (escrito antes de ver)

1. **`Gender`: disparidade < 3 pp.** Gênero é notoriamente sem sinal em churn de telecom, e a
   Etapa 5 mediu que remover as demográficas custa **zero** de PR-AUC. Se der grande, desconfio
   primeiro de tamanho de grupo, não de viés.
2. **`Senior Citizen`: `selection_rate` claramente MAIOR no grupo idoso, e disparidade de recall
   pequena.** É o caso em que as duas definições de fairness divergem visivelmente — e é por isso
   que ele é o atributo principal da auditoria.
3. **`Dependents` e `Partner` são os candidatos reais a estourar**, porque correlacionam com
   `Contract` e `Tenure Months`, que são as features fortes. Se houver problema, está aqui.
4. **Nenhum grupo estoura os 10 pp.** Modelo linear, atributo sensível **dentro** das features (a
   Etapa 5 manteve de propósito), sem reponderação: espero disparidade moderada. ⚠️ Se eu estiver
   certo, o resultado **não** prova que o modelo é justo — prova que ele não é injusto *nesta
   métrica, neste limiar, nestes quatro atributos*. A distinção vai escrita no Model Card.

#### 5. O que acontece se estourar — decidido antes, para não virar racionalização

Em ordem, e **nenhuma delas é "remover a coluna"**:
1. **Diagnosticar**: é disparidade de erro ou é diferença de prevalência? `selection_rate` e
   prevalência por grupo respondem.
2. **Re-derivar o limiar por grupo** — a saída tecnicamente honesta, e a que muda a decisão
   operacional (a fila) sem mexer no modelo.
3. **Aceitar declarando**, com o número no Model Card e a justificativa. ⚠️ Só é legítimo se a
   decisão tiver dono; solo, o dono sou eu, e isso está dito.
4. **Reponderar/mitigar** (`fairlearn.reductions`) — e aqui o trade-off é pré-registrado: **se a
   mitigação derrubar o PR-AUC global, isso é resultado, não fracasso**, e os dois números vão para
   a documentação. Um modelo 0,01 pior que atende os grupos de forma equilibrada pode ser a escolha
   certa; o que não é legítimo é medir os dois e reportar só o que convém.

🚨 **O que NÃO será feito, e é o reflexo mais comum:** remover `Gender`/`Senior Citizen` das
features. Isso é *fairness through unawareness* — o modelo reconstrói o atributo pelos proxies
(`Contract`, `Tenure Months`, `Monthly Charges`) e o único efeito garantido é **destruir a
capacidade de medir a disparidade**. Ficaria um modelo igualmente enviesado e cego para provar o
contrário. A Etapa 5 já registrou que mantê-los custa **zero** de PR-AUC — a decisão de mantê-los
foi tomada lá, para poder ser auditada aqui.

#### 6. O gate — porque Model Card sem `assert` é marketing

O limite dos 10 pp vira **teste no CI** (`test_fairness_gate`), ao lado do gate de dois eixos e do
teste de caracterização. O critério da casa é o mesmo desde a Etapa 9.5: **um controle que nunca
falhou não foi testado, foi presumido** — então ele entra **verificado reprovando** (baixar o
limite artificialmente ⇒ vermelho ⇒ reverter).

⇒ Depois da auditoria, este bloco ganha a seção de **resultados** logo abaixo, com o placar do
pré-registro (quantas das 4 previsões se confirmaram), no mesmo formato das Etapas 7 e 8.

### Resultado da auditoria (19/08/2026) — três das quatro previsões erradas

    make auditar

| atributo | grupo pior | recall | recall do outro grupo | disparidade | limite (10 pp) |
|---|---|---|---|---|---|
| `Gender` | Male | 0,7306 | 0,8122 | **8,16 pp** | ✅ |
| `Senior Citizen` | **No** | 0,7374 | 0,8646 | **12,72 pp** | ❌ |
| `Partner` | Yes | 0,7105 | 0,8108 | **10,03 pp** | ❌ |
| `Dependents` | **Yes** | **0,2174** | 0,8063 | 🚨 **58,89 pp** | ❌ |

**A leitura de negócio:** de cada 100 clientes com dependentes que iam cancelar,
**o modelo marca 22**. Os outros 78 não entram na fila e não recebem campanha —
o dano que a acurácia global (0,77 de recall) esconde inteiro.

#### Placar do pré-registro: 1 acerto, 3 erros

| # | previsão | resultado |
|---|---|---|
| 1 | `Gender` < 3 pp | ❌ **errada** — 8,16 pp (embora os IC95 se sobreponham: [0,748; 0,866] × [0,662; 0,792], ou seja, não é distinguível de ruído) |
| 2 | `Senior Citizen` com `selection_rate` bem maior e recall parecido | 🟡 **metade** — a seleção é 34 pp maior (0,66 × 0,32), como previsto, mas o recall diverge 12,72 pp |
| 3 | `Dependents`/`Partner` são os candidatos reais | ✅ **certa, e subestimada** — previ "candidatos a estourar", veio 58,89 pp |
| 4 | nenhum grupo estoura os 10 pp | ❌ **errada** — três estouram |

🔑 **O placar ruim é o argumento.** Se as quatro previsões tivessem batido, a
auditoria teria confirmado o que eu já achava e não teria produzido informação
nenhuma. Errar três significa que o instrumento mediu algo que a intuição não
alcançava — que é a única razão de auditar em vez de afirmar.

#### O diagnóstico, antes da decisão

**1. A causa é aritmética, não um defeito.** Prevalência de churn: **7,01%** no
grupo com dependentes contra **32,47%** no outro. Limiar global de 0,29 aplicado
a um grupo de baixo risco marca pouca gente **por construção** — a
`selection_rate` do grupo é 3,66%. É o **teorema da impossibilidade**
(Kleinberg; Chouldechova) em forma concreta: com prevalências diferentes,
calibração e paridade de erro não podem valer ao mesmo tempo.

🔑 **E a explicabilidade já tinha anunciado isso**, na tabela de odds ratios que
existia antes da auditoria: `Dependents_Yes` = **0,349**, o segundo menor
coeficiente do modelo. O modelo aprendeu — corretamente — que esse grupo cancela
pouco. *A disparidade não é o modelo errando; é o modelo acertando sobre um grupo
cuja base é diferente.* O que não o isenta: o efeito sobre a pessoa é o mesmo.

**2. O pior achado é o de menor amostra — e o IC não o salva.** `Dependents=Yes`
tem **23 churners** na validação; IC95 de Clopper-Pearson = **[0,075; 0,437]**,
largura 0,362. ⚠️ Mas o **limite superior continua abaixo** de 0,8063: mesmo no
cenário mais favorável, o grupo é pior atendido. Reportar o IC é o que impede
duas leituras erradas simétricas — precisão falsa ("recall é 0,2174") e descarte
fácil ("são só 23 casos, não conta").

#### As três saídas, medidas — e a que foi adotada

| saída | disparidade | PR-AUC | custo/ciclo | fila |
|---|---|---|---|---|
| **manter e declarar (adotada)** | 58,89 pp | **0,6646** | R$ 31.750 | 37,69% |
| limiar por grupo (0,05 no grupo `Yes`) | **6,33 pp** | 0,6646 | R$ 34.668 (**+R$ 2.918**) | 45,42% (+109 clientes) |
| remover `Dependents` das features | 13,20 pp | **0,6427** | — | — |

**Decisão (Luca, 19/08/2026): manter o modelo e declarar a disparidade.**
O motivo registrado: mitigar por limiar por grupo é **tratar pessoas de forma
explicitamente diferente com base em atributo protegido**. Isso é defensável como
ação afirmativa e é tecnicamente a melhor saída medida — mas é uma decisão que
exige dono jurídico, e este projeto não tem. *Registrar o número, o preço da
correção (R$ 2.918 por ciclo, +9,2%) e quem deveria decidir é mais honesto que
aplicar uma correção que ninguém autorizou.*

🚨 **Achado que corrige o catálogo pela metade: `unawareness` ATENUA.** A regra
que o repo carregava era *"remover a coluna não remove o viés"*. Medido, o
resultado é mais matizado e mais útil: a disparidade cai de 58,89 para **13,20 pp**
— o viés sobrevive nos proxies (`Contract`, `Tenure Months`), mas **enfraquece**.
O que mata a opção não é ela ser inócua, é o preço: **PR-AUC 0,6427, abaixo do
piso de 0,66** ⇒ o modelo nem seria promovível. A saída "mais justa" é reprovada
pelo outro eixo do gate, e isso vale mais que o slogan.
⚠️ Ressalva geral que vai junto: **um modelo pior tende a parecer mais justo** —
no limite, um modelo aleatório tem disparidade zero. Métrica de fairness nunca é
lida sozinha.

#### 🚨 O gate de fairness NÃO barra o limite — e essa é a decisão, não um descuido

A tentação era `assert disparidade <= 0.10` no CI, como a skill sugere. Aplicado
aqui, ele **reprovaria o modelo em produção a cada push**, e a saída óbvia seria
afrouxar o número até o verde — que é exatamente a negociação *post hoc* que o
pré-registro existe para impedir. Um gate que nasce sendo violado não protege
nada; ele treina o time a editar o limite.

✅ **A saída é a divisão que o repo já usa para desempenho:** *piso* pergunta *"é
bom o bastante?"*, *contrato* pergunta *"é o mesmo?"*. Aqui o "bom o bastante" foi
**conscientemente não atingido**, e o que resta é o contrato:

| teste | o que protege |
|---|---|
| `test_caracterizacao_da_disparidade` | os 4 números não mudam (±1e-4) sem um diff que alguém revisa — **para pior ou para melhor** |
| `test_auditoria_usa_o_limiar_de_OPERACAO` | auditar em 0,5 mediria um modelo que o projeto não usa |
| `test_grupo_pequeno_e_marcado_como_incerto` | 23 churners aparecem como 23, e o IC é largo de verdade |
| `test_o_grupo_pior_e_identificado…` | disparidade **a favor** do grupo protegido não vira alarme falso (`Senior Citizen`) |
| `test_unawareness_NAO_resolve_e_ainda_reprova_o_gate` | a alternativa descartada continua medida, não vira folclore |
| `test_atributos_sensiveis_continuam_nas_features` | impede que alguém remova as demográficas "para não discriminar" e apague a auditoria |
| `test_campeao_auditado_e_o_mesmo_do_gate` | a auditoria não caracteriza um terceiro modelo construído com o mesmo código |

⚠️ **E a limitação vai escrita nos dois lugares** (aqui e no Model Card): enquanto
a decisão for "aceitar", **nada no CI impede o modelo de continuar a 58,89 pp**.
Quem carrega o compromisso é o Model Card — por isso o número está na primeira
tela dele, e não num apêndice.

#### Verificação (a disciplina de sempre: reprovando)

7 testes novos. Três sabotagens, cada uma derrubando o que devia:

| sabotagem | caiu |
|---|---|
| limiar da auditoria 0,29 → 0,5 | 4 testes (`Dependents` vai a 0,7681) |
| `Dependents` fora de `config.CAT_DISPONIVEIS` | `test_unawareness…` e `test_atributos_sensiveis…` |
| `MIN_CHURNERS_CONFIAVEL` 30 → 10 | `test_grupo_pequeno_e_marcado_como_incerto` |

⚠️ **Erro de procedimento cometido no caminho, e vale o registro:** `git checkout
src/fairness.py` **não reverteu** a primeira sabotagem, porque o arquivo ainda era
*untracked* — e a terceira rodou com a primeira ainda aplicada, produzindo 4
falhas onde devia haver 1. 🔑 *O mecanismo de desfazer só funciona sobre o que o
sistema já conhece* — mesma família de "quem lê este arquivo, e ele lê mesmo?",
agora do lado do `git`. Refeito com cópia de segurança explícita.

#### `MODEL_CARD.md` — o que o diferencia de um folheto

Na raiz do repositório, versionado. Três propriedades que a skill exige e que a
maioria dos cards não tem: **usos NÃO pretendidos** enumerados (6 itens, incluindo
"não usar como evidência sobre um indivíduo"), **limite numérico de disparidade**
escrito antes da medição, e **métricas com o piso ao lado** — nenhuma sozinha.
E a seção 7 abre com o resultado ruim: um card que só publica o que favorece o
modelo não governa nada.

#### Conformidade — o que ficou declarado

- **Base legal:** legítimo interesse, **assumido** (num contexto real, quem valida
  é o jurídico — e isso está dito, em vez de fingir que a questão não existe).
- **Minimização:** `CustomerID` fora, geográficas fora (cardinalidade **e** proxy
  de renda/raça), 6 colunas removidas por ablação na Etapa 5.
- **Art. 20 (revisão):** predição é apoio; a resposta traz score, decisão **e** o
  limiar, para que quem opera possa discordar do corte.
- **Art. 20 (explicação):** a seção 6 do card — e aqui o modelo linear paga um
  dividendo que não estava no plano: a explicação **é** o modelo, sem SHAP.
- **AI Act:** churn com ação comercial **não** é alto risco (ao contrário de
  crédito e saúde). Citado para registrar que o enquadramento foi **verificado**,
  não presumido.
- **RACI:** escrito mesmo solo, e a linha que importa é *"aceita a disparidade:
  dono do produto + jurídico"*, com o registro de que aqui foi uma pessoa só.
---

## 6. Decisão do modelo final

*(preenchida ao fim da Etapa 8 — a fase de modelagem está fechada; o teste segue intocado até a
Etapa 11.)*

- **Modelo escolhido:** **Regressão Logística** nos defaults do scikit-learn (`C=1,0`, `penalty` L2
  — que são decisão da biblioteca, e por isso estão escritas aqui), 13 features, dentro de um
  `Pipeline` com imputação, `OneHotEncoder` sem `drop` e `StandardScaler`.
  **PR-AUC 0,6646 na validação** (piso da métrica: 0,2654), recall@10% 0,278 (73,8% do teto
  estrutural), Brier 0,1339, limiar de operação **0,29**, custo do erro R$ 31.750 por ciclo contra
  R$ 72.556 de não fazer nada.

- **Concorrentes e seus números** (validação, mesmo protocolo, teste intocado):

  | Modelo | Etapa | PR-AUC | Brier | custo/ciclo | recall@10% |
  |---|---|---|---|---|---|
  | **LogReg (escolhido)** | 3 | **0,6646** | 0,1339 | R$ 31.750 | 0,278 |
  | LogReg 1-SE (`C=0,1`, L1) | 7 | 0,6620 | 0,1342 | R$ 31.326 | 0,283 |
  | **MLP `(32,)` PyTorch** | 8 | 0,6615 | 0,1344 | R$ 31.271 | 0,278 |
  | MLP 1-SE `()` | 8 | 0,6600 | 0,1354 | R$ 31.499 | 0,278 |
  | HGB tunado | 7 | 0,6589 | 0,1351 | R$ 31.604 | 0,281 |
  | Random Forest regularizada | 6 | — (eliminada por dominância) | | | |

  **Seis candidatos em 0,0057 de PR-AUC**, contra desvio entre folds de 0,019–0,024. Não é
  indecisão: é a medição de que a escolha se desloca para critérios não-métricos.

- **Por que este, além da métrica:**
  1. **Regra 1-SE** (Hastie et al., 2009): o mais simples dentro de 1 erro padrão do melhor. Ele
     entra no envelope nas **três** leituras da dispersão, e a regra foi aplicada — e pré-registrada
     — antes de os resultados existirem.
  2. **Interpretabilidade que vira ação:** 13 coeficientes com tabela de odds ratios contra 961
     parâmetros da rede. O time de retenção precisa saber *por que* o cliente está na fila, e a
     LGPD Art. 20 dá direito a explicação — SHAP sobre a rede seria uma reconstrução do que a
     LogReg entrega direto.
  3. **Custo operacional:** sem `torch` em produção, latência menor, artefato menor, menos
     superfície de falha na API.
  4. **Precedente citável:** Netflix Prize — o ensemble vencedor de US$ 1M melhorou o RMSE em
     10,05% e **nunca foi implantado**, porque o ganho marginal não pagou a complexidade. Aqui o
     ganho marginal é de sinal **negativo**.
  5. ⚠️ **O que pesa contra, registrado:** a rede tem custo em reais R$ 479 menor por ciclo. A
     hierarquia de métricas da Etapa 0 (PR-AUC seleciona · recall@k reporta · custo decide o limiar)
     foi declarada antes e é o que decide — mas o número fica no backlog em vez de ser omitido.

- **O MLP superou os modelos clássicos?** ⬜ sim · ☑️ **não** — e o "não" tem quatro medições
  independentes por trás, todas com o mesmo diagnóstico: *o sinal do Telco é essencialmente linear
  no logit, e capacidade adicional não encontra o que fazer com a liberdade que ganha.* A rede foi
  implementada, avaliada sob protocolo idêntico ao dos demais candidatos e reportada com o número
  que deu. A literatura de dados tabulares — **inclusive o material desta disciplina** (M02-A04,
  pág. 23) — prevê exatamente isso, e a previsão estava escrita no repositório antes da primeira
  execução.

---

## 6b. O conjunto de teste — a leitura única (19/08/2026)

O teste foi particionado na Etapa 2 e **não foi tocado nenhuma vez** entre 10/08 e 19/08:
features, algoritmo, hiperparâmetros, arquitetura da rede, limiar de operação e gate do CI
saíram todos da validação. Esta seção é a única leitura, feita com o modelo já escolhido e
sobre o **artefato promovido** (`b8109cce…`) — o objeto que a API serve, não um irmão
retreinado com o mesmo código.

```
PR-AUC          : 0.6496   (piso 0.2654 = modelo sem informação; IC95 [0.5960; 0.7016])
                  validação 0.6646  ⇒  gap val−teste +0.0150
ROC-AUC         : 0.8495   (piso 0,5000)          Brier : 0.1352  (validação 0.1339)
recall@10%      : 0.286   (75,9% do teto estrutural, lift 2,86×)
recall@20%      : 0.519   (68,8% do teto estrutural, lift 2,59×)
limiar 0,29     : precisão 0.5287 · recall 0.7647 · F1 0.6251
                  TP 286 · FP 255 · FN 88 · TN 780
custo do erro   : R$ 32.882 (modelo) × R$ 72.556 (não abordar ninguém) × R$ 64.170 (abordar todos)
```

Registro em `docs/resultado-teste-final.json`; figuras em `docs/figuras/`.

### 🎯 O achado: o IC95 é 18,5× maior que a distância entre os seis candidatos

O intervalo de bootstrap da PR-AUC tem **0,1056** de largura. Os seis finalistas das Etapas
6-8 cabiam em **0,0057**.

> 🔑 **O empate técnico nunca foi indecisão de método — é a resolução da amostra.** Com 1.409
> linhas e prevalência de 26,5%, **nenhum** conjunto de teste deste tamanho poderia desempatar
> LogReg, HGB e MLP: a diferença entre eles é dezoito vezes menor que a incerteza da própria
> medição. Reportar 0,6496 sem o intervalo sugeriria uma precisão de quatro casas que o
> tamanho da amostra não sustenta.

É o mesmo instrumento do IC binomial da auditoria de fairness (§5g), aplicado à métrica
principal: ele impede as **duas** leituras erradas simétricas — precisão falsa (*"o modelo
tem 0,6496"*) e ceticismo fácil (*"caiu, então piorou"*).

### O gap de 0,0150 — o que ele é, e o que ele não é

As duas partições têm o **mesmo tamanho** (1.409) e a **mesma prevalência** (26,54%,
estratificada), então a diferença não vem de composição. Duas causas se somam, e nenhuma
delas é degradação do modelo:

1. **Sorteio.** ±0,05 de IC95 sobre um número absorve 0,0150 sem esforço.
2. **Viés de seleção**, já medido neste projeto com grupo de controle na Etapa 7: o campeão
   foi eleito **por ser o melhor na validação** entre seis candidatos separados por 0,0057 —
   e o vencedor de uma disputa dentro do ruído tende a cair quando muda o conjunto. É a mesma
   mecânica que lá produziu um excedente de −0,0126 para uma grade de 40 candidatos.

🔑 **O que se pode afirmar:** *o desempenho em dados nunca vistos é compatível com o medido na
validação*. O que **não** se pode afirmar é qual das duas causas pesou mais — para isso seria
preciso um segundo conjunto de teste, que não existe. *Atribuir a uma causa a diferença que
tem duas* é armadilha catalogada neste projeto desde a Etapa 8; ela vale aqui também.

### 🚨 A métrica agregada caiu e a operacional SUBIU

| | validação | teste |
|---|---|---|
| PR-AUC (seleciona) | 0,6646 | **0,6496** ↓ |
| recall@10% (reporta o valor operacional) | 0,278 (73,8% do teto) | **0,286** (75,9%) ↑ |
| recall@20% | 0,516 | **0,519** ↑ |
| ROC-AUC | 0,8472 | **0,8495** ↑ |

A hierarquia de três níveis declarada na Etapa 0 existe exatamente para este caso: **PR-AUC é
uma integral sobre todos os limiares, inclusive sobre a faixa em que a campanha nunca vai
operar**; `recall@k` mede o ponto onde a decisão acontece. O nível que paga a conta não seguiu
o nível que seleciona — e a campanha de 10% da base captura **mais** churners no teste do que
capturava na validação.

⚠️ Nenhuma das duas variações é grande o bastante para ser lida como efeito: as duas cabem no
IC. O que a tabela demonstra não é "melhorou", é **que as perguntas são diferentes**.

### 🚨 O piso do gate é 0,66 e o teste deu 0,6496 — dito com todas as letras

O gate do CI reprovaria este número, se fosse aplicado a ele. Não é, e a razão está escrita
desde a Etapa 2: **o gate mede a validação**, porque um gate no teste toma uma decisão de
promoção a cada push e converte o teste em validação depois de alguns deles.

> **A tentação a nomear:** ajustar `GATE_PR_AUC_MIN` para 0,64 agora, "para ficar coerente".
> Isso seria mover o limite **depois de ver o resultado** — a mesma negociação *post hoc* que o
> pré-registro da Etapa 10.5 existe para impedir, e que o gate de fairness recusou fazer. O
> piso continua onde estava, medindo o conjunto para o qual foi calibrado.

O Brier, aliás, passa nos dois lugares (0,1352 no teste contra teto de 0,14) — o eixo de
calibração atravessa a troca de conjunto sem se mover.

### Por que só o campeão foi avaliado no teste

Avaliar os seis finalistas aqui e publicar a tabela seria **usar o teste para comparar**, que é
a mesma coisa que usá-lo para escolher com um passo a mais de negação. A comparação foi feita,
está registrada com número nas §5b–5d, e foi feita no conjunto certo. O teste responde a uma
pergunta só: *qual é o desempenho esperado do modelo que vai para produção?*

### O mecanismo — a disciplina virou código

`python -m src.reportar` **não recalcula nada** quando o registro existe: ele imprime o que foi
medido. Recalcular exige `--reexecutar`, que **confere** contra o registro em vez de
substituí-lo (verificado: reproduz em todos os eixos). E um teste da suíte amarra o número
publicado ao artefato promovido — se alguém promover outro modelo sem re-tocar o teste, a suíte
fica vermelha em vez de a documentação passar a descrever um objeto que não existe mais.

🔑 *O teste é gasto pelo uso, e "não vou olhar de novo" não é um mecanismo — é uma intenção.*

### A conta de negócio, fechada no conjunto de teste

| estratégia | custo do erro por ciclo | contra o modelo |
|---|---|---|
| não abordar ninguém | R$ 72.556 | +R$ 39.674 (**+120,7%**) |
| abordar a base inteira | R$ 64.170 | +R$ 31.288 (**+95,2%**) |
| **modelo, limiar 0,29** | **R$ 32.882** | — |

A segunda linha é a que sustenta a recusa de "otimizar recall puro": a solução degenerada
captura 100% dos churners **e custa quase o dobro**. Números reais do conjunto que ninguém
usou para decidir nada.

---

## 7. Limitações conhecidas

*(preenchida na Etapa 11. Regra usada para montar a tabela: **uma limitação só entra se alguém
puder verificá-la** — cada linha aponta para a medição, a seção ou o arquivo em que ela aparece.
O que não foi feito está escrito como não feito; um ❌ justificado vale mais que um ✅ inventado.)*

### 7a · Dados — o que o dataset não permite

| Limitação | Impacto | Mitigação possível |
|---|---|---|
| **O Telco é um retrato, sem eixo temporal** (§0) | não há split temporal: o modelo é validado sobre uma foto, e não sobre "treinar no passado, prever o futuro". Toda a política de retreino repousa num pressuposto de estabilidade que este dado não pode confirmar | base transacional com data de evento ⇒ split temporal e *backtesting* por safra. Sem isso, a validação superestima o desempenho sob deriva |
| **A janela cega do ground truth (~60 dias) é PRESSUPOSTO, não medição** (§5f/10d) | é o número que estrutura a política de retreino inteira — e ele foi assumido (dois ciclos de faturamento + carência), não observado | medir a distribuição real do atraso entre predição e desfecho na base transacional; a política já está escrita para receber esse número no lugar do assumido |
| **Nenhum dado comportamental de uso ou de contato com suporte** (§1) | é o bloco mais preditivo em churn de telecom, e ele simplesmente não existe na base. A Etapa 4 mediu o corolário: *feature engineering não cria informação, só reorganiza a existente* — as 4 features derivadas não deram ganho porque não havia o que derivar | integrar consumo, chamadas ao suporte, reclamações e tickets. É a mudança com maior retorno esperado do projeto — maior que qualquer troca de algoritmo medida nas Etapas 6-8 |
| **Geográficas fora das features** (`City`, `Zip Code`, `Lat/Long` — §1) | perda de sinal potencial (cobertura, concorrência local). Decisão dupla: 4,3 clientes por CEP torna a cardinalidade inviável, **e** o CEP é proxy de renda/raça | agregação por região com volume suficiente, auditada por fairness antes de entrar. Ficou como experimento controlado no backlog, não como omissão |
| **`Churn Score` e `CLTV` da IBM fora do treino** (§1, §5) | descarta o preditor mais forte da base — e é deliberado: o `Churn Score` tem **zero exceções em 1.409 linhas** (nenhum não-churner acima de 80, nenhum churner abaixo de 65), ou seja foi calculado com o desfecho conhecido. Usá-lo seria prever o modelo da IBM, não churn | nenhuma: a coluna é gabarito vazado. O `CLTV` segue em uso legítimo **fora do treino**, ordenando a fila por `P(churn) × valor` |
| **11 clientes com `Total Charges` vazio** (tenure 0) | são o cliente do primeiro mês — a população que a campanha mais quer pontuar, e a que a API chegou a **recusar com 422** até a 9e encontrar o defeito com dado real | já corrigido (§5e/9e), e o caso virou teste. Fica registrado porque a classe de erro reaparece a cada contrato novo: *contrato escrito à parte do objeto que ele descreve* |

### 7b · Modelo — o que o número não diz

| Limitação | Impacto | Mitigação possível |
|---|---|---|
| **PR-AUC 0,6646 é um ranking bom, não um oráculo** (piso 0,2654) | 2,5× o piso da métrica, e mesmo assim o `recall@10%` é **0,278** — de cada 100 churners, a campanha de 10% da base alcança 28. O teto estrutural desse ponto é 0,377, então o modelo está a 73,8% do máximo **possível**, não a 27,8% do desejável | mais features (7a) move o teto de desempenho; **nenhuma mudança de algoritmo move**, e isso está medido em três etapas independentes (6, 7 e 8) |
| **Seis candidatos dentro de 0,0057 de PR-AUC**, contra desvio entre folds de 0,019–0,024 (§6) | a escolha do campeão **não é sustentada pela métrica** — ela se desloca para critérios não-métricos (interpretabilidade, custo operacional, 1-SE). Quem ler a tabela esperando um vencedor claro não vai encontrar | é resultado, não defeito: está declarado assim na §6 e é o que a regra 1-SE existe para resolver. O que o corrigiria de verdade é mais sinal, não mais busca |
| **A rede neural não superou os modelos clássicos** (§5d) | o item de maior peso da avaliação (25%) produziu um resultado nulo. Ele é **demonstrado**, não alegado: controle positivo em `make_moons` (+9,93 pp onde há não-linearidade) contra −0,31 pp no Telco, mesmo desenho | nenhuma — a conclusão (*o sinal é essencialmente linear no logit*) tem quatro medições independentes. Com dados comportamentais (7a), vale retestar: a rede está versionada e roda com um comando |
| **O ganho do tuning foi exatamente zero** (§5c) | o default do sklearn já era o pico da grade de 18 configurações. Pré-registrado antes de medir, isso é previsão confirmada; sem o pré-registro seria desculpa | — |
| **Todas as métricas são OFFLINE** | não existe conversão observada da campanha de retenção: a cadeia KPI usa uma taxa de conversão **assumida** para converter FN/FP em reais. O R$ 31.750 por ciclo é uma estimativa condicionada a esse número, não um valor medido | A/B com grupo de controle em produção — que é exatamente o que a política de retreino (§5f/10d) exige para medir o modelo limpo. Fora do escopo de implementação, dentro do escopo da política |
| **Feedback loop não observável no escopo** | a campanha muda o rótulo de quem o modelo acertou, e o modelo é penalizado por ter acertado. Sem grupo de controle, a próxima safra de rótulos descreve um mundo que o próprio modelo alterou | fatia dos preditos que **não** recebe campanha. Está escrito na política e **não** implementado — não há produção real onde implementá-lo |

### 7c · Operação — o que o plano gratuito e o escopo impõem

| Limitação | Impacto | Mitigação possível |
|---|---|---|
| **API sem autenticação**, decisão registrada (§5e, item 102) | qualquer um com a URL pontua clientes. É API interna, sem frontend e sem usuários — e a limitação está **declarada**, não esquecida, com o custo de fechá-la já medido: **+0,329 ms** por requisição e ~25 linhas (API Key, `secrets.compare_digest`, chave por env var obrigatória) | implementar quando houver consumidor real. O ponto medido é que implementar depois custa o mesmo que implementar agora |
| **Sem rate limiting no processo** — de propósito (M04-A05/A08) | o teto de lote (`max_length=5.000`, 422 antes de tocar o modelo) contém o pior caso por requisição, mas não há cota por origem | mora no ingress/API Gateway/Redis, nunca no processo: medido, um limitador em memória perde o efeito ao escalar (60 concorrentes, limite 10 ⇒ passaram 31 com 4 workers), **nunca esquece um IP** (804,7 MB para 1 milhão de origens) e derruba a própria probe do orquestrador |
| **HTTPS termina no proxy da plataforma**; sem CORS, sem WAF, sem gerenciador de segredos | a aplicação fala HTTP na rede interna. CORS desabilitado é decisão (comunicação server-to-server), não omissão | infraestrutura de plataforma paga. ⚠️ Se um frontend entrar: **nunca** `allow_origins=["*"]` com `allow_credentials=True` — medido, o Starlette ecoa a origem do atacante |
| **Plano gratuito: 512 MB · 0,1 CPU · dorme em 15 min** | **cold start ≈ 30 s** — risco de **entrega**, não de operação (gravar a demonstração com a API dormindo trava no primeiro `curl`). E `--workers 1`, porque o nº de workers sai da RAM (171,2 MiB por worker) e não da vazão | plano pago ou VM. A imagem é portátil por construção: trocar de destino não reescreve a aplicação |
| **Latência de produção medida no cliente: p50 254 ms / p95 670 ms** (Brasil→Oregon) contra **1,16 ms** no cronômetro da própria API | 219× de diferença, e as duas leituras estão certas: o que sobra é rede e plataforma. Um SLA escrito sem o **ponto de medição** seria violado sempre ou nunca | região mais próxima e plano pago atacam a rede; nada disso é problema de modelo, e a **ação** do alerta manda confrontar as duas leituras justamente para não retreinar por causa de latitude |
| **Monitoramento offline, por script** — e o argumento é de arquitetura, não de escopo | Prometheus é modelo **pull** e o plano gratuito **escala a zero**: manter o scrape impede o serviço de dormir (que é o que o torna gratuito), deixá-lo dormir enche a série de buracos que disparam alerta de indisponibilidade toda noite — fadiga de alerta por construção. A própria M04-A08 escreve, sobre esse alerta noturno, *"no nosso caso isso é normal"* | stack de observabilidade (OTel/Prometheus/Grafana) sobre infraestrutura que não hiberna. É o padrão da indústria; o que não se sustenta é rodá-lo **sobre esta** arquitetura |
| 🚨 **O `simulate_drift.py` não roda contra produção** | no PaaS o stdout pertence à plataforma e não há como lê-lo de volta programaticamente. A detecção de drift é demonstrada **localmente**, sobre logs locais | coletor de logs (Datadog, Loki, S3) ou banco de inferências. É a terceira camada do *"quem lê este arquivo?"* cobrando na direção contrária: *o log fácil de emitir é o mesmo difícil de recuperar* |
| **O detector de drift é UNIVARIADO** (PSI/KS coluna a coluna) | cego ao caso em que nenhuma coluna mudou sozinha mas a **combinação** virou inédita. A própria simulação demonstra o limite: somando 12 meses a `tenure` sem mexer em `total_charges`, o segundo fica em PSI 0,034 (estável) enquanto o primeiro vai a 3,59 | `IsolationForest` treinado no treino e aplicado à janela, com a fração de anômalos ao lado do PSI (item 110 do backlog). Um detector de data drift é, por natureza, um modelo não supervisionado |
| **Rollback declarado e NÃO testado** (§5f/10e) | não há versão ruim em produção para reverter, e fabricar uma custaria um ciclo de deploy sem ninguém observando. Onde o teste era possível ele foi feito — gate, healthcheck e baseline foram **verificados reprovando** | ensaio de rollback numa janela combinada. A ausência está escrita porque um ✅ inventado aqui é pior que o ❌ |
| **Sem Model Registry persistente no CI** (job 3 omitido, §9.5) | a linhagem existe por outra via — `commit_hash()` + `sha256_dataset()` + `sha256` do artefato + versão do sklearn gravada dentro dele —, mas não há alias `Staging`/`Production` a trocar | tracking server externo (`MLFLOW_TRACKING_URI` por secret). O job foi omitido **com os dois bloqueios comentados dentro do YAML**: `mlruns/` morre com o runner, e escrever o job antes disso produziria um CI que declara sucesso sem ter feito nada |
| **O pacote ainda não é instalável** (`pip install .`) e o pacote de topo se chama `src` | `PYTHONPATH` aparece em três lugares (Dockerfile, alvo do Makefile, scripts fora da raiz), e um nome genérico colide entre projetos | item 17/9g — é trabalho do Módulo 05. Vale registrar que o campeão **não** depende disso: nenhum `FunctionTransformer` entrou nele, e há teste que carrega o artefato **sem o repo no `sys.path`** |
| **Artefato 8,3 KB × imagem 510 MB = 1 : 61.000** | promover um modelo novo republica 61 mil vezes mais bytes do que a mudança real | artefato em volume ou registry, servido por download no boot. Não construído agora **de propósito**: no plano gratuito não há volume, e a alternativa seria retreinar no container — o que exigiria o dado bruto (LGPD) dentro de uma imagem publicável |

### 7d · Governança — o que ficou aceito, e por quem

| Limitação | Impacto | Mitigação possível |
|---|---|---|
| 🚨 **Disparidade de recall de 58,89 pp em `Dependents`**, aceita e declarada (§5g) | de cada 100 clientes com dependentes que iam cancelar, o modelo marca **22** (contra 81 no outro grupo). A causa é aritmética — prevalência 7,01% × 32,47% sob limiar global —, o que **não isenta**: o efeito sobre a pessoa é o mesmo | as três saídas estão medidas: limiar por grupo (6,33 pp, +R$ 2.918/ciclo) resolve mas é *disparate treatment* e **exige dono jurídico que o projeto não tem**; remover a coluna atenua para 13,20 pp e **reprova o gate** (PR-AUC 0,6427). O que falta não é técnica, é autoridade para decidir |
| **O CI não barra a disparidade** — de propósito | um `assert disparidade <= 0.10` reprovaria o modelo em produção a cada push, e a saída praticada seria afrouxar o número até o verde. O CI **caracteriza** os quatro valores (±1e-4, bilateral) e impede que alguém remova as demográficas das features | quando houver decisão de negócio sobre a mitigação, o piso volta como piso. Enquanto não houver, quem carrega o compromisso é o Model Card, e a limitação está escrita **nos dois lugares** |
| **A auditoria de fairness cobre 4 atributos**, todos do próprio dataset | raça, renda e escolaridade não existem na base — e são exatamente os eixos em que proxies geográficos costumam morder | dados demográficos consentidos, ou auditoria por proxy declarada. Registrar a ausência é o que impede ler "quatro atributos auditados" como "auditado" |
| **Sem DPIA, sem base legal validada juridicamente** | a base legal assumida é legítimo interesse (§10.5d); ninguém do jurídico validou | avaliação de impacto formal antes de uso real. O RACI do Model Card já diz **quem faria** cada papel num contexto real — o projeto é solo e isso está dito |
| **A predição é apoio à decisão, com humano no meio** — e isso é requisito, não limitação técnica | se alguém automatizar a ação sobre a fila, o desenho inteiro (LGPD Art. 20, revisão humana, usos proibidos do Model Card) deixa de valer sem que nada no código mude | os *usos NÃO pretendidos* estão escritos no Model Card, que é o lugar onde essa fronteira é verificável |

---

## 8. Reprodutibilidade

**A pergunta que esta seção responde:** *"alguém clona o repositório numa máquina limpa e chega
ao mesmo número?"* — e a resposta tem de ser verificável, não declarativa.

### O comando

```bash
git clone <repo> && cd tc-mle-fase1
make setup      # cria o venv e instala as versões TRAVADAS (requirements.txt)
make ci         # lint + 133 testes + gate: treina o campeão do bruto e mede na validação
```

`make ci` é o que o GitHub Actions roda, **na mesma ordem e com os mesmos binários** — os alvos
invocam `.venv/bin/ruff` e `.venv/bin/pytest` por caminho explícito, nunca o nome solto. O motivo
é um erro real: `make ci` rodado fora do venv chamou o `ruff` **global** (0.6.4 contra 0.16.2 do
lock) e reportou 6 erros que o CI não vê. *Ferramenta de validação cujo resultado depende do
ambiente do chamador não valida.*

O gate **treina de ponta a ponta a partir do dado bruto** em vez de ler uma métrica salva: uma
métrica lida de arquivo provaria apenas que o arquivo existe.

### O que está fixado

| Item | Valor | Onde |
|---|---|---|
| **Seed única do projeto** | `SEED = 42` | `src/config.py`; usada nas duas chamadas de `train_test_split`, no `random_state` de todo estimador, no `StratifiedKFold`/`RepeatedStratifiedKFold` e em `np.random.seed()` no topo de cada `main()` |
| **Seed do PyTorch** | `torch.manual_seed(seed)` por execução, **≥5 seeds** na Etapa 8 | `src/mlp.py` — a seed é **fator do experimento**, não detalhe: comparar uma média (LogReg, determinística) com um sorteio seria comparar réguas diferentes |
| **Partição** | 60/20/20 estratificado, derivado do índice do DataFrame | `src/data.py::dividir()` — duas chamadas, com a fração da segunda recalculada sobre o que sobrou |
| **Dataset** | `Telco_customer_churn.xlsx`, sha256 `1bcbc0ccc9b352175216979102628e579cfbde2c3ff57b005de168a433122640` | versionado no Git (1,3 MB, imutável); `data.sha256_dataset()` recalcula e o valor vai como **tag do run do MLflow** e **dentro do artefato promovido** |
| **Versões de biblioteca** | `requirements.txt` com **pins exatos** (`scikit-learn==1.9.0`, `numpy==2.4.6`, `pandas==2.3.3`) | gerado a partir de `requirements.in`; `requirements-serve.txt` é **derivado do mesmo lock**, não resolvido de novo |
| **Python** | 3.12.5 no desenvolvimento · `python:3.12-slim@sha256:2c941e…` na imagem | tag é mutável, digest não. E a versão é **parte do contrato do artefato**: `src/artefato.py` compara a versão do scikit-learn gravada na promoção com a do ambiente e **mata o processo na carga** se divergirem |
| **Plataforma da imagem** | `--platform linux/amd64` explícito no `make docker-build` | build no Apple Silicon produziria arm64 e o `exec format error` apareceria **no deploy**, não no build local que passou |
| **Linhagem de cada run** | `commit_hash()` + `sha256_dataset()` como tags do MLflow | responde *"qual commit e qual snapshot de dados geraram este modelo?"*. O terceiro artefato de Sculley — o **ambiente** — está no artefato (`versoes_treino`) e no digest da base |

### O artefato servido, identificado

```
artefato        : models/campeao.joblib   (8.306 → 10.398 bytes após a 10a-2)
sha256          : b8109cce2efabbd8f15747019deed6b395e48913c97cdb4b5b0ef527f8e2e97d
versao_modelo   : 1.0.0        modelo: logreg        features: 13
limiar_operacao : 0.29
promovido_em    : 2026-08-18T22:35:23+00:00     commit: dc77f2e
dataset_sha256  : 1bcbc0ccc9b35217…
ambiente        : python 3.12.5 · scikit-learn 1.9.0 · numpy 2.4.6 · pandas 2.3.3
```

O mesmo `sha256` é declarado pelo `GET /health`, ecoado em cada linha do log de inferência e
verificável em produção com um `curl` — é o que torna a pergunta *"o que está servindo é o que
foi avaliado?"* respondível em vez de presumida.

### Como se sabe que reproduz — as três verificações

1. **Determinismo medido, não suposto:** três execuções do pipeline completo produzem o **mesmo
   hash** das probabilidades. É o que autoriza o teste de caracterização a fixar `|PR-AUC −
   0,6646| ≤ 1e-4` — a tolerância existe para troca de biblioteca, não para ruído de execução.
2. **Round-trip do artefato, bit a bit** (`np.array_equal`, não `approx`): treinar → salvar →
   limpar da memória → recarregar → as predições batem exatamente. Serialização não é aproximação
   numérica: diferença na décima casa significaria objeto **reconstruído** em vez de restaurado.
3. **Reprodução em outro sistema operacional, verificada:** o mesmo artefato em macOS e em Linux
   dá **PR-AUC idêntico nos 10 dígitos** (0,6646020519) e **0 decisões trocadas** no limiar, com
   40% das linhas diferindo no último bit (4 ulps, BLAS). 🔑 As duas metades são o resultado:
   reproduz o que decide, **e** proíbe que qualquer coisa rio abaixo compare predições por
   igualdade exata.

### O que a reprodutibilidade deste repositório NÃO cobre

- **Não há DVC.** O dataset bruto é pequeno (1,3 MB) e **imutável**, então versioná-lo no Git
  resolve o mesmo problema com menos peças; a linhagem que o DVC daria já vem do `sha256` gravado
  no artefato e no run. É decisão justificada (M03-A02: *"Git para código e arquivos pequenos;
  DVC/LFS para dados brutos volumosos"*), não omissão.
- **`mlruns/` é local e não é versionado.** Os números das Etapas 3–8 são reproduzíveis pelos
  comandos (`make comparacao`, `make tuning`, `make mlp`), mas o histórico de runs de quem
  executou não viaja no clone. É a mesma pendência do registro no CI (§7c).
- **O Dockerfile não é reprodutível; a imagem é.** A receita só repete o resultado porque **as
  três** fontes de variação estão fixadas (base por digest, lockfile com pins, plataforma
  explícita). Retirar qualquer uma delas produziria imagens diferentes em meses diferentes com o
  mesmo `git checkout`.

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
| 2026-08-11 | 9.5 | **Gate mede na VALIDAÇÃO**, não no conjunto de teste | divergência deliberada do enunciado da Aula 08: decidir promoção olhando o teste a cada push o converte em validação e ele deixa de estimar generalização. O gap treino-teste continua bonito — o sintoma não denuncia |
| 2026-08-11 | 9.5 | Piso do gate **absoluto (0,66)**, não relativo a % do baseline | *"≥80% do baseline"* aceitaria 0,53 com o baseline em 0,6646 — pior que modelos já rejeitados. A folga de 0,005 cobre variação numérica entre plataformas, não degradação aceita |
| 2026-08-11 | 9.5 | Gate **treina** em vez de ler métrica salva | precisa provar que o pipeline inteiro roda de ponta a ponta com estas versões, a partir do bruto. Métrica lida de arquivo provaria só que o arquivo existe |
| 2026-08-11 | 9.5 | **Job de registro no Model Registry deliberadamente NÃO escrito** | seria a pior espécie de CI — o que declara sucesso sem fazer nada. `mlruns/` morre com o runner e o artefato serializado é a Etapa 9. Os dois bloqueios estão comentados no próprio workflow |
| 2026-08-11 | 9.5 | **Continuous Delivery, não Deployment** | a predição dispara ação comercial com custo real (desconto), logo o último passo para produção é humano |
| 2026-08-11 | 9.5 | `ConvergenceWarning` **vira erro** no pytest (`filterwarnings`) | é falha silenciosa: o sklearn devolve os coeficientes onde o LBFGS parou e não reclama de novo. Fecha o item 28 do revisita de forma permanente, não por inspeção manual |
| 2026-08-11 | 9.5 | `NPY002` ignorado no ruff **com justificativa**, não por conveniência | a regra pede trocar `np.random.seed()` por `Generator`; aqui a chamada é deliberadamente global, porque é esse estado que o sklearn consulta onde não expõe `random_state`. Regra genérica errada para o caso |
| 2026-08-11 | 9.5 | `pyproject.toml` **não declara dependências** | a fonte de verdade é `requirements.in` → `requirements.txt` travado. Duplicar a lista criaria duas verdades que divergem em silêncio |
| 2026-08-11 | 9.5 | 🎯 **Reprodutibilidade cross-platform PROVADA, não declarada** | o gate rodou no runner Ubuntu (Python 3.12.13, BLAS diferente) e devolveu **exatamente** os mesmos números do macOS local: PR-AUC **0,6646**, limiar 0,29, custo R$ 31.750, mesmo sha256 do dataset. É a resposta empírica a *"alguém clona numa máquina limpa e obtém o seu número?"* — e a folga de 0,005 do gate, prevista para cobrir variação de plataforma, mostrou-se desnecessária (variação real: zero). Mantida por prudência |
| 2026-08-11 | 7 | Expectativas do tuning **pré-registradas** no log antes da primeira execução | previsão confirmada é evidência de método; previsão escrita depois do resultado é narrativa. As quatro se confirmaram, inclusive a de ganho pequeno |
| 2026-08-11 | 7 | Estratégia de busca escolhida pelo **tamanho da grade**: grid na LogReg (18), random no HGB (40) | Bergstra & Bengio (2012), bibliografia oficial: num grid, um eixo inerte gasta k× o orçamento medindo a mesma coisa. Grade pequena com todos os eixos relevantes não tem esse problema |
| 2026-08-11 | 7 | `refit` = **regra 1-SE** (Hastie et al., 2009), não `argmax` | com 12 de 18 candidatos dentro do envelope, o pico é escolha entre empates. A 1-SE recupera a complexidade verdadeira em 98% das amostras contra 66% do argmin (300 seeds, hands-on da M02-A07) |
| 2026-08-11 | 7 | Envelope 1-SE calculado em **três leituras** da dispersão, não uma | `dp/√K` superestima a precisão porque dobras de CV repetida não são independentes (Nadeau–Bengio, ausente do material). Concordância entre as três é o que torna a conclusão robusta |
| 2026-08-11 | 7 | 🔑 **Tuning da LogReg: ganho ZERO — o default já era o pico da grade** (0,6904) | contraparte do achado da Etapa 6: a regularização move o número onde ela estava ausente (ensembles até a pureza), não num modelo que a biblioteca já entrega regularizado |
| 2026-08-11 | 7 | Eixo `penalty` substituído por **`l1_ratio`** com `solver='saga'` nos três pontos | o sklearn 1.8 depreciou `penalty` (remoção em 1.10); `saga` mantém a função objetivo constante ao longo do eixo, enquanto `liblinear` penalizaria também o intercepto |
| 2026-08-11 | 7 | Grade do HGB **estendida** para onde o aviso de borda apontou (`max_leaf_nodes` até 2) | aviso que não gera ação é decoração. Resultado: a 1-SE desceu para o toco de decisão e continua na borda — terceira medição de que o sinal do Telco é simples |
| 2026-08-11 | 7 | 🚨 **Viés de seleção medido com grupo de controle**, não só declarado | o campeão (0 candidatos) cai 0,0258 da CV para a validação; a LogReg (18) cai o mesmo; o HGB (40) cai 0,0384. O excedente cresce com a grade, como a aula prevê — e inverteu o ranking dos finalistas |
| 2026-08-11 | 7 | **Nunca reportar o pico da grade como generalização** — número da documentação sai da validação | o pico do HGB (0,7007) anunciaria +0,0088 sobre o campeão; o número honesto (0,6589) é −0,0057. Diferença de 0,042, sem nenhum sintoma que a denuncie |
| 2026-08-11 | 7 | Teste **pareado** (t e Wilcoxon) sobre as 15 dobras, como evidência corroborante | HGB vence em 13/15 dobras com p<0,001 e mesmo assim perde na validação: significância e relevância são perguntas diferentes, e o pareamento tem poder para detectar +0,011, que a operação não distingue de zero |
| 2026-08-11 | 7 | **Campeão mantido** (LogReg default); config 1-SE não adotada apesar de a regra apontá-la | empate na validação (−0,0026) contra custo concreto: a margem do gate cairia para 0,0020, menos que a folga de 0,005 reservada à variação entre plataformas |
| 2026-08-11 | 7 | **Gate do CI mantido em 0,66** | ele acompanha o campeão, e o campeão não mudou. Subir o piso sem modelo melhor é obstáculo arbitrário; a regra continua sendo que ele sobe quando o campeão subir |
| 2026-08-11 | 7 | `Total Charges` confirmada redundante por um **terceiro método independente** (L1 zerou) | correlação 0,9996 (EDA) + 7º na permutação (Etapa 6) + coeficiente zero (Etapa 7). Mantida no contrato v1 porque remover não muda métrica: CV 0,6883 × 0,6904 |
| 2026-08-11 | 7 | Etapa 8 comparará o MLP contra o **HGB tunado**, não contra a configuração de referência | comparar a rede com um boosting mal ajustado repetiria o erro que o gate de justiça da Etapa 6 existe para impedir |
| 2026-08-11 | 8 | Expectativas da Etapa 8 **pré-registradas** (6 previsões + critério de refutação de cada uma) | placar final 5 confirmadas, 1 refutada. A refutada foi a mais útil: produziu o experimento do controle de orçamento, que não existiria se a previsão tivesse acertado |
| 2026-08-11 | 8 | MLP embrulhado como **estimador do scikit-learn** e rodado no MESMO `Pipeline`, CV 5×3 e seed das Etapas 6 e 7 | é a condição para o número conversar com a tabela existente; e o teste pareado nas mesmas 15 dobras sai de graça porque a CV tem seed fixa |
| 2026-08-11 | 8 | Early stopping por **PR-AUC em validação interna extraída do treino**, nunca na validação da Etapa 2 | `MLPClassifier(early_stopping=True)` pontua por `accuracy_score` e não expõe `scoring`; e parar o treino olhando a validação faria do nº de épocas mais um parâmetro escolhido nela |
| 2026-08-11 | 8 | Pesos restaurados são os da **melhor época**, não os da última | parar por paciência e ficar com o estado final entrega um modelo pior do que aquele que o próprio critério de parada elegeu — sem sintoma visível |
| 2026-08-11 | 8 | 🎯 **A regra 1-SE, dentro da família das redes, elegeu profundidade ZERO** | quarta medição independente de que o sinal é linear no logit, e a mais forte: a família mais flexível de todas escolheu não usar a capacidade que tinha |
| 2026-08-11 | 8 | Avaliar **dois** candidatos: a escolha da 1-SE e a melhor rede COM camada oculta | quando o resultado metodológico é "profundidade zero", entregável e achado divergem — reportar só um dos dois esconderia metade |
| 2026-08-11 | 8 | 🚨 **Controle de orçamento** (`LogRegMesmoOrcamento`) criado para decompor a diferença do controle | dois fatores mudavam juntos (otimizador e os 15% que o early stopping consome); com dois números a atribuição seria escolha, não medida. Resultado: −0,0014 dados + −0,0053 otimizador |
| 2026-08-11 | 8 | **Variância de seed medida nos dois regimes** (com e sem camada oculta), não em um só | "variância de inicialização" agrega três fontes; sem camada oculta o problema é convexo e não há mínimo local. O convexo variou MAIS (0,0035 × 0,0008), o oposto do que o nome sugere |
| 2026-08-11 | 8 | **Resposta ao item 36: o protocolo da Etapa 6 NÃO precisa da dimensão seed** | desvio entre seeds é 0,04× o desvio entre folds. Mas é medição, não intuição: o XOR da M02-A05 falhava em 14% das inicializações — a conclusão de lá não transferia |
| 2026-08-11 | 8 | **Campeão mantido; o MLP é entregue como artefato da fase, não como modelo de produção** | −0,0031 de PR-AUC e −0,0046 no pareado, sem consistência (5/15 dobras, p=0,064). Um "não" documentado vale mais que um resultado forçado |
| 2026-08-11 | 8 | 🚨 **PR-AUC e custo em reais apontaram para lados opostos** — decidido pela hierarquia da Etapa 0 | a rede custa R$ 479 a menos por ciclo (maior que a dispersão entre suas seeds, R$ 203) e perde na métrica de seleção declarada. Trocar de régua depois de ver o resultado é o pecado que a 1-SE existe para evitar; o número foi ao backlog, não ao lixo |
| 2026-08-11 | 8 | **Gate do CI ganhou o segundo eixo: `Brier ≤ 0,14`** (item 41), verificado reprovando | gate de um eixo aprova modelo que ordena igual e calibra pior — e a fila é ordenada por `P(churn) × CLTV`, então a probabilidade é multiplicada por reais. Entrou antes de ser necessário, que é quando é barato |
| 2026-08-11 | 8 | **Item 16 fechado: as 4 features da Etapa 4 seguem sem efeito no MLP** (+0,0006) | terceira família de modelos a rejeitá-las. "A rede aprende interação sozinha" corta nos dois sentidos: se valesse algo, ela a acharia sem a feature pronta |
| 2026-08-17 | 9c | **Artefato promovido existe**: `models/campeao.joblib`, LogReg v1.0.0, sha256 `5f2b62dc…` | até hoje todas as medições rodavam contra um pipeline treinado na hora. Comparar modelos assim é legítimo; **servir** não é — o serviço tem de servir *o* modelo avaliado, não um irmão treinado com o mesmo código |
| 2026-08-17 | 9c | `gate.treinar_campeao()` extraída: **promoção e gate treinam pelo mesmo caminho** | duas definições do mesmo modelo divergem em silêncio, e o gate **retreina** em vez de conferir o artefato ⇒ o CI aprovaria um modelo enquanto a API serve outro, os dois verdes |
| 2026-08-17 | 9c | **Um único arquivo** com pipeline + metadados dentro | modelo e metadados separados, sem verificação de que combinam, produzem rótulo errado com HTTP 200 — o modo de falha mais caro numa API de ML, porque não falha |
| 2026-08-17 | 9c | Serializado como **`dict` puro**, nunca uma dataclass de `src.artefato` | serializar classe nossa obrigaria o ambiente de inferência a ter `src` importável — recriaria o acoplamento do item 17 por comodidade de tipagem. Há teste que carrega o artefato **sem o repo no `sys.path`** |
| 2026-08-17 | 9c | 🚨 **Versão do sklearn gravada à mão e verificada na CARGA** (item 96b) | `__setstate__` faz `state.pop("_sklearn_version")` e o `InconsistentVersionWarning` é `UserWarning`: o modelo carrega, prediz **0,6601 em vez de 0,6646** e o carimbo some. Se não for escrito ao promover, não existe depois |
| 2026-08-17 | 9c | **Falhar na carga** (`ArtefatoIncompativel`), não avisar no log | subir degradado é escolher a falha silenciosa — 200 OK com a probabilidade de outro modelo. É o servidor nº 8 da Knight Capital em versão doméstica |
| 2026-08-17 | 9c | **Limiar de operação 0,29 viaja dentro do artefato**, nunca literal na API | o corte é propriedade do modelo servido e muda com ele; a segunda cópia não daria erro, só cortaria a fila no lugar errado. Fecha o 9h antes de a API existir |
| 2026-08-17 | 9c | Contrato de colunas declarado a partir de **`pipe.feature_names_in_`**, não de `config.FEATURES` | o config acompanha o código, o artefato acompanha o modelo servido. É a fonte de verdade do schema Pydantic da 9d (item 94) |
| 2026-08-17 | 9c | **Nada é promovido sem passar no gate**, verificado reprovando (piso forçado a 0,99) | nada gravado e o artefato anterior **idêntico byte a byte**. Promoção que grava mesmo reprovada não é promoção, é sobrescrita |
| 2026-08-17 | 9c | **Round-trip bit a bit** conferido no ato da promoção *e* em teste (`np.array_equal`, não `approx`) | serialização não é aproximação numérica: diferença na décima casa significaria objeto reconstruído em vez de restaurado. A suíte tinha 56 testes e **nenhum tocava o objeto serializado** |
| 2026-08-17 | 9c | **Item 96(a) aplicado**: `InconsistentVersionWarning` como erro no `filterwarnings` | o `requirements.in` tinha a razão do pin escrita em comentário desde a Etapa 3 e nenhum mecanismo que a aplicasse. Uma linha, mesmo argumento do `ConvergenceWarning` |
| 2026-08-17 | 9 | `httpx2` e `joblib` declarados no `requirements.in` | `httpx2` era **bloqueio**, não planejamento: sem ele o `TestClient` levanta `RuntimeError` e nenhum teste de rota roda (Starlette 1.6 pede `httpx2`, não `httpx`). `joblib` passou a ser import **direto** de `src/artefato.py` |
| 2026-08-17 | 9c | ⚠️ **Pendência declarada para a 9f:** o artefato **não** é versionado no Git (`models/*` ignorado) | a imagem precisa dele e o clone não o traz. Ou o build promove (e o dado bruto LGPD entraria na imagem, o que a M03-A04 proíbe), ou o artefato é injetado como camada/volume. Decisão da 9f, registrada como pendência e não como esquecimento |
| 2026-08-17 | 9d | **API em três camadas** (`schema` / `servico` / `app`), separadas por uma razão para mudar | o critério não é estético: os testes de limiar, contrato de erro e formato de resposta rodam contra o serviço e contra dublês, sem `TestClient` e sem tocar o disco |
| 2026-08-17 | 9d | 🚨 **Contrato de entrada DERIVADO do artefato** (13 nomes + ordem, 25 valores em `Literal`, 3 faixas) | fecha os quatro payloads que davam **200 com predição corrompida**: `Contract` inexistente (153 clientes trocam de lado), `Tenure = -999` (P=1,0000 nas 1.409 linhas, 878 cruzam o limiar), `Monthly = -999`, `Total = 1e9`. Custo: 0,002 ms por requisição |
| 2026-08-17 | 9d | ⚠️ **Faixas numéricas são decisão de negócio** (0–120 · 0–200 · 0–12.000), não o máximo do treino | colar o `le` em 72 meses rejeitaria cliente legítimo de 73. O alvo é a sentinela de nulo, não a cauda. Teste exige que toda numérica do artefato tenha faixa declarada |
| 2026-08-17 | 9d | 🔑 **Derivar o contrato do artefato e carregar por `lifespan` são incompatíveis** — a factory `criar_app(artefato)` resolve | o schema precisa do artefato **antes de as rotas existirem**; o `lifespan` roda depois. A factory carrega ainda mais cedo e serve ao mesmo propósito medido (o deploy paga os ~715 ms, não o primeiro cliente — 383×) |
| 2026-08-17 | 9d | **`def`, nunca `async def`**, em todas as rotas | vazão idêntica (o GIL serializa), mas o atraso do event loop vai de 32,58 ms para 206,28 ms = o lote inteiro. Quem fica preso junto é o `/health` do orquestrador ⇒ container reiniciado sob carga |
| 2026-08-17 | 9d | **`/predict-batch` é o endpoint principal**, com `max_length=5.000` | 825× (2,853 ms × 2.353 ms para 1.409 linhas): o custo do sklearn é fixo por chamada. Sem teto, o lote é vetor de negação de serviço, porque o trabalho é síncrono |
| 2026-08-17 | 9d | **`response_model` em todas as rotas** e **nenhum eco da entrada** | o campo não declarado não sai mesmo que o `return` o inclua (36 × 374 bytes). Correlação por `request_id` gerado no servidor, no corpo e no header |
| 2026-08-17 | 9d | 🚨 **Handler de `RequestValidationError` removendo `input`/`ctx`** — `extra="forbid"` sozinho não basta | `forbid` **concentra** o vazamento: o erro devolve `loc` E `input` (`CustomerID: 3668-QPYBK`). O mesmo handler cobre o 500, cujo `str(e)` do sklearn cita o dado do cliente. Verificado nos 8 payloads inválidos: nenhum devolve valor do payload |
| 2026-08-17 | 9d | **`/health` declara identidade** (versão, sha256, nº de features, limiar) e o processo **morre no boot** sem artefato válido | um 200 que não diz o que está carregado é o oitavo servidor da Knight Capital. Verificado reprovando: artefato apagado ⇒ uvicorn não sobe, com mensagem acionável — em vez de responder `healthy` e falhar no `/predict` |
| 2026-08-17 | 9d | **`pd.DataFrame` com `by_alias=True` + `predict_proba` + limiar do artefato** | seleção por nome, não por posição (teste embaralha as chaves do JSON); e `.predict()` custaria R$ 7.546/ciclo e 83 churners. A resposta leva probabilidade, decisão e o limiar aplicado |
| 2026-08-17 | 9d | 🚨 **Achado: lote e unitário não são bit-idênticos** — 495 de 1.409 linhas (35%) diferem em até 2,2e-16 | é o BLAS mudando o caminho de vetorização com o nº de linhas. **Zero** decisões mudam no limiar 0,29, mas nada rio abaixo pode comparar predições por igualdade exata (cache por hash, deduplicação, reconciliação do log da Etapa 10) |
| 2026-08-17 | 9d | **Escopo declarado:** sem autenticação, `/docs` aberta, sem padrões GoF, sem log JSONL | limitação declarada vale mais que omissão; `/docs` publica a descrição do modelo, não dado pessoal; padrão que não paga é patternitis; o log é Etapa 10, e o `request_id` já é o gancho |
| 2026-08-17 | 9d | 🚨 **Nenhum `app = criar_app()` de módulo: o uvicorn recebe a FACTORY** (`--factory`) | erro cometido e pego pelo CI: o objeto de módulo tornava a carga do artefato **efeito colateral do import**, e a suíte inteira falhava na coleta no runner limpo (`models/` vazio por decisão) enquanto `make ci` local passava. 🔑 *O defeito só existe onde o arquivo não existe* — nenhuma execução local podia encontrá-lo |
| 2026-08-17 | 9d | `config.ARTEFATO` configurável por **`TC_ARTEFATO`**, com teste de import em subprocesso | traz o defeito acima para dentro do alcance do desenvolvimento (simula a máquina limpa) e é o que o container vai querer de qualquer forma, quando o artefato vier de um volume |
| 2026-08-18 | 9f | **`.dockerignore` escrito ANTES do Dockerfile, e como allowlist** | 🚨 o Docker **não lê** o `.gitignore`. Medido construindo de propósito com `COPY . .` e sem exclusão: contexto **1,5 GB** → imagem **2,03 GB**, com o `.xlsx` de 7.043 clientes reais, o `mlruns/` e o `.venv` de binários **arm64 de macOS** dentro. Camada Docker é imutável — `RUN rm` posterior não apaga da camada anterior ⇒ é exposição de dado pessoal por imagem publicada, não peso. E allowlist porque *denylist protege contra o que já existe; allowlist protege contra o que ainda vai existir* |
| 2026-08-18 | 9f | Base por **digest** + lockfile com pins + `--platform linux/amd64` explícito | 🔑 *o Dockerfile é a receita; a imagem é o artefato* — só a imagem é reprodutível por construção. `python:3.9-slim` (o da aula) **não constrói** este projeto: `scikit-learn 1.9.0` exige `>=3.11`. Essa falha alta só existe **porque** há pins; sem eles o pip acharia versões antigas, a imagem subiria e o artefato quebraria só na carga. E build no Apple Silicon produz arm64 ⇒ `exec format error` no deploy, não no build local que passou |
| 2026-08-18 | 9f | `requirements-serve.txt` **derivado do lock existente**, não resolvido de novo | imagem de serviço **510 MB** contra venv de **1.400 MB**; `torch` sozinho tem **1.057 MB** — o dobro da imagem inteira — e o campeão não o importa em lugar nenhum (*código ≠ dependência*: o perdedor fica versionado e fora do runtime). Derivar do lock faz a divergência de versão do sklearn — que `src/artefato.py` mata no boot — ser **impossível** em vez de detectável |
| 2026-08-18 | 9f | `--workers 1` e `HEALTHCHECK` exigindo `status == "pronto"` | o nº de workers sai da **RAM**, não da vazão: o container mede 171,2 MiB ⇒ 4 workers ≈ 685 MB contra o teto de 512 MB. E o healthcheck foi **verificado reprovando em três cenários** — serviço real exit 0, porta morta exit 1, e **um servidor que responde 200 com `degradado` exit 1**: só o terceiro distingue prontidão de "a porta abriu" |
| 2026-08-18 | 9f | Artefato **versionado como exceção NOMEADA** (`!models/campeao.joblib`, 8.306 B) | a plataforma de deploy builda a partir do **clone do Git**, sem máquina nossa no caminho e sem volume no plano gratuito: ou o artefato está versionado, ou a imagem sobe sem modelo. É o arquivo, nunca `!models/*` — nomear impede o próximo `.joblib` experimental de entrar de carona. 🔑 *Exceção declarada com motivo é decisão; exceção silenciosa é a regra apodrecendo* |
| 2026-08-18 | 9e | 🚨 **O teste de integração manda as 1.409 linhas REAIS da validação**, não payload sintético — e foi isso que achou o defeito | a linha 487 tomou **422 da própria API**: um dos 11 clientes com `Total Charges` vazio (todos com `tenure = 0`, sem ciclo de faturamento), que o pipeline sempre soube pontuar (0,2449). *O contrato ficou mais **estreito** que o pipeline que ele protege* — erro simétrico do `-999`, mesma raiz: contrato escrito à parte do objeto que descreve. **Consequência de negócio: a API recusava exatamente a população que a campanha mais quer pontuar** |
| 2026-08-18 | 9e | A correção do vazio tem **duas peças, e nenhuma serve sozinha**; e `Tenure Months: null` continua 422 de propósito | `null` aceito só onde o grupo `zero` do `ColumnTransformer` declara tratamento (de novo o artefato decidindo), **mais** `None → np.nan` no serviço: sem a segunda peça o 422 vira **500**, porque `pd.DataFrame` com `None` vira `dtype=object` e o LogReg levanta `ValueError`. A assimetria é deliberada: em `Total Charges` o vazio *significa* algo; em `Tenure` seria a sentinela de nulo entrando pela porta da frente |
| 2026-08-18 | 9e | `/health` separado em **`versoes_treino` e `versoes_runtime`** | dentro da imagem o campo dizia Python **3.12.5** (carimbo do treino) enquanto o processo rodava **3.12.14**. Nada errado no artefato — o **rótulo** respondia a outra pergunta, no endpoint cuja pergunta é *"o que este serviço tem?"*. 🔑 *Campo ambíguo é pior que campo ausente: parece resposta* |
| 2026-08-18 | 9e | 📏 **macOS × Linux: 40% das linhas diferem em até 4 ulps, 0 decisões trocadas, PR-AUC idêntico nos 10 dígitos** | as duas metades importam: não muda nenhuma decisão de negócio nem move a métrica reportada, **e** proíbe que qualquer coisa rio abaixo compare predições por igualdade exata (cache por hash, deduplicação, reconciliação do log da Etapa 10). Mesma lição do lote × unitário, agora entre sistemas operacionais |
| 2026-08-18 | 9f-quater | **Destino = PaaS (Render)**; FaaS descartado **por medição**, IaaS por custo de administração | o fechamento transitivo de serviço tem **253,2 MB** contra o limite de **250 MB** do Lambda por zip — estoura por 1,3% **antes** do adaptador e do artefato. *O que não cabe na função serverless não é o modelo (8,3 KB), é o que ele precisa para existir.* E a portabilidade da imagem é a resposta a *"e se quisesse trocar de nuvem?"*: trocar de destino não reescreve a aplicação |
| 2026-08-18 | 9f-quater | 🚨 **Declarar `PORT: 8000`** — revertendo a omissão que fora deliberada | a porta foi deixada em branco *de propósito* (*"é parâmetro que a plataforma reivindica"*) e o resultado foi **48% das requisições com `x-render-routing: no-server`**, com a aplicação respondendo 200 a tudo que lhe chegava e zero restarts: `EXPOSE 8000` contra processo em `${PORT:-8000}` = 10000, e a plataforma *detectando* qual era (a doc dela diz *"usually able to detect"*). Declarado ⇒ **120/120**. 🔑 *Não declarar não é delegar: delegar é declarar o valor que ela usa* |
| 2026-08-18 | 9f-quater | **`autoDeployTrigger: checksPass`** no `render.yaml` | o default do PaaS é redeploy a cada push — *entrega contínua **sem** integração contínua*, publicando o que o `make ci` ainda não aprovou. Uma linha põe o gate do CI no caminho, sem webhook e sem secret; verificado funcionando (o commit da correção de porta só implantou depois do CI verde). ⚠️ Consequência a conhecer: commit **sem check nenhum não é implantado**, em silêncio |
| 2026-08-18 | 9f-quater | Verificação final: **o mesmo script da 9e rodado contra a URL pública** | PR-AUC **0,6646020519** local × **0,6646020519** na nuvem, **0 decisões trocadas** no limiar. Custou zero porque o script já recebia o alvo por variável de ambiente — *escrever o teste parametrizável desde o começo é o que permite reusá-lo no ambiente que importa*. Latência Brasil→Oregon com 0,1 CPU: unitário p50 254 ms / p95 670 ms; `/health` sobreviveu a 8 lotes de 1.409 em voo |
| 2026-08-18 | 10a | Log JSON (uma linha por requisição, stdout) com os 6 campos canônicos **+ `artefato_sha256`** | ter `model_version` na resposta não basta: a resposta é efêmera e o log é o rastro de auditoria. Sem o hash, quando o PSI cruzar o limiar daqui a três semanas não se consegue descartar *"trocaram o artefato no meio da janela"* — e as duas explicações são indistinguíveis |
| 2026-08-18 | 10a | 🚨 **Máscara LGPD em dois níveis:** `scores` sempre, `features` só com `TC_LOG_FEATURES=1` | as duas famílias de drift têm custo de privacidade **oposto**: prediction drift é número sem atributo ao lado (não identifica ninguém ⇒ vigilância de graça rodando na nuvem); data drift custa `Gender`/`Senior Citizen`/`Partner`/`Dependents` no stream de um **terceiro**. 🔑 O `.gitignore` protege o repo e o `.dockerignore` protege a imagem — **nada protege o stdout do container**, e as duas primeiras são regras de exclusão que a terceira não pode ter, porque logar é a finalidade. *Onde não se pode excluir, decide-se o que se escreve* — antes da primeira linha, porque log emitido não volta |
| 2026-08-18 | 10a | `--no-access-log` no `CMD` | o uvicorn tem log de acesso próprio: medido, 3 requisições davam **3 linhas JSON + 3 de texto puro**, e o consumidor do `.jsonl` é código. `logger.propagate = False` não resolve — o emissor é outro processo lógico |
| 2026-08-18 | 10a | 🚨 Dois defeitos que **só aparecem para teste que olha o conteúdo da linha** | (a) o handler de `Exception` do FastAPI roda **por fora** do middleware ⇒ ler `resposta.status_code` perderia **toda falha interna** no log; (b) `StreamHandler` **congela o stream na construção** ⇒ sob pytest a linha ia para um descritor que nem `capsys` nem `capfd` liam, e um teste *"respondeu 200, logo logou"* ficaria **verde sem nunca ter visto uma linha**. Verificado reprovando em 3 sabotagens |
| 2026-08-18 | 10a-2 | **Estatísticas de referência DENTRO do artefato**, não num JSON ao lado | mesmo argumento do limiar: a distribuição de referência é propriedade **daquele modelo**, não do repositório. Um arquivo ao lado recria o par *"dois arquivos que podem não combinar"*, e nesta variante a falha **não falha** — baseline de um modelo contra predições de outro mede drift fantasma. 🔑 O argumento decisivo é operacional: guardando **proporções por bin** (não dados), o baseline roda **no container sem o dataset**, que é justamente o que não pode estar lá |
| 2026-08-18 | 10a-2 | Congelar **três** coisas: estatísticas, **bordas dos bins** e a distribuição das **saídas** | PSI com bins recalculados na janela compara duas escalas e a régua `<0,10/0,25` deixa de valer; bins **abertos nas pontas** porque produção tem valor fora do intervalo do treino — e binning fechado faria a janela mais deslocada dar o PSI mais tranquilo. 🚨 **Furo próprio, achado no meio da execução:** a 1ª versão congelou só `P(X)`, e como as features só vão ao log com a flag e os `scores` vão **sempre**, havia baseline para a vigilância que quase nunca roda e **nenhum** para a que sempre roda |
| 2026-08-18 | 10a-2 | O baseline de scores sai da **VALIDAÇÃO**, não do treino | in-sample o modelo é otimista, e baseline otimista faz produção parecer deslocada **para sempre** — drift fantasma permanente, que treina o time a ignorar o alerta. Medido aqui: PSI treino×validação = 0,0022, ou seja o erro sairia barato *neste* modelo regularizado; não sairia num HGB profundo |
| 2026-08-18 | 10a-2 | 🚨 A verificação do risco previsto **inverteu o risco** ⇒ 4ª checagem na carga | esperava-se que a chave nova recusasse artefato antigo; `carregar(estrito=True)` não valida o conjunto de chaves (tudo é `.get`), então o risco real era o **oposto**: artefato **sem** baseline carregando em silêncio e a falha aparecendo semanas depois como **ausência de alerta** — indistinguível de ausência de problema |
| 2026-08-18 | 10c-bis | **Drift fabricado em dois cenários**, com o detector verificado disparando | drift real leva meses e um TC de seis semanas nunca veria um: em vez de esperar, fabrica-se — e o que se testa não é o modelo, é o **sistema de vigilância**. Dois porque as duas famílias se movem independentes: reajuste ×1,15 ⇒ data drift **1,49 (agir)** com prediction drift **0,03 (estável)**; tenure +12 ⇒ as duas acusam **e a fila encolhe 37%** (tenure alto = menos churn) 🚨 *o alerta chega vestido de boa notícia* |
| 2026-08-18 | 10c-bis | 🎯 **Achado que inverte a premissa do enunciado:** categoria inédita **não passa silenciosa** nesta API | o schema é derivado do artefato (`Literal[tuple(ohe.categories_)]`) ⇒ **422 antes de tocar o modelo**, em vez do tudo-zero silencioso que o `OneHotEncoder` produziria. Nesta arquitetura, categoria nova é evento de **serviço**, não de drift: o detector certo é **alerta de taxa de 4xx**, e ele dispara na **primeira** requisição em vez de ao fim de uma janela |
| 2026-08-18 | 10c-bis | ⚠️ As três limitações da simulação escritas junto — e a terceira vale mais que o script | (1) é **covariate shift**, detectável *por construção*, porque foi construído — não simula concept drift; (2) KS com amostra grande acusa qualquer coisa ⇒ decidir por **PSI**, magnitude e não p-valor; (3) 🔑 o deslocamento é univariado **e o detector também**: somar 12 meses a `tenure` sem mexer em `total_charges` cria clientes com dois anos de casa e a fatura de um — combinação que **não existe no mundo** — e `total_charges` fica em PSI 0,034 (estável) enquanto `tenure` vai a 3,59. *A simulação também mostra o limite do detector, não só o poder dele* |
| 2026-08-19 | 10b | **Banda morta em todo alerta: dois limiares (gatilho e desarme) + persistência de janelas**, e não um limiar único | um limiar oscila em torno de si mesmo e cada oscilação consome um ciclo de investigação. 🔑 A régua do PSI (`<0,10 / 0,10–0,25 / >0,25`) **já era** um termostato e ninguém a lê assim: as pontas viram gatilho e desarme, o meio vira zona morta. Mesma histerese da regra 1-SE, aplicada ao tempo |
| 2026-08-19 | 10b | Limiares de drift **calibrados contra o ruído medido**, não copiados da regra de bolso | validação × treino (só ruído de partição) dá PSI **0,0128** ⇒ o gatilho de 0,25 está a **19,5×** desse piso e o de investigação a 7,8×. É *"observar a variabilidade normal antes de fixar o threshold"* com número em vez de intenção |
| 2026-08-19 | 10b | **Toda taxa é escrita com denominador visível, e a janela é de VOLUME (`n ≥ 400`)**, não de tempo | `rate(5xx[5m]) > 0.05` (forma do material) é 5xx **por segundo**, não 5%: dispara com 3 erros/min e piora conforme o produto cresce. E sem carga real, janela por dia produziria decis de dois pontos — 400 é a janela em que os controles foram medidos |
| 2026-08-19 | 10b | O limiar de latência declara o **ponto de medição** | mesma requisição: **254 ms** no cliente (Brasil→Oregon) × **1,16 ms** no cronômetro da própria API — 219×. As duas leituras estão certas; um SLA sem o ponto declarado é métrica sem piso. Cold start (~30 s) fica fora por construção |
| 2026-08-19 | 10b | Tráfego, saturação e taxa de aceitação da campanha ficam **explicitamente vazios** | não há carga real nem conversão observável (métricas offline). Limiar inventado sobre tráfego inexistente é decoração — um ❌ justificado vale mais que um ✅ que ninguém pode verificar |
| 2026-08-19 | 10d | **Agendado (trimestral) como piso + drift como reação**; degradação de performance só como confirmação | a janela cega do ground truth (~60 dias, **pressuposto declarado** — o dataset não tem eixo temporal) torna o gatilho por performance um retrovisor. E drift dispara **investigação**, nunca retreino automático |
| 2026-08-19 | 10d | **O log de inferência não é fonte de rótulo**; o rótulo vem da base transacional, reconciliado por `request_id` | retreinar com a predição é self-training degenerativo: o erro da versão anterior vira alvo da próxima e as métricas melhoram enquanto a realidade se afasta. É a razão de o `request_id` existir desde a 9d, antes de haver log |
| 2026-08-19 | 10d | **Grupo de controle** exigido pela política (fatia dos preditos que não recebe campanha) | feedback loop: a retenção altera o rótulo de quem o modelo acertou, e o modelo é penalizado por ter acertado. É a única forma de medir o modelo limpo **e** o valor real da campanha. Fora do escopo de implementação, dentro do escopo da política |
| 2026-08-19 | 10d | 🔑 **Achado: o CI já era a trava do retreino, sem ter sido projetado para isso** | o teste de caracterização (`\|PR-AUC − 0,6646\| ≤ 1e-4`) reprova **qualquer** retreino, inclusive melhor ⇒ promover exige editar `PR_AUC_REF`/`BRIER_REF` no mesmo commit, o que é um diff revisável. O *Continuous Delivery* escolhido por escrito na 9.5 já estava implementado em código. Gate = "é bom o bastante?"; caracterização = "é o mesmo?" |
| 2026-08-19 | 10e | **Rollback em duas camadas, nesta ordem:** painel do Render (imagem, minutos) → `git revert` do artefato + CI + push (definitivo) | 🚨 como o artefato é versionado junto do código e o Render implanta por `checksPass`, **o rollback só no painel é desfeito pelo próximo commit de qualquer natureza** — um ajuste de README reconstrói a imagem da `main` e traz o artefato ruim de volta, com o CI verde. *Onde o artefato é versionado junto do código, o rollback também tem de ser* |
| 2026-08-19 | 10e | Passo obrigatório de **conferência de identidade** (`/health` → `artefato_sha256`) depois de cada camada do rollback | reverter e não conferir é o cenário da Knight Capital com outra roupa. Custa um `curl` porque o `/health` foi construído na 9d para responder isso |
| 2026-08-19 | 10e | Blue-green e canary **descartados com motivo**; endpoint versionado (`/v1`) reconhecido como a estratégia que já estava feita | blue-green dobra o custo (plano gratuito e único) e não mede nada antes da comutação; canary exige roteador de tráfego. E canary sem teste de significância é 1% de tráfego produzindo uma decisão de 100% — o pecado do pico da grade, em produção |
| 2026-08-19 | 10e | **Rollback declarado e NÃO testado**, dito com todas as letras | não há versão ruim em produção para reverter, e fabricar uma custaria um ciclo de deploy sem ninguém observando. Onde o teste é possível ele é obrigatório (gate, healthcheck, baseline — todos verificados reprovando); onde não é, a ausência é escrita |
| 2026-08-19 | 10.5 | **Pré-registro commitado ANTES da primeira medição** (commit `9f8c596`) | o `git log` vira a prova da ordem, e ela deixa de ser uma afirmação minha no meio do texto. Custa um commit, e é a diferença entre *"o limite foi escrito antes"* e *"o limite foi escrito antes, veja o hash"* |
| 2026-08-19 | 10.5 | Limite pré-registrado: **disparidade máxima de recall = 10 pp** em cada atributo sensível | limite escrito depois do resultado vira negociação *post hoc* (*"12 pontos tá ruim? dá pra viver"*). 🎯 **O placar ruim é o argumento: 3 das 4 previsões erraram** — se as quatro tivessem batido, a auditoria teria confirmado o que eu já achava e não teria produzido informação nenhuma |
| 2026-08-19 | 10.5 | 🚨 **Achado: 58,89 pp de disparidade de recall em `Dependents`** (0,2174 × 0,8063) | *de cada 100 clientes com dependentes que iam cancelar, o modelo marca 22.* `Senior Citizen` (12,72 pp) e `Partner` (10,03 pp) também estouram; só `Gender` (8,16 pp) passa. A causa é **aritmética**, não defeito: prevalência 7,01% × 32,47% com limiar global — o teorema da impossibilidade em forma concreta. Não isenta: o efeito sobre a pessoa é o mesmo |
| 2026-08-19 | 10.5 | **Auditar no limiar de OPERAÇÃO (0,29), nunca no 0,5** — e isso virou teste | medido: `Dependents` dá **58,89 pp** em 0,29 e **76,81 pp** em 0,5. São números sobre modelos diferentes, e só um deles existe. É o `.predict()` da Etapa 9i um andar acima: sai plausível, não falha, e descreve uma fila que ninguém corta |
| 2026-08-19 | 10.5 | **IC binomial por grupo**, ao lado de cada recall | o pior achado é o de **menor amostra** (23 churners; recall 0,2174, IC95 [0,075; 0,437]). O IC impede as duas leituras erradas simétricas — precisão falsa e descarte fácil (*"são só 23 casos"*) — e o que fecha é o **limite superior continuar abaixo** de 0,8063: a incerteza não salva o resultado |
| 2026-08-19 | 10.5 | ⚖️ **Decisão: manter o modelo e declarar**, com as três saídas medidas **antes** de decidir | manter (58,89 pp · PR-AUC 0,6646 · R$ 31.750) × **limiar por grupo** (6,33 pp · +R$ 2.918/ciclo · +109 na fila) × remover a coluna (13,20 pp · **PR-AUC 0,6427, abaixo do piso** ⇒ nem promovível). O limiar por grupo é tratamento explicitamente diferente por atributo protegido (*disparate treatment*): defensável como ação afirmativa, mas **exige dono jurídico** que este projeto não tem. Registrar o número, o preço da correção e quem deveria decidir é mais honesto que aplicar correção que ninguém autorizou |
| 2026-08-19 | 10.5 | 🚨 **Correção ao catálogo, medida:** *fairness through unawareness* **atenua** — não é inócua | remover `Dependents` levou a disparidade de **58,89 → 13,20 pp**: o viés sobrevive nos proxies mas **enfraquece**, e o slogan corrente (*"remover a coluna não remove o viés"*) estava pela metade. O que mata a opção é o **preço** (reprova o gate de PR-AUC), não a inutilidade. ⚠️ Ressalva que vai junto: **um modelo pior tende a parecer mais justo** — no limite, um aleatório tem disparidade zero. *Métrica de fairness nunca se lê sozinha* |
| 2026-08-19 | 10.5 | 🚨 **O gate de fairness NÃO barra os 10 pp — de propósito**; o CI caracteriza em vez de reprovar | um `assert disparidade <= 0.10` reprovaria **o modelo em produção a cada push**, e a saída óbvia seria afrouxar o número até o verde: exatamente a negociação que o pré-registro existe para impedir. *Gate que nasce violado não protege nada; ensina o time a editar o limite.* Com o piso conscientemente não atingido, entra o **contrato**: os 4 números caracterizados (±1e-4, bilateral) e um teste que falha se alguém remover as demográficas das features. A limitação — *nada aqui impede o modelo de continuar a 58,89 pp* — está escrita no Model Card **e** no teste |
| 2026-08-19 | 10.5 | ⚠️ Erro de procedimento registrado: **`git checkout` não reverte arquivo untracked** | ao verificar os testes reprovando, a 1ª sabotagem continuou aplicada durante a 3ª e produziu 4 falhas onde devia haver 1 — e por um minuto pareceu que o teste estava mal escrito. 🔑 *O mecanismo de desfazer só funciona sobre o que o sistema já conhece.* Mesma família do `make ci` fora do venv: ferramenta de segurança cujo resultado depende de um estado que ninguém verificou |
| 2026-08-19 | 11 | **Conjunto de teste tocado UMA vez**, com o **artefato promovido** (`b8109cce…`), não com um modelo retreinado | é o objeto que a API serve: comparar modelos treinando na hora é legítimo, *servir* não é, e **reportar segue o que serve**. PR-AUC **0,6496** (piso 0,2654), Brier 0,1352, recall@10% **0,286** (75,9% do teto), custo R$ 32.882/ciclo contra R$ 72.556 de não fazer nada |
| 2026-08-19 | 11 | 🎯 **Achado: o IC95 de bootstrap tem 0,1056 de largura — 18,5× a distância entre os seis finalistas (0,0057)** | *o empate técnico das Etapas 6-8 nunca foi indecisão de método; é a resolução da amostra.* Com 1.409 linhas e prevalência 26,5%, **nenhum** conjunto de teste deste tamanho poderia desempatar LogReg, HGB e MLP. Reportar 0,6496 sem o intervalo sugeriria uma precisão de quatro casas que a amostra não sustenta — mesmo instrumento do IC binomial da §5g, agora na métrica principal |
| 2026-08-19 | 11 | O gap validação→teste (**+0,0150**) é atribuído a **duas** causas e a nenhuma delas isoladamente | as partições têm o mesmo n e a mesma prevalência ⇒ não é composição. Somam-se **sorteio** (0,0150 cabe folgado no IC) e **viés de seleção** (o campeão foi eleito por ser o melhor na validação entre seis candidatos dentro do ruído). Separá-las exigiria um segundo conjunto de teste, que não existe — e *atribuir a uma causa a diferença que tem duas* é armadilha catalogada aqui desde a Etapa 8 |
| 2026-08-19 | 11 | 🚨 **A métrica agregada caiu e a operacional subiu** — e as duas leituras estão certas | PR-AUC 0,6646 → 0,6496 enquanto `recall@10%` 0,278 → **0,286** e ROC-AUC 0,8472 → 0,8495. É a hierarquia da Etapa 0 valendo: PR-AUC integra sobre limiares em que a campanha **nunca vai operar**; `recall@k` mede o ponto onde a decisão acontece. ⚠️ Nenhuma das variações escapa do IC — o que a tabela mostra não é "melhorou", é que **as perguntas são diferentes** |
| 2026-08-19 | 11 | 🚨 **O piso do gate (0,66) NÃO foi movido para acomodar o teste (0,6496)** | ajustá-lo "para ficar coerente" seria mover o limite **depois de ver o resultado** — a negociação *post hoc* que o pré-registro da 10.5 existe para impedir. O gate mede a **validação** por decisão registrada desde a Etapa 2, e é para ela que o piso foi calibrado. O Brier passa nos dois conjuntos (0,1352 ≤ 0,14) |
| 2026-08-19 | 11 | **Só o campeão foi avaliado no teste** — os outros cinco finalistas, não | publicar a tabela dos seis no teste seria usá-lo para **comparar**, que é usá-lo para escolher com um passo de negação a mais. A comparação está feita, com número, nas §5b–5d, e no conjunto certo. O teste responde a uma pergunta só: *qual é o desempenho esperado do que vai para produção?* |
| 2026-08-19 | 11 | A disciplina do toque único virou **mecanismo**, não intenção | `python -m src.reportar` não recalcula quando o registro existe (imprime o que foi medido); recalcular exige `--reexecutar`, que **confere** em vez de substituir (verificado: reproduz em todos os eixos); e um teste da suíte amarra o número publicado ao **sha256 do artefato promovido** ⇒ promover outro modelo sem re-tocar o teste deixa a suíte **vermelha**, em vez de deixar a documentação descrevendo um objeto que não existe mais |
| 2026-08-19 | 11 | **Curva de ganho cumulativo com TRÊS linhas** (modelo · acaso · teto estrutural) como o gráfico principal da entrega | o teto `k/prevalência` é o que impede o gráfico de mentir: sem ele, um ranking a 75,9% do **máximo possível** parece fraco. E a curva não precisa de tradução — *"contatando 10% da base, a campanha alcança 28,6% de quem ia cancelar"* é frase que o negócio consome direto. Bônus do mesmo lote: histograma das probabilidades por classe verdadeira, que é o Brier em forma de figura |
| 2026-08-25 | 8-bis | **`MLPClassifier` do scikit-learn rodado sob o protocolo idêntico** — o requisito do enunciado cumprido com número, não com justificativa | 20 configurações × 15 dobras. Melhor rede `(8,)` α=0: CV **0,6874** contra **0,6904** da LogReg. 🎯 A **1-SE elegeu `hidden=()` outra vez** — segunda implementação independente a desistir da profundidade pelo mesmo critério. 🔑 E o **gap treino−CV é monotônico na capacidade** (0,0033 → 0,0241 → 0,0499 → 0,0885): cada neurônio a mais foi inteiramente para dentro do treino |
| 2026-08-25 | 8-bis | 🚨 **O custo do `early_stopping` que pontua por acurácia, medido** — a defesa do PyTorch deixou de ser leitura de código-fonte | mesma configuração, mesmas seeds, muda uma palavra-chave: PR-AUC **0,6846 → 0,6542** na CV (−0,0304, mais de um desvio entre folds), **0,6655 → 0,6378** na validação, Brier +0,0068, **+R$ 1.838/ciclo**. E o número que decidiu a parada está gravado no objeto: `best_validation_score_ = 0,8175` — **acurácia**, num problema cujo piso de acurácia é 73,46%. ⚠️ A terceira coluna é a que o fonte não dava: a **dispersão entre seeds triplica** (±0,0054 → ±0,0191), porque a época passa a ser escolhida por métrica de degrau sobre split aleatório |
| 2026-08-25 | 8-bis | 🔑 **Réplica não planejada do achado nº 2 da Etapa 8**, com outra biblioteca | o controle de profundidade zero em Adam ficou **−0,0059** abaixo da LogReg em LBFGS — contra os **−0,0053** que a decomposição do terceiro modelo da Etapa 8 havia atribuído ao otimizador. *Duas implementações de Adam, equipes diferentes, mesmo preço em relação ao LBFGS*, com 0,0006 de distância. A decomposição era uma conta feita com um modelo construído para o fim; esta é uma medição independente |
| 2026-08-25 | 8-bis | 🚨 **O candidato novo ficou marginalmente ACIMA do campeão na validação — e o campeão foi mantido** | `MLPClassifier (8,)`: PR-AUC **0,6655** × 0,6646, Brier −0,0009, custo −R$ 957, R@10% +0,005 — quatro métricas para o mesmo lado. Motivos de manter, todos numéricos: (a) na **CV**, que é onde a hierarquia da Etapa 0 manda selecionar, o campeão ganha (0,6904 × 0,6874); (b) o desvio entre as 5 seeds é **±0,0054, seis vezes** a diferença, e a seed 2024 sozinha deu 0,6586 — *qual dos dois ganha depende de qual inicialização for promovida*; (c) o IC95 do teste tem 0,1056 de largura, **117×** a diferença; (d) promover custaria re-tocar o teste, a auditoria de fairness, o baseline de drift e a caracterização, para servir um modelo **não determinístico** e sem os 13 coeficientes que o Art. 20 torna úteis. 🎯 *O teste de caracterização (±1e-4) barrou a troca na primeira vez em que houve um candidato melhor — e ele não foi projetado para isso* |

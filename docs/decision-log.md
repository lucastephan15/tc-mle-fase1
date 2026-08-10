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

| Decisão | Escolha | Alternativa considerada | Por quê |
|---|---|---|---|
| Estratégia de split | | | |
| Imputação numérica | | | |
| Imputação categórica | | | |
| Encoding | | | |
| Escalonamento | | | |
| Tratamento de desbalanceamento | | | |
| Tratamento de outliers | | | |

---

## 3. Features criadas

| Feature | Hipótese de negócio | Disponível no momento da predição? | Ganho medido | Mantida? |
|---|---|---|---|---|
| | | ⬜ sim / ⬜ não | | |

> A coluna "Disponível no momento da predição?" é obrigatória. Um "não" = descarte imediato,
> por mais que a métrica melhore.

---

## 4. Seleção de features

- **Método usado:**
- **Por que esse método:**
- **Nº de features antes → depois:**
- **Features descartadas que surpreenderam:**
- **Features com importância suspeitamente alta (investigadas?):**

---

## 5. Experimentos — tabela mestra

> Toda linha deve corresponder a um run rastreável no MLflow.

| # | Run MLflow | Modelo | Features | Hiperparâmetros | Métrica primária (teste) | F1 | Precision | Recall | Gap treino-teste | Observação |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | | **Baseline** (chute na majoritária) | — | — | | — | — | — | — | piso absoluto |
| 1 | | LogReg (MVP) | | | | | | | | baseline a bater |
| | | | | | | | | | | |

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

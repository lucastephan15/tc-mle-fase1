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

**Fonte / versão do dataset:**

**Dimensões (linhas × colunas):**

### Colunas descartadas por suspeita de leakage
| Coluna | Por que suspeita | Decisão | Justificativa |
|---|---|---|---|
| | | | |

### Cobertura das 4 categorias de variáveis
> Forçar cobertura para não pender só para o demográfico. Em churn, as **comportamentais**
> são as mais preditivas.

| Categoria | Variáveis disponíveis no dataset | Faltam / seria bom ter |
|---|---|---|
| Demográficas | | |
| **Comportamentais** ⭐ | | |
| Históricas | | |
| Contextuais | | |

### Achados relevantes do EDA
| Achado | Implicação para a modelagem |
|---|---|
| | |

### Zeros e vazios investigados (armadilha do rótulo censurado)
| Coluna | O zero/vazio é medição ou ausência de medição? | Tratamento |
|---|---|---|
| | | |

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

# Model Card — Predição de Churn (Telecom) v1.0.0

> **Este documento é um contrato de desempenho, não um folheto.** Os números
> abaixo são verificados por testes automatizados (`tests/test_gate.py`,
> `tests/test_fairness.py`); mudá-los exige um commit que alguém revisa. A seção
> de fairness traz um resultado **ruim** logo na primeira tela — está aí de
> propósito: um Model Card que só publica o que favorece o modelo não governa
> nada.

## 1. Identificação

| | |
|---|---|
| **Modelo** | Regressão Logística (`scikit-learn`), `C=1,0`, penalidade L2 |
| **Versão** | 1.0.0 |
| **Pipeline** | imputação → `OneHotEncoder` (sem `drop`) → `StandardScaler` → LogReg, tudo em um `Pipeline` serializado |
| **Artefato** | `models/campeao.joblib` (10.398 B), identidade verificada na carga e declarada em `GET /health` |
| **Dataset** | IBM Telco Customer Churn (variante estendida, 33 colunas) · sha256 `1bcbc0cc…` |
| **Commit de promoção** | `dc77f2e` · seed `42` |
| **Ambiente de treino** | Python 3.12.5 · scikit-learn 1.9.0 · numpy 2.4.6 · pandas 2.3.3 |
| **Autor / responsável** | Luca Stephan — Tech Challenge Fase 1, FIAP+Alura PosTech |
| **Data** | 19/08/2026 |

## 2. Uso pretendido

**Priorizar clientes para campanha de retenção.** A saída é uma **probabilidade**
(`P(churn)`) mais uma decisão calculada no limiar de operação **0,29**, derivado
da economia do erro (R$ 194 por churner perdido × R$ 62 por atenção desperdiçada).
O consumo previsto é uma **fila ordenada** sobre a carteira, em lote, com revisão
humana antes de qualquer contato.

⚠️ **É apoio à decisão, não decisão automatizada.** Isso não é retórica: a LGPD
(Art. 20) dá ao titular direito a revisão de decisão tomada unicamente por meios
automatizados, e a arquitetura foi desenhada para que exista um humano no
caminho — a API devolve score, decisão **e o limiar aplicado**, para que quem
opera possa discordar do corte.

## 3. Usos NÃO pretendidos

- ❌ **Negar, suspender ou degradar serviço** a um cliente.
- ❌ **Definir preço, reajuste ou condição contratual** individual.
- ❌ **Decidir sozinho**, sem revisão humana, qualquer ação que atinja o cliente.
- ❌ **Aplicar a pessoa jurídica** ou a segmentos ausentes do treino (o dataset é
  de clientes residenciais da Califórnia).
- ❌ **Usar como evidência sobre um indivíduo** ("este cliente é infiel"). O
  modelo ordena risco numa carteira; ele não faz afirmação sobre uma pessoa.
- ❌ **Reaproveitar o score fora do ciclo de retenção** (crédito, cobrança,
  qualquer decisão de maior risco). O modelo foi validado para uma coisa só.

## 4. Dados

- **Origem:** IBM Telco Customer Churn, 7.043 clientes, retrato **sem eixo
  temporal**. Partição 60/20/20 estratificada (`random_state=42`); o conjunto de
  **teste segue intocado** até a documentação final.
- **Alvo:** `Churn Value` — prevalência **26,54%**.
- **13 features** usadas: `Total Charges`, `Tenure Months`, `Monthly Charges`,
  `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Multiple Lines`,
  `Internet Service`, `Online Security`, `Tech Support`, `Contract`,
  `Paperless Billing`.
- **Excluídas, com o motivo:**

| coluna | por quê |
|---|---|
| `Churn Reason`, `Churn Label` | **leakage**: só existem depois do cancelamento |
| `Churn Score` (IBM) | **gabarito vazado** — nenhum churner abaixo de 65 e nenhum não-churner acima de 80 em 1.409 linhas |
| `CustomerID` | identificador e dado pessoal (LGPD) |
| `City`, `Zip Code`, `Latitude`, `Longitude` | cardinalidade inviável (4,3 clientes/CEP) **e** proxy de renda e raça |
| 6 colunas de serviço | removidas por ablação na Etapa 5, sem custo de PR-AUC |

🔑 **Os quatro atributos sensíveis (`Gender`, `Senior Citizen`, `Partner`,
`Dependents`) foram MANTIDOS nas features de propósito.** Removê-los não
removeria o viés — o modelo os reconstrói pelos proxies — removeria a capacidade
de **medi-lo**. A seção 7 traz a medição que sustenta isso.

## 5. Métricas globais (validação, n=1.409)

| métrica | valor | **piso** (modelo trivial) |
|---|---|---|
| **PR-AUC** (primária) | **0,6646** | 0,2654 |
| ROC-AUC | 0,8472 | 0,5 |
| Brier (calibração) | 0,1339 | — |
| Recall @ limiar 0,29 | 0,7701 | — |
| Precisão @ limiar 0,29 | 0,5424 | — |
| recall@10% | 0,2781 (**73,8% do teto estrutural** de 0,377) | 0,10 |
| Custo do erro por ciclo | **R$ 31.750** | R$ 72.556 (não fazer nada) · R$ 64.170 (abordar todos) |

⚠️ **Nenhuma métrica aparece sem o piso.** Um modelo que responde "ninguém sai"
tem 73,46% de acurácia nesta base — é por isso que a acurácia não é reportada
como métrica de decisão em lugar nenhum deste projeto.

**Gate automatizado (CI):** promoção exige **PR-AUC ≥ 0,66 E Brier ≤ 0,14** —
dois eixos, porque um modelo pode melhorar a ordenação e piorar a calibração ao
mesmo tempo, e aí o limiar herdado passa a cortar a fila no lugar errado.

### 5b. Conjunto de teste — a leitura única (Etapa 11)

O teste ficou **intocado** de 10/08 a 19/08: toda seleção — features, algoritmo,
hiperparâmetros, arquitetura da rede, limiar e gate — saiu da validação. Esta é a
única leitura, feita sobre **o artefato promovido** (`b8109cce…`), não sobre um
modelo retreinado.

| métrica | teste (n=1.409) | validação | piso |
|---|---|---|---|
| **PR-AUC** | **0,6496** · IC95 [0,5960; 0,7016] | 0,6646 | 0,2654 |
| ROC-AUC | 0,8495 | 0,8472 | 0,5 |
| Brier | 0,1352 | 0,1339 | — |
| recall@10% | **0,286** (75,9% do teto de 0,377) | 0,278 | 0,10 |
| Custo do erro por ciclo | **R$ 32.882** | R$ 31.750 | R$ 72.556 · R$ 64.170 |

🎯 **O IC95 tem 0,1056 de largura — 18,5× a distância entre os seis finalistas
(0,0057).** O empate técnico entre LogReg, HGB e MLP não era indecisão de método:
**nenhum** conjunto de teste deste tamanho poderia desempatá-los.

🚨 **A métrica agregada caiu e a operacional subiu** (PR-AUC 0,6646 → 0,6496 com
recall@10% 0,278 → 0,286). É a hierarquia declarada na Etapa 0 valendo: PR-AUC
integra sobre limiares em que a campanha nunca vai operar; `recall@k` mede o
ponto onde a decisão acontece. Nenhuma das variações escapa do IC.

⚠️ **O piso do gate (0,66) NÃO foi movido** para acomodar o 0,6496. O gate mede a
**validação**, por decisão registrada desde a Etapa 2; ajustá-lo agora seria mover
o limite depois de ver o resultado. Registro em `docs/resultado-teste-final.json`,
e um teste da suíte amarra o número publicado ao sha256 do artefato promovido.

## 6. Explicabilidade (LGPD Art. 20)

O modelo é linear, então a explicação não é reconstruída *a posteriori*: ela **é**
o modelo. Odds ratios (`exp(β)`, features padronizadas):

| feature | odds ratio | leitura |
|---|---|---|
| `Contract = Month-to-month` | **1,877** | contrato mensal quase dobra as chances de churn |
| `Dependents = No` | 1,813 | não ter dependentes eleva o risco |
| `Total Charges` | 1,706 | |
| `Monthly Charges` | 1,604 | fatura mensal alta eleva o risco |
| `Contract = Two year` | 0,405 | contrato longo protege |
| `Dependents = Yes` | **0,349** | **ter dependentes reduz muito o risco** |
| `Tenure Months` | **0,256** | tempo de casa é o maior fator de proteção |

⚠️ Duas ressalvas: *odds ratio* **não é risco relativo** (divergem com
prevalência de 26,5%), e os coeficientes estão **encolhidos pela L2**, logo os
valores são conservadores.

🔑 **Note que a explicabilidade já anuncia o achado da seção 7:** `Dependents_Yes`
= 0,349 é o segundo menor coeficiente do modelo. O modelo aprendeu — corretamente
— que esse grupo cancela pouco, e é exatamente isso que produz a disparidade
abaixo.

## 7. 🚨 Fairness — métricas desagregadas, e o resultado ruim

**Auditadas na validação, no limiar de OPERAÇÃO (0,29)**, com `fairlearn`.
Auditar em 0,5 mediria um modelo que este projeto não usa.

**A métrica que carrega o dano é o recall**, não a acurácia: o cliente que ia
cancelar e não foi marcado **não entra na fila**, não recebe oferta e vai embora
— e isso nunca vira linha de planilha, porque ninguém registra a campanha que não
foi feita.

| atributo | grupo | n | prevalência | **recall** | selection rate | disparidade |
|---|---|---|---|---|---|---|
| **Gender** | Female | 715 | 0,2531 | 0,8122 | 0,3818 | **8,16 pp** ✅ |
| | Male | 694 | 0,2781 | 0,7306 | 0,3718 | |
| **Senior Citizen** | No | 1.180 | 0,2356 | 0,7374 | 0,3212 | **12,72 pp** ❌ |
| | Yes | 229 | 0,4192 | 0,8646 | 0,6638 | |
| **Partner** | No | 716 | 0,3101 | 0,8108 | 0,4735 | **10,03 pp** ❌ |
| | Yes | 693 | 0,2193 | 0,7105 | 0,2771 | |
| **Dependents** | No | 1.081 | 0,3247 | 0,8063 | 0,4801 | 🚨 **58,89 pp** ❌ |
| | **Yes** | **328** | **0,0701** | **0,2174** | **0,0366** | |

### O que isso significa em português

**De cada 100 clientes com dependentes que iam cancelar, o modelo marca 22.** Os
outros 78 não entram na fila de retenção e não recebem campanha. Em `Senior
Citizen` a disparidade também estoura o limite, mas na direção oposta: o grupo
idoso é **melhor** atendido (recall 0,86 × 0,74) — é disparidade, é reportada, e
não é o mesmo dano.

### Limite de disparidade — a política, escrita ANTES de medir

> **Disparidade de recall ≤ 10 pontos percentuais**, por atributo sensível, na
> validação, no limiar de operação.

Pré-registrada em `docs/decision-log.md` §5g e **commitada antes** de a auditoria
existir (`git log`, commit `9f8c596`). **Não é cumprida por 3 dos 4 atributos.**

### O diagnóstico, antes da conclusão

1. **A causa é aritmética, não um defeito de código.** A prevalência de churn no
   grupo com dependentes é **7,01%** contra 32,47% no outro. Um limiar global
   aplicado a um grupo de baixo risco marca pouca gente por construção. É o
   **teorema da impossibilidade** (Kleinberg; Chouldechova) acontecendo: com
   prevalências diferentes, calibração e paridade de erro não podem valer juntas.
2. **O pior achado é o de menor amostra.** O grupo tem **23 churners** na
   validação; IC95 do recall = [0,075; 0,437]. ⚠️ Mas o **limite superior do
   intervalo continua muito abaixo** de 0,8063 — a incerteza não salva o
   resultado, só impede que ele seja lido com precisão falsa.

### As três saídas foram medidas — e a decisão está registrada

| saída | disparidade | PR-AUC | custo/ciclo | veredito |
|---|---|---|---|---|
| **manter (adotada)** | 58,89 pp | 0,6646 | R$ 31.750 | ✅ passa no gate; disparidade **declarada** |
| limiar por grupo (0,05 para o grupo) | 6,33 pp | 0,6646 | R$ 34.668 | viável; **+R$ 2.918/ciclo**, +109 na fila, e é tratamento diferenciado explícito por atributo protegido |
| remover a coluna | 13,20 pp | **0,6427** | — | ❌ **reprova o gate** (piso 0,66) e nem resolve |

**Decisão adotada: manter o modelo e declarar a disparidade.** O motivo:
mitigá-la por limiar por grupo significa tratar pessoas de forma explicitamente
diferente com base em atributo protegido — o que é defensável como ação
afirmativa, mas é uma decisão que precisa de dono jurídico que este projeto não
tem. Registrar o número, o custo da correção e quem deveria decidir é mais
honesto que aplicar uma correção que ninguém autorizou.

🔑 **E a terceira linha é o argumento contra o reflexo mais comum da área.**
*Fairness through unawareness* — remover o atributo sensível — foi medido:
**atenua** (58,89 → 13,20 pp, porque o viés sobrevive nos proxies), **não
resolve** (continua acima de 10 pp) e **derruba o modelo abaixo do piso do gate**.
Cabe também a ressalva geral: um modelo pior tende a parecer mais justo — no
limite, um modelo aleatório tem disparidade zero.

### O que o CI garante — e o que não garante

`tests/test_fairness.py` **caracteriza** os quatro números acima (tolerância
1e-4): eles não mudam, para melhor ou para pior, sem um commit que alguém revisa.
E `test_atributos_sensiveis_continuam_nas_features` impede que alguém remova as
demográficas "para não discriminar" e apague a capacidade de auditar.

⚠️ **O que o CI NÃO faz: barrar os 10 pp.** Enquanto a decisão for "aceitar",
nenhum teste impede o modelo de continuar a 58,89 pp — e escrever um gate que
reprovasse o próprio modelo em produção só levaria a afrouxar o número até o
verde. É o Model Card que carrega este compromisso, e é por isso que o número
está na primeira tela.

## 8. Limitações e riscos

| limitação | consequência |
|---|---|
| **Métricas offline** | não há A/B nem conversão real da campanha; o valor de negócio é estimado, não medido |
| **Dataset sem eixo temporal** | o split é aleatório, não temporal; sazonalidade e tendência não são capturadas |
| **Sem dados de uso e de suporte** | a EDA mostrou que faltam as variáveis **comportamentais**, que são as mais preditivas em churn. Nenhuma feature engineering inventa o que não foi coletado |
| **População específica** | clientes residenciais da Califórnia; não transfere para PJ nem para outra geografia sem revalidação |
| **Feedback loop** | a campanha altera o rótulo de quem o modelo acertou. Sem **grupo de controle** (previsto na política de retreino, não implementado), a próxima avaliação é contaminada |
| **Janela cega do rótulo** | ~60 dias até o desfecho se confirmar (**pressuposto**, não medido) — a degradação só é observável com atraso; por isso o monitoramento em tempo real é de **drift**, não de performance |
| **Disparidade em `Dependents`** | ver seção 7 — assumida, declarada, com o custo da correção medido |
| **Sem autenticação na API** | serviço interno, limitação declarada; custo de fechá-la medido (+0,329 ms, ~25 linhas) |

## 9. Governança e conformidade

| item | como é tratado aqui |
|---|---|
| **Base legal (LGPD)** | legítimo interesse (retenção de cliente da própria base), **assumido** — num contexto real, validado pelo jurídico |
| **Minimização** | `CustomerID` fora das features; geográficas fora; 6 colunas removidas por ablação |
| **Logs** | log de inferência com máscara em dois níveis: `scores` **sempre** (número sem atributo não identifica), features **só com `TC_LOG_FEATURES=1`**. Decidido **antes** de ligar o log, porque log emitido não volta |
| **Art. 20 — revisão** | predição é apoio; a resposta traz score, decisão e limiar para que o humano possa discordar |
| **Art. 20 — explicação** | seção 6; o modelo é linear e a explicação é o próprio modelo |
| **Enquadramento de risco** | pelo AI Act europeu (referência, não vinculante aqui), churn com ação comercial **não** é alto risco — ao contrário de crédito ou saúde. Citado para deixar claro que o enquadramento foi verificado, não presumido |
| **Retenção dos logs** | fora do escopo: em produção quem retém é a plataforma (PaaS), com a política dela — limitação declarada |

### RACI — quem faria o quê num contexto real

| papel | quem seria | aqui |
|---|---|---|
| treina e avalia | engenheiro de ML | Luca |
| **aprova a promoção** | dono do produto | Luca |
| **aceita a disparidade** | dono do produto **+ jurídico** | Luca — 🚨 e a seção 7 registra que esta decisão não deveria ser de uma pessoa só |
| monitora | plataforma / dono do modelo | Luca |
| age sobre a predição | time de retenção | — (não existe) |

---

*Este Model Card é versionado junto do código. Os números da seção 5 são
verificados por `tests/test_gate.py` e os da seção 7 por `tests/test_fairness.py`
— se este arquivo divergir do modelo, o CI falha.*

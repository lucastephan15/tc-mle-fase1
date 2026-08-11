# Backlog de revisita

> **Para que serve:** o TC é construído **em paralelo** às aulas dos Módulos 02 a 05. Quando uma
> aula futura trouxer algo que afeta uma etapa **já fechada**, a regra é: **não refazer na hora.**
> Anota uma linha aqui e segue. No fim, varre-se o backlog de uma vez, com o conteúdo fresco e
> sem perder o fio do que já funcionava.
>
> Isso existe para resolver um risco específico e nomeado: *"ficar refazendo tudo do zero e me
> perder no que estou acrescentando/editando"*. As outras duas metades da solução são o **git**
> (o que mudou) e o **decision log** (por que mudou).

**Como usar:** uma linha por item, com a etapa afetada. Marcar `[x]` quando revisitado —
e se a revisita mudar uma decisão, a mudança vira linha no `decision-log.md`, não some aqui.

---

## Pendências abertas

| # | Aula / origem | O que revisitar | Etapa afetada | Status |
|---|---|---|---|---|
| 1 | M01-A08 | ~~Confirmar previsões feitas antes de abrir o dataset~~ → **4/4 acertadas** (7.043 linhas · churn 26,54% · `Total Charges` texto com 11 vazios · `CustomerID` descartável). Errei só o nº de colunas: 33, não 21 — é a variante estendida da IBM | 1 | ✅ |
| 2 | M02 (a estudar) | Algoritmos e técnicas de validação novas → candidatos para a Etapa 6 | 6, 7 | ⬜ |
| 3 | M02 (a estudar) | Conteúdo de redes neurais → arquitetura e regularização do MLP | 8 | ⬜ |
| 4 | M03 (a estudar) | Padrões de modularização, testes e clean code → **refatoração esperada** de `src/` | Fase 1, 9.5 | ⬜ |
| 5 | M04 (a estudar) | FastAPI a fundo → construir a API só depois deste módulo | 9 | ⬜ |
| 6 | M05 (a estudar) | Empacotamento e SDK → como distribuir o pipeline como biblioteca | 9, 11 | ⬜ |
| 7 | M01-A08 (§1 EDA) | Testar `Latitude`/`Longitude` como **experimento controlado**: performance E disparidade, com e sem. Escapam da cardinalidade (são contínuas), não da objeção ética | 4, 10.5 | ⬜ |
| 8 | M01-A08 (§1 EDA) | Investigar **por que fibra ótica churna 41,9% × DSL 19,0%** — contraintuitivo para o produto premium. Hipótese a testar: preço x expectativa de qualidade | 1, 4 | ⬜ |
| 9 | M01-A08 (§1 EDA) | Decidir o destino de `Total Charges` dado que correlaciona **0,9996** com `Tenure × Monthly` — manter para árvores, remover para LogReg, ou manter e aceitar | 5 | ⬜ |
| 10 | Enunciado FIAP | **Vídeo de 5 min (método STAR) vale 10%** e `pyproject.toml` + `Makefile` contam em qualidade de código (20%). Não são código — fáceis de esquecer | 11 | ⬜ |
| 11 | M01-A08 (§2) | 🚨 O arquivo bruto está **ordenado pelo alvo** → o "subset minúsculo" do CI leve **tem de ser estratificado**, senão o treino recebe uma classe só e o erro do sklearn não aponta a causa | 9.5 | ⬜ |
| 12 | M01-A08 (§3) | Congelar as **estatísticas de referência do treino** (`describe()` + frequências) junto do artefato — sem baseline não há drift a detectar depois | 9, 10 | ⬜ |
| 13 | M01-A08 (§3) | O limiar de operação (**0,22** no baseline) é parâmetro de negócio: precisa ser **configurável na API**, não constante no código | 9 | ⬜ |
| 14 | M01-A08 (§5) | Repetir a comparação com **validação cruzada estratificada** antes de declarar vencedor entre algoritmos — um split único não distingue ganho de variância | 6 | ⬜ |
| 15 | M02 (a estudar) | Avaliar **calibração** (Platt / isotônica) explicitamente: a fila ordena por `P(churn) × CLTV`, então probabilidade descalibrada corrompe a ordenação | 6, 7 | ⬜ |
| 16 | M01-A08 (§3 Etapa 4) | Reavaliar as 4 features com o **MLP** (`--features`). Expectativa baixa: a rede aprende interações sozinha, e elas já não ajudaram nem linear nem em árvore | 8 | ⬜ |
| 17 | M01-A08 (§3 Etapa 4) | 🚨 O `FunctionTransformer` acopla o artefato ao código-fonte (`skops_trusted_types`). Se alguma FE entrar no modelo final, o **Dockerfile e a API precisam ter `src.features` importável no mesmo caminho** | 9 | ⬜ |
| 18 | M01-A08 (§3 Etapa 4) | A RF *sem tuning* deu **0,6878** contra 0,6868 da LogReg — empate técnico. Não concluir nada disso na Etapa 6 sem tunar e sem CV repetida | 6 | ⬜ |
| 19 | M02-A01 (§2) | *(opcional — "se sobrar tempo")* Detector de drift **multivariado** com `IsolationForest` ao lado do PSI. Data drift **é** drift de `P(X)`, o objeto do não supervisionado; KS/qui²/PSI olham **uma coluna por vez** e são cegos ao caso em que só a **combinação** virou inédita (`tenure` alto **com** contrato mensal). O PSI já cumpre o requisito — isto é diferencial, não obrigação | 10 | ⬜ |
| 20 | M02-A01 (§8) | Escrever no decision log a frase de defesa do **"por que supervisionado?"**: o rótulo `Churn` já existe no histórico e a pergunta de negócio é `P(cancelar \| perfil)` → o objeto a estimar é `P(y\|X)`. Pergunta de abertura previsível da banca | 0, 11 | ⬜ |
| 21 | M02-A02 (§1) | 🔑 **Registrar no decision log** que `LogisticRegression()` traz `penalty='l2'`, `C=1.0` por default → o baseline **já é uma Ridge**, e é essa L2 que torna o `OneHotEncoder` **sem `drop`** seguro (dummies somam 1 = intercepto → `XᵀX` singular em OLS puro). Resposta pronta para *"não tem dummy trap aí?"*. Listar `get_params()` efetivos, não os digitados | 2, 3, 11 | ⬜ |
| 22 | M02-A02 (§2) | Etapa 7: `C` na grade em **escala logarítmica** (`np.logspace(-3,2,6)`) + `penalty` ∈ {l1, l2} como 2º eixo. ⚠️ `C = 1/λ` — **escala invertida** em relação a `alpha`. Expectativa a registrar *antes* de medir: com `n >> p`, `C` alto e ganho pequeno sobre 1.0 | 7 | ⬜ |
| 23 | M02-A02 (§3) | Etapa 9: `/health` vira **prontidão** (readiness), não só vitalidade — valida artefato carregado + **schema de features idêntico ao do treino** + versão. É onde o item **17** é verificado em runtime | 9 | ⬜ |
| 24 | M02-A02 (§5) | *(condicional)* Se a documentação for mostrar **coeficientes** para explicar a priorização, rodar **VIF** antes: multicolinearidade não derruba PR-AUC, mas infla erro-padrão e **inverte sinal**. Suspeitos: `Contract` (domina, 39,9 pp) e a redundância `Electronic check` × mês-a-mês (78%) | 11 | ⬜ |
| 25 | M02-A03 (§2) | 🎯 **Tabela de odds ratios** (`np.exp(coef_)`) das 13 features na documentação/vídeo — coeficiente cru não comunica, `exp(β)` sim ("mês-a-mês multiplica por X as chances"). Duas ressalvas em rodapé: **odds ratio ≠ risco relativo** (divergem com prevalência 26,5%) e coeficientes **encolhidos pela L2** ⇒ conservadores. Artefato mais barato/persuasivo da entrega | 11 | ⬜ |
| 26 | M02-A03 (§3) | Usar o argumento **teórico** na abertura das Etapas 6 e 8: a LogReg supõe **linearidade no log-odds** e é estruturalmente incapaz de efeito não monotônico; árvores e MLP não têm essa restrição. Melhor que "é exigência da fase". Contrafactual também vale: se os não-lineares não ganharem, o achado é *"o sinal é essencialmente linear no logit"* | 6, 8, 11 | ⬜ |
| 27 | M02-A03 (§4) | Etapa 8: usar **`BCEWithLogitsLoss`** (estável) e registrar que é **a mesma perda** do baseline (log-verossimilhança da LogReg com sinal trocado = entropia cruzada) ⇒ os dois modelos otimizam o mesmo objetivo, muda só a família de funções | 8, 11 | ⬜ |
| 28 | M02-A03 (§5) | ⚡ *(30 s)* Conferir **convergência do LBFGS**: `model.n_iter_` × `max_iter=1000`. Se não converge, o sklearn devolve os coeficientes onde parou e só avisa por `ConvergenceWarning` — **falha silenciosa** no output do MLflow | 3 | ⬜ |

## Decisões tomadas sob incerteza — reavaliar se surgir informação

| # | Decisão | Sob qual incerteza | O que a mudaria |
|---|---|---|---|
| 1 | Métrica primária = PR-AUC | assumindo que o consumo é uma fila priorizada | se o negócio exigisse decisão binária automática, voltaria a F2 com limiar fixo |
| 2 | Razão de custo FN:FP = 3:1 | premissas de ARPU, margem e conversão de campanha inventadas por nós | qualquer número real de operação refaz a tabela — e move o limiar de operação |
| 3 | Split aleatório estratificado | imposto pelo dataset ser um retrato sem eixo temporal | se aparecer versão longitudinal do dataset, split temporal passa a ser obrigatório |

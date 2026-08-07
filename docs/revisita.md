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
| 1 | M01-A08 | Confirmar as previsões feitas **antes** de abrir o dataset: ~7.043 × 21 linhas/colunas · churn ≈ 26,5% · `TotalCharges` como texto com ~11 vazios · `customerID` a descartar | 1 | ⬜ |
| 2 | M02 (a estudar) | Algoritmos e técnicas de validação novas → candidatos para a Etapa 6 | 6, 7 | ⬜ |
| 3 | M02 (a estudar) | Conteúdo de redes neurais → arquitetura e regularização do MLP | 8 | ⬜ |
| 4 | M03 (a estudar) | Padrões de modularização, testes e clean code → **refatoração esperada** de `src/` | Fase 1, 9.5 | ⬜ |
| 5 | M04 (a estudar) | FastAPI a fundo → construir a API só depois deste módulo | 9 | ⬜ |
| 6 | M05 (a estudar) | Empacotamento e SDK → como distribuir o pipeline como biblioteca | 9, 11 | ⬜ |

## Decisões tomadas sob incerteza — reavaliar se surgir informação

| # | Decisão | Sob qual incerteza | O que a mudaria |
|---|---|---|---|
| 1 | Métrica primária = PR-AUC | assumindo que o consumo é uma fila priorizada | se o negócio exigisse decisão binária automática, voltaria a F2 com limiar fixo |
| 2 | Razão de custo FN:FP = 3:1 | premissas de ARPU, margem e conversão de campanha inventadas por nós | qualquer número real de operação refaz a tabela — e move o limiar de operação |
| 3 | Split aleatório estratificado | imposto pelo dataset ser um retrato sem eixo temporal | se aparecer versão longitudinal do dataset, split temporal passa a ser obrigatório |

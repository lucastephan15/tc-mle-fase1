# Roteiro do vídeo — 5 minutos, método STAR

> **Como usar este arquivo.** O texto em blocos `FALA` está escrito para ser dito, não lido:
> frases curtas, sem subordinada longa, com o número sempre antes da explicação. Ensaie duas
> vezes em voz alta e depois **fale**, não recite — se sair 5% diferente, melhor.
> **Orçamento total: ~700 palavras faladas — medido, não estimado.** Português falado com naturalidade roda a
> ~145 palavras/minuto; passar de 800 palavras é o que faz o vídeo estourar os 5 minutos.

---

## ⚠️ Checklist de pré-gravação (fazer 10 minutos antes)

- [ ] 🚨 **ACORDAR A API.** `curl https://tc-churn-api.onrender.com/health` — o plano gratuito
      dorme após 15 min e a primeira chamada leva ~30 s. **Gravar com a API dormindo trava a
      demonstração no primeiro `curl`.** Repetir a cada 10 minutos até começar a gravar.
- [ ] Abas abertas, **nesta ordem**: (1) `docs/RELATORIO.md` no GitHub · (2) `docs/figuras/curva-ganho.png`
      em tela cheia · (3) terminal com o `curl` já digitado (não digitar no vídeo) ·
      (4) `https://tc-churn-api.onrender.com/docs` · (5) a página do Actions com o CI verde ·
      (6) `MODEL_CARD.md` na seção 7 (fairness).
- [ ] Terminal com fonte grande (≥16 pt) e tema claro — vídeo comprimido come texto pequeno.
- [ ] Rodar `make reportar` uma vez antes, para a saída estar no scrollback caso precise.
- [ ] Notificações desligadas. Microfone testado com 15 s de gravação de teste.
- [ ] Cronômetro visível fora da área capturada.

---

## Mapa de tempo

| bloco | tempo | tela | palavras |
|---|---|---|---|
| **S** · Situação | 0:00 – 0:40 | rosto / slide simples com o número do churn | 84 |
| **T** · Tarefa | 0:40 – 1:15 | tabela do enquadramento (relatório §1) | 96 |
| **A** · Ação | 1:15 – 3:35 | tabela mestra → `make reportar` → `curl` → CI | 352 |
| **R** · Resultado | 3:35 – 4:45 | curva de ganho + a conta em reais + fairness | 173 |
| Fechamento | 4:45 – 5:00 | rosto | 47 |
| **total** | **5:00** | | **752** |

---

## S · SITUAÇÃO — 0:00 a 0:40

**[TELA: você, ou um slide com "26,5%" grande]**

> **FALA**
>
> Numa operadora de telecom, mais de um em cada quatro clientes cancela. Nesta base, 26,5%.
>
> A equipe de Retenção trabalha cerca de 10% da carteira por ciclo. Então a pergunta do negócio
> nunca foi *"esse cliente vai cancelar, sim ou não?"* — é **"quem eu ligo primeiro?"**.
>
> E os dois erros custam diferente: deixar escapar quem ia cancelar custa **194 reais**; dar
> atenção a quem ficaria custa **62**. Assimetria de três para um — e é ela que define tudo o
> que vem depois.

---

## T · TAREFA — 0:40 a 1:15

**[TELA: a tabela de enquadramento — alvo, janela, métrica, custo]**

> **FALA**
>
> Minha tarefa foi o ciclo completo: do enquadramento do problema até uma API em produção,
> passando por uma rede neural em PyTorch, exigência da fase.
>
> A decisão mais importante veio primeiro: o artefato é um **ranqueador**, não um classificador.
> Por isso a métrica é **PR-AUC**, que não depende de limiar — e por isso o limiar de corte é
> **parâmetro de negócio**: a API devolve probabilidade, e a equipe muda onde corta a fila sem
> retreinar nada.
>
> E toda métrica aparece com o **piso** ao lado. Aqui, um modelo que responde "ninguém sai" já
> acerta 73%.

---

## A · AÇÃO — 1:15 a 3:35

### A.1 — O percurso de modelagem *(1:15 – 2:15)*

**[TELA: a tabela mestra de experimentos, rolando devagar]**

> **FALA**
>
> Baseline: regressão logística, PR-AUC 0,66 contra piso de 0,27. Sinal existe.
>
> Depois testei tudo o que poderia superá-la: feature engineering, Random Forest, boosting,
> tuning. Nada bateu o baseline — os ensembles empatam dentro de 0,07 desvio-padrão, e o tuning
> deu ganho **exatamente zero**, porque o default da biblioteca já era o pico da grade. Eu tinha
> escrito essa previsão **antes** de rodar.
>
> E a rede neural também não superou. Só que "não superou" tem duas leituras: ou não existe
> não-linearidade, ou meu experimento não a detectaria. Então rodei o **mesmo desenho** num
> dataset sintético não-linear: lá a rede ganha quase **10 pontos**. Aqui, nada.
>
> Quatro medições independentes dizendo o mesmo: **o sinal deste dataset é essencialmente
> linear**. Modelo simples por evidência, não por preguiça.

### A.2 — O conjunto de teste *(2:15 – 2:45)*

**[TELA: saída do `make reportar`]**

> **FALA**
>
> O conjunto de teste ficou **intocado por nove dias**: toda escolha saiu da validação. Decidir
> vinte vezes olhando o teste o transforma em validação, e o sintoma é traiçoeiro — o gap
> continua bonito.
>
> Toquei o teste **uma vez**, no fim. E o achado está no intervalo, não no ponto: ele tem 0,10
> de largura, e os seis finalistas cabiam em 0,006. **Nenhum** conjunto de teste desse tamanho
> poderia desempatá-los — o empate não era indecisão minha, é a resolução da amostra.

### A.3 — A engenharia *(2:45 – 3:35)*

**[TELA: `curl` na API pública → resposta JSON → `/docs` → CI verde]**

> **FALA**
>
> Daqui em diante o trabalho é engenharia: a entrega não é o modelo, é um **serviço**.
>
> **[executar o curl]** Esta é a API em produção. Ela devolve probabilidade, decisão e o limiar
> aplicado.
>
> Três decisões. O contrato de entrada é **derivado do artefato**: sem isso, um `-999` — a
> sentinela de nulo mais comum em sistema legado — devolve **200 OK com a probabilidade
> errada**, e 62% dos clientes trocam de lado na fila. Com o contrato, é 422.
>
> O `/health` declara **qual modelo está carregado**, por hash — é o caso Knight Capital: 440
> milhões perdidos porque o que rodava não era o que tinha sido validado.
>
> E o gate do CI tem **dois eixos**, desempenho e calibração — porque um modelo pode ordenar
> melhor e calibrar pior ao mesmo tempo, e aí o limiar corta a fila no lugar errado com o CI
> verde.

---

## R · RESULTADO — 3:35 a 4:45

**[TELA: a curva de ganho cumulativo, em tela cheia]**

> **FALA**
>
> Este é o gráfico que resume a entrega, e tem três linhas de propósito. Embaixo, o acaso. Em
> cima, o **teto estrutural** — o máximo que qualquer modelo poderia capturar, porque nos 10%
> do topo só cabem 10% dos clientes. No meio, o modelo.
>
> **Contatando 10% da base, a campanha alcança 28,6% de quem ia cancelar** — quase 76% do máximo
> possível. Sem a linha do teto, esse número pareceria fraco.
>
> Em reais: o erro custa **32 mil** por ciclo, contra **72 mil** de não fazer nada e **64 mil**
> de ligar para a base inteira — ligar para todo mundo captura 100% dos churners e **também é
> caro**.
>
> E um resultado que não me favorece: a auditoria de fairness encontrou **59 pontos** de
> disparidade de recall entre clientes com e sem dependentes. Corrigir por limiar por grupo
> funciona e custa 2.900 por ciclo — mas é tratamento diferente por atributo protegido, o que
> exige um dono jurídico que este projeto não tem. Decisão: **manter e declarar**, com o número
> no Model Card.

---

## Fechamento — 4:45 a 5:00

**[TELA: você]**

> **FALA**
>
> O maior limite aqui não é o algoritmo, é o dado: falta comportamento de uso e contato com o
> suporte — e nenhuma feature inventa o que não foi coletado.
>
> Está tudo no repositório: relatório, Model Card e o registro de decisões escrito **durante** a
> execução. Obrigado.

---

## 🎤 Perguntas prováveis da banca — respostas de 20 segundos

*(não entram no vídeo; existem para você não ser pego de surpresa na defesa)*

**"Por que não usou a rede neural, se a fase exige?"**
> Usei — está implementada em PyTorch, avaliada sob o mesmo protocolo, e o número está no
> relatório. Ela não superou, e a literatura de dados tabulares prevê exatamente isso: eu
> escrevi essa previsão no repositório **antes** de rodar. E demonstrei que o experimento
> detectaria o ganho se ele existisse, com um controle positivo.

**"Seu PR-AUC de 0,65 é baixo, não?"**
> Contra o piso de 0,27, é 2,4×. E a leitura que importa é a operacional: 76% do máximo
> estruturalmente possível no ponto em que a campanha opera. O que subiria esse número é dado
> comportamental, não algoritmo — e isso está medido em quatro etapas.

**"O piso do seu gate é 0,66 e o teste deu 0,6496. Não reprovou?"**
> O gate mede a **validação**, por decisão registrada desde a Etapa 2 — gate no teste o
> converte em validação. E eu **não** movi o piso depois de ver o teste: seria mover o limite
> depois do resultado, exatamente o que o pré-registro existe para impedir. O gap de 0,015 cabe
> folgado no intervalo de confiança.

**"Por que uma API se churn é caso de batch?"**
> Concordo, e está escrito assim no decision log. A API existe para integração com o CRM e para
> o requisito da fase — e por isso o endpoint principal é o de **lote**: 1.409 clientes custam
> 2,9 ms em lote contra 2.353 ms um a um.

**"Como você sabe que o que está no ar é o modelo que você avaliou?"**
> Por hash. O `/health` declara o sha256 do artefato, ele vai em cada resposta e em cada linha
> de log, e o processo **morre na inicialização** se o artefato não bater. Rodei o mesmo script
> de integração contra a URL pública: PR-AUC idêntico nos dez dígitos.

**"E se o modelo começar a errar daqui a seis meses?"**
> Log estruturado com o hash do artefato, baseline de drift congelado **dentro** do artefato, e
> um detector que eu **verifiquei disparando** — fabriquei dois cenários de drift, porque
> esperar seis meses não era opção. Política de retreino com gatilho e desarme, e rollback em
> duas camadas: a plataforma para o sangramento, o `git revert` torna definitivo.

**"Removeu `gender` para evitar viés?"**
> Não, de propósito. Remover o atributo não remove o viés — remove a **capacidade de medi-lo**.
> Aliás, medi: remover `Dependents` levaria a disparidade de 59 para 13 pontos, então atenua
> mesmo; o que mata essa opção é que derruba a PR-AUC abaixo do piso do gate.

---

## ❌ Erros a evitar na gravação

1. **Não abrir com "olá, meu nome é..."** — os primeiros 10 segundos são os que prendem. Abra
   com o número.
2. **Não fazer tour de código.** A banca não avalia se você sabe abrir arquivos; avalia se você
   sabe defender decisões. Toda tela existe para sustentar uma frase.
3. **Não digitar no vídeo.** Comando já digitado, só apertar Enter.
4. **Não ler a tabela em voz alta.** Diga o que ela mostra; ela está aí para quem lê rápido.
5. **Não estourar o tempo.** Se algum bloco estiver longo no ensaio, corte da **A.1** — é a
   parte mais fácil de resumir sem perder o argumento.
6. **Não esconder o resultado ruim.** A disparidade de fairness é o item que mais diferencia a
   entrega; ela mostra que a auditoria foi feita para **descobrir**, não para confirmar.

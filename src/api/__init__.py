"""
A API de inferência — Etapa 9d.

Três camadas, separadas por **uma razão para mudar** (SRP):

| módulo | responsabilidade | muda quando |
|---|---|---|
| `schema.py` | o contrato de entrada e saída (Pydantic) | muda o contrato de features |
| `servico.py` | carregar o artefato, aplicar o limiar, pontuar | muda o modelo ou a regra de negócio |
| `app.py` | HTTP: rotas, status, erros, `/health` | muda o contrato com o cliente |

O ganho não é estético: é `servico.py` poder ser testado sem subir servidor, e
as rotas poderem ser testadas com um dublê de três linhas, sem tocar o disco.

⛔ **Sem Strategy/Factory/Observer aqui.** Um endpoint de predição não passa no
teste do próprio padrão (*só paga se as estratégias forem numerosas ou
complexas*): uma função que carrega e uma que pontua cobrem tudo. Recusar o
padrão é o conteúdo da aula sobre padrões, não uma esquiva dele.

Para servir:

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

# 🏦 Sistema de Escoragem de Risco de Crédito (*Credit Scoring*) — Home Credit

> **Projeto de Engenharia de Dados e Machine Learning** voltado à estimativa de probabilidade de inadimplência (*Probability of Default - PD*) e ao apoio à decisão na concessão de crédito, estruturado segundo a metodologia **CRISP-DM** com foco na construção de um **MVP (*Minimum Viable Product*)** funcional de ponta a ponta.

---

## 🌐 Acesso à Aplicação Publicada (Ambiente de Demonstração)

A solução encontra-se publicada em ambiente de nuvem gratuito para testes e simulações em tempo real:

* **🖥️ Aplicação Web (Mesa de Crédito — Streamlit):** [https://home-credit-default-risk-jkr.streamlit.app/](https://home-credit-default-risk-jkr.streamlit.app/)
* **🔌 Documentação Interativa da API (FastAPI — Swagger UI):** [https://home-credit-risk-api-e6n6.onrender.com/docs](https://home-credit-risk-api-e6n6.onrender.com/docs)

> **Nota sobre o ambiente gratuito:** A API está hospedada na camada gratuita do *Render*, que entra em modo de espera após 15 minutos de inatividade. Na primeira requisição, o serviço pode levar cerca de 40 a 50 segundos para ser reativado; as consultas subsequentes respondem normalmente em milissegundos.

---

## 📌 1. Contexto de Negócio e Metodologia (CRISP-DM)

Na concessão de crédito ao consumidor, busca-se equilibrar dois objetivos complementares: **ampliar o acesso ao crédito** para bons pagadores (incluindo clientes com histórico bancário reduzido) e **controlar a exposição ao risco de inadimplência (*default*)**.

Neste projeto, utilizou-se a base histórica da *Home Credit* ($307.511$ solicitações de empréstimo acompanhadas de registros de birô de crédito) para estruturar um fluxo analítico com três propósitos principais:

1. Consolidar dados cadastrais e histórico de mercado (*Bureau*) em uma tabela analítica padronizada (*Analytical Base Table - ABT*).
2. Estimar a probabilidade de inadimplência (`TARGET = 1`) por meio de um conjunto enxuto de variáveis alinhadas aos **5 Cs do Crédito**.
3. Combinar o *score* probabilístico do modelo com regras prudenciais de negócio, organizando a recomendação operacional em três faixas de decisão.

---

## 🗺️ 2. Etapas do Projeto (Ciclo CRISP-DM)

O desenvolvimento foi dividido em cinco fases sequenciais, conectando a preparação dos dados à disponibilização do serviço:

- [x] **Fase 1 — Entendimento dos Dados e Engenharia ELT (Arquitetura Medalhão):** Ingestão de dados brutos (*Bronze*), higienização e padronização via **PySpark** (*Silver*) e agregação relacional em formato colunar `.parquet` (*Gold*).
- [x] **Fase 2 — Análise Exploratória (EDA) e Seleção de Variáveis:** Investigação das distribuições estatísticas, avaliação de valores ausentes e correlações, e definição das 12 variáveis do MVP com base nos **5 Cs do Crédito**.
- [x] **Fase 3 — Modelagem Preditiva e Calibração de Limiares:** Construção do pipeline de pré-processamento (`Pipeline` + `ColumnTransformer`), avaliação comparativa de algoritmos com validação cruzada estratificada e definição das faixas operacionais de corte.
- [x] **Fase 4 — Serviço de Inferência REST e Containerização:** Disponibilização do pipeline via **FastAPI** (predições unitárias e em lote), validação de contratos de entrada com **Pydantic**, testes automatizados com **Pytest** e empacotamento em **Docker**.
- [x] **Fase 5 — Simulação de Negócio e Interface Corporativa:** Aplicação interativa em **Streamlit** integrando o *score* estatístico a travas de política institucional (*Hard Rules*), orquestrada localmente via **Docker Compose** e publicada em nuvem.

---

## 🏗️ 3. Engenharia de Dados e Arquitetura Medalhão (*ELT*)

Para processar os $307.511$ contratos principais (`application_train.csv`) em conjunto com $1.716.428$ registros históricos de créditos em outras instituições (`bureau.csv`), implementou-se uma esteira local em **PySpark** documentada no notebook `notebooks/ELT.ipynb`.

### Estrutura das Camadas

* **Camada Bronze (`data/bronze/`):** Armazenamento dos arquivos CSV originais sem alterações semânticas.
* **Camada Silver (`data/silver/`):** Padronização de tipagem e tratamento de inconsistências cadastrais e contábeis:
  * Conversão de datas relativas negativas (`DAYS_BIRTH` e `DAYS_EMPLOYED`) para escala positiva em anos (`AGE_YEARS` e `YEARS_EMPLOYED`).
  * Substituição do código sentinela `365243` em `DAYS_EMPLOYED` por `Null` (valor presente majoritariamente em registros de aposentados e proponentes sem vínculo ativo).
  * Ajuste de saldos devedores negativos na base do birô (`AMT_CREDIT_SUM_DEBT < 0` ajustados para `0.0`) e conversão de flags textuais (`'Y'`/`'N'`) em inteiros binários (`1`/`0`).
  * Manutenção dos valores ausentes originais nas variáveis contínuas para que a imputação ocorra apenas dentro do pipeline de treinamento.
* **Camada Gold (`data/gold/abt_credit_risk.parquet`):** Sumarização prévia da tabela `bureau` por proponente (`SK_ID_CURR`) seguida de `LEFT JOIN` com a tabela de propostas, preservando a proporção de $1:1$ ($307.511$ linhas e $20$ colunas) e evitando duplicação de registros. Os proponentes sem histórico prévio no birô ($14,31\%$ da base) tiveram suas contagens e somas de mercado preenchidas com zero.

### Atributos Derivados na Camada Gold

* `BUREAU_TOTAL_LOANS` e `BUREAU_ACTIVE_LOANS`: Total de contratos registrados no birô e quantidade de contratos vigentes (`Active`).
* `BUREAU_TOTAL_CREDIT_SUM` e `BUREAU_TOTAL_CREDIT_DEBT`: Somatório dos valores contratados e do saldo devedor atual no mercado.
* `DTI_RATIO` (*Debt-to-Income*): Razão entre o valor da anuidade/parcela e a renda declarada ($\frac{\text{AMT\_ANNUITY}}{\text{AMT\_INCOME\_TOTAL}}$).
* `CREDIT_INCOME_RATIO`: Razão entre o montante total solicitado e a renda anual ($\frac{\text{AMT\_CREDIT}}{\text{AMT\_INCOME\_TOTAL}}$).

---

## 📊 4. Análise Exploratória (EDA) e Seleção de Variáveis (5 Cs do Crédito)

O estudo exploratório conduzido em `notebooks/EDA_modeling_pipeline.ipynb` orientou as escolhas de pré-processamento e a redução de dimensionalidade para o serviço de inferência.

### 4.1. Proporção da Variável Alvo (`TARGET`)

A base apresenta uma taxa histórica de inadimplência (*Bad Rate*) de $8,07\%$ ($24.825$ contratos inadimplentes) frente a $91,93\%$ de contratos adimplentes ($282.686$), resultando em uma proporção de $11,39 : 1$.

* **Abordagem adotada:** Como a acurácia simples é pouco informativa em cenários desbalanceados, utilizou-se divisão estratificada (`stratify=y`), ponderação de classes nos algoritmos e avaliação baseada na capacidade de ordenação de risco (**ROC-AUC**, **PR-AUC** e **Estatística KS**).

### 4.2. Distribuições Assimétricas e Tratamento de Nulos

Variáveis monetárias como `AMT_INCOME_TOTAL` (mediana de $147.150$ e máximo de $117.000.000$) e `BUREAU_TOTAL_CREDIT_DEBT` apresentam forte assimetria à direita. Paralelamente, variáveis relevantes possuem ausências legítimas de negócio, como `EXT_SOURCE_3` ($19,83\%$ de nulos para clientes sem score naquele birô) e `YEARS_EMPLOYED` ($18,01\%$ de nulos).

* **Abordagem adotada:** Para reduzir a sensibilidade a valores extremos, optou-se pela **imputação numérica via mediana (`SimpleImputer(strategy='median')`)** acoplada ao escalonamento padronizado (`StandardScaler`) dentro do pipeline de validação.

### 4.3. Comportamento Bivariado Observado na Amostra

Na comparação entre os grupos adimplente (`TARGET = 0`) e inadimplente (`TARGET = 1`), destacam-se as seguintes tendências medianas:

* Pontuações externas menores entre inadimplentes: mediana de `EXT_SOURCE_3` $30,57\%$ menor ($0,3791$ vs. $0,5460$) e de `EXT_SOURCE_2` $23,27\%$ menor ($0,4404$ vs. $0,5739$).
* Menor tempo médio de vínculo empregatício (`YEARS_EMPLOYED` com mediana de $3,37$ anos vs. $4,63$ anos) e faixa etária ligeiramente mais jovem ($39,1$ anos vs. $43,5$ anos).
* Maior alavancagem simultânea no birô: mediana de $2,0$ empréstimos ativos (`BUREAU_ACTIVE_LOANS`) e saldo devedor mediano $59,15\%$ superior ($133.812$ vs. $84.078$).

### 4.4. Contrato de Entrada do MVP (12 Variáveis e os 5 Cs do Crédito)

Visando manter a interface da API enxuta e reduzir redundâncias (como a alta colinearidade entre `AMT_GOODS_PRICE` e `AMT_CREDIT`, e entre `BUREAU_TOTAL_LOANS` e `BUREAU_ACTIVE_LOANS`), foram selecionadas **12 variáveis** distribuídas conforme os **5 Cs do Crédito**:

| Variável no Contrato | Nome de Negócio | Pilar dos 5 Cs | Descrição Resumida |
| :--- | :--- | :--- | :--- |
| `EXT_SOURCE_2` | Score Externo 2 | Capital / Condições | Pontuação normalizada de birô externo ($0,0$ a $1,0$). |
| `EXT_SOURCE_3` | Score Externo 3 | Capital / Condições | Segunda pontuação de birô externo ($0,0$ a $1,0$, aceita nulo). |
| `AMT_CREDIT` | Valor do Crédito | Capacidade | Montante total solicitado na proposta. |
| `AMT_INCOME_TOTAL` | Renda Total Anual | Capacidade | Renda anual bruta declarada pelo proponente. |
| `DTI_RATIO` | Comprometimento de Renda (DTI) | Capacidade | Proporção da renda comprometida com a parcela (aceita nulo). |
| `AGE_YEARS` | Idade | Caráter | Idade do proponente em anos na data da análise. |
| `YEARS_EMPLOYED` | Tempo de Emprego | Caráter | Anos no vínculo atual (aceita nulo para aposentados/sem vínculo). |
| `NAME_EDUCATION_TYPE` | Escolaridade | Caráter | Nível de escolaridade informado (5 categorias). |
| `NAME_INCOME_TYPE` | Tipo de Vínculo / Renda | Caráter / Capacidade | Natureza da fonte de renda (8 categorias). |
| `BUREAU_ACTIVE_LOANS` | Empréstimos Ativos no Birô | Condições / Histórico | Quantidade de contratos ativos em outras instituições. |
| `BUREAU_TOTAL_CREDIT_DEBT` | Dívida Total Ativa no Birô | Capital | Soma do saldo devedor em aberto no mercado. |
| `FLAG_OWN_CAR` | Possui Veículo Próprio | Colateral | Indicativo binário de posse de veículo ($1 = \text{Sim}$, $0 = \text{Não}$). |

---

## 🤖 5. Modelagem e Avaliação Estatística

### 5.1. Estrutura de Pré-Processamento e Validação

Os dados foram divididos de forma estratificada em conjuntos de treino, validação e teste (*holdout*). As transformações de imputação e codificação categórica (`OneHotEncoder(handle_unknown='ignore')`) foram encapsuladas em um `Pipeline` único do Scikit-Learn para que os parâmetros fossem estimados apenas sobre o conjunto de treinamento.

Foram comparados diferentes algoritmos lineares e baseados em árvores (como *Logistic Regression*, *Random Forest*, *HistGradientBoosting* e *CatBoost*), utilizando validação cruzada estratificada. O **CatBoost Classifier** apresentou o conjunto mais equilibrado de resultados nas métricas de ordenação de risco e generalização, sendo escolhido para compor o artefato final (`models/credit_risk_pipeline.joblib`).

### 5.2. Desempenho no Conjunto de Teste e Faixas Operacionais

Na partição de teste isolada, o modelo selecionado registrou:

* **ROC-AUC:** $\approx 0,735$ (com intervalo de confiança via *bootstrap* documentado em `models/model_metrics.json`).
* **Estatística KS:** $\approx 35,1\%$ de separação máxima entre as distribuições acumuladas de bons e maus pagadores.

Para apoiar a operação bancária de forma mais flexível do que uma decisão binária simples, os scores probabilísticos foram divididos em três faixas operacionais (*Semáforo de Risco*), calibradas a partir das curvas de validação:

| Faixa de Score (`prob`) | Classificação Operacional | Recomendação de Negócio |
| :---: | :--- | :--- |
| **`< 0.5027`** | 🟢 **APROVADO AUTOMÁTICO** | Proposta dentro do apetite de risco para aprovação no fluxo automatizado. |
| **`0.5027` a `0.6399`** | 🟡 **ANÁLISE ADICIONAL / CONDICIONAL** | Faixa intermediária (*zona cinzenta*); sugere avaliação manual, comprovação adicional ou ajuste de valor/garantia. |
| **`>= 0.6400`** | 🔴 **REPROVADO AUTOMÁTICO** | Concentração elevada de risco de inadimplência; recomenda-se a recusa nas condições solicitadas. |

Na análise de importância das variáveis do modelo (`feature_importance_contract_pct`), observa-se que os indicadores externos de mercado (`EXT_SOURCE_2` e `EXT_SOURCE_3`) concentram a maior parcela de contribuição preditiva, atuando em conjunto com os indicadores de alavancagem (`AMT_CREDIT`, `DTI_RATIO`, `BUREAU_TOTAL_CREDIT_DEBT`) e estabilidade (`AGE_YEARS`, `YEARS_EMPLOYED`).

---

## 📂 6. Estrutura do Repositório

```text
├── data/                           # Diretório local de dados [Ignorado no Git]
│   ├── bronze/                     # Arquivos CSV brutos (application_train.csv, bureau.csv)
│   ├── silver/                     # Tabelas tratadas (application_train.parquet, bureau.parquet)
│   └── gold/                       # Tabela Analítica Final (abt_credit_risk.parquet)
├── models/
│   ├── credit_risk_pipeline.joblib # Pipeline serializado (Pré-processamento + CatBoost)
│   └── model_metrics.json          # Métricas de avaliação, hiperparâmetros e limiares de corte
├── notebooks/
│   ├── ELT.ipynb                   # Pipeline PySpark (Bronze -> Silver -> Gold ABT)
│   └── EDA_modeling_pipeline.ipynb # EDA, seleção de variáveis, treinamento e avaliação
├── outputs/                        # Gráficos gerados durante a análise e avaliação do modelo
├── app/                            # Serviço REST em FastAPI
│   ├── main.py                     # Rotas da API (/health, /metadata, /predict, /predict/batch)
│   ├── schemas.py                  # Contratos de entrada e saída em Pydantic V2
│   └── model.py                    # Serviço de carregamento do modelo e inferência vetorizada
├── frontend/                       # Interface em Streamlit (Mesa de Crédito)
│   ├── app.py                      # Aplicação visual, tradução PT-BR e regras de política
│   ├── Dockerfile                  # Imagem Docker do front-end (porta 8501)
│   ├── requirements.txt            # Dependências do front-end
│   └── .streamlit/
│       └── config.toml             # Configuração de tema visual
├── tests/
│   └── test_api.py                 # Testes automatizados da API com Pytest
├── Dockerfile                      # Imagem Docker da API (porta 8000)
├── docker-compose.yml              # Orquestração local dos contêineres (API + Front-End)
├── requirements.txt                # Dependências da API e modelagem
└── README.md                       # Documentação geral do projeto
```

---

## 🔌 7. Serviço de Inferência (API FastAPI)

O serviço de back-end (`app/`) carrega o artefato `.joblib` e os metadados durante o ciclo de inicialização (`lifespan`) e valida os dados recebidos com **Pydantic V2** (`extra="forbid"`, restrições de domínio e aceitação de valores `null` nas variáveis opcionais).

| Método | Endpoint | Função |
| :---: | :--- | :--- |
| `GET` | `/health` | Verifica a disponibilidade do serviço e se o modelo está carregado em memória. |
| `GET` | `/metadata` | Retorna as métricas oficiais, hiperparâmetros e limiares registrados em `model_metrics.json`. |
| `POST` | `/predict` | Realiza a inferência para uma proposta individual. |
| `POST` | `/predict/batch` | Realiza a inferência vetorizada para uma lista de propostas em uma única chamada. |

**Exemplo de Payload (`POST /predict`):**
```json
{
  "EXT_SOURCE_2": 0.55,
  "EXT_SOURCE_3": 0.50,
  "AMT_CREDIT": 450000.0,
  "YEARS_EMPLOYED": 6.5,
  "AGE_YEARS": 38.0,
  "DTI_RATIO": 0.28,
  "AMT_INCOME_TOTAL": 180000.0,
  "BUREAU_TOTAL_CREDIT_DEBT": 35000.0,
  "BUREAU_ACTIVE_LOANS": 2,
  "FLAG_OWN_CAR": 1,
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_INCOME_TYPE": "Working"
}
```

---

## 🖥️ 8. Interface Visual (Streamlit) e Decisão em Dupla Camada

A aplicação **Mesa de Crédito** (`frontend/app.py`) oferece uma interface em português voltada a analistas de crédito, dividida em três abas:

1. **Análise de Proposta Individual:** Formulário organizado pelos 5 Cs do Crédito, com botões de simulação rápida (*presets* de baixo, médio e alto risco) e tradução interna dos domínios categóricos para o formato esperado pela API.
2. **Análise de Lotes (Batch via CSV):** Permite baixar uma planilha modelo, enviar múltiplas propostas simultaneamente para o endpoint `/predict/batch` e exportar o relatório avaliado.
3. **Dicionário de Dados & Política de Risco:** Consulta às descrições das 12 variáveis, faixas do semáforo e metadados técnicos do modelo.

### Regras de Política Institucional (*Hard Rules*)

Em operações reais de crédito, modelos estatísticos atuam em conjunto com travas de política institucional, pois um *score* histórico favorável de birô pode não refletir uma restrição imediata de capacidade de pagamento. Na interface, implementou-se uma segunda camada de verificação sobre o retorno do modelo:

| Regra de Política | Condição Verificada | Efeito na Decisão Final |
| :--- | :--- | :--- |
| **Ausência de Renda Ativa** | `NAME_INCOME_TYPE == "Unemployed"` (Desempregado) | 🔴 Reclassifica para **Reprovado Automático** |
| **Alavancagem Elevada** | `AMT_CREDIT > 10 × AMT_INCOME_TOTAL` | 🔴 Reclassifica para **Reprovado Automático** |
| **DTI Mensal Crítico** | `DTI_RATIO > 50%` | 🔴 Reclassifica para **Reprovado Automático** |
| **DTI Mensal em Atenção** | `35% < DTI_RATIO ≤ 50%` (quando o modelo indica aprovação) | 🟡 Encaminha para **Análise Adicional / Condicional** |

As regras de política atuam de forma conservadora (podem tornar a decisão mais cautelosa, mas não convertem uma reprovação estatística em aprovação). Quando uma trava é acionada, o painel exibe tanto o **Score Estatístico Puro do Modelo** quanto o motivo da sobreposição pela política institucional.

---

## 🚀 9. Instruções para Execução Local

### Opção A: Subir a Stack Completa via Docker Compose (Recomendado)

Com o Docker e o Docker Compose instalados, execute na raiz do repositório:

```bash
docker compose up --build
```

O contêiner do front-end aguarda a confirmação de saúde (`healthcheck`) da API antes de iniciar. Após a inicialização, acesse no navegador:

* **Front-End (Streamlit):** `http://localhost:8501`
* **API (Swagger UI):** `http://localhost:8000/docs`

Para encerrar os serviços: `docker compose down`.

### Opção B: Execução em Ambiente Virtual Python (Testes e Notebooks)

```bash
# 1. Clonar o repositório e ativar o ambiente virtual
git clone [https://github.com/JeanKahlilR/Home-Credit-Default-Risk.git](https://github.com/JeanKahlilR/Home-Credit-Default-Risk.git)
cd Home-Credit-Default-Risk
python -m venv .venv
source .venv/bin/activate

# 2. Instalar as dependências e executar a suíte de testes da API
pip install -r requirements.txt
pytest -v
```
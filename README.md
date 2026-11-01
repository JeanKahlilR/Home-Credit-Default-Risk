# 🏦 Sistema de Escoragem de Risco de Crédito (*Credit Scoring*) — Home Credit

> **Projeto de Engenharia de Machine Learning e Ciência de Dados** voltado à predição de probabilidade de inadimplência (*Probability of Default - PD*) e apoio automatizado à tomada de decisão na concessão de crédito, desenvolvido sob a metodologia **CRISP-DM** com foco em entrega de **MVP (*Minimum Viable Product*)** de alta performance.

---

## 📌 1. Contexto de Negócio e Metodologia (CRISP-DM)

Instituições financeiras enfrentam continuamente o dilema entre **maximizar a inclusão financeira** (aprovar bons pagadores que frequentemente possuem histórico de crédito limitado) e **proteger o capital contra o risco de calote (*default*)**.

Neste projeto, utilizou-se a base histórica da *Home Credit* ($307.511$ solicitações de empréstimo distribuídas em múltiplas tabelas relacionais) para o desenvolvimento de uma solução ponta a ponta voltada a:

1. Consolidar o histórico cadastral e de birô de crédito (*Bureau*) em uma tabela analítica íntegra (*Analytical Base Table - ABT*).
2. Estimar a probabilidade de atraso severo de parcelas (`TARGET = 1`) por meio de um **MVP enxuto fundamentado nos 5 Cs do Crédito**.
3. Traduzir o *score* matemático em uma **Política de Crédito de 3 Faixas (Semáforo de Decisão)** auditável e pronta para operação via API.

---

## 🗺️ 2. Roadmap de Execução (Fases do CRISP-DM)

O ciclo de vida do projeto foi estruturado conforme as etapas clássicas do **CRISP-DM**, conectando o entendimento de negócio ao *deploy* produtivo:

- [x] **Fase 1 — *Business & Data Understanding / Data Preparation* (Arquitetura Medallion):** Compreensão das regras de concessão, ingestão de dados relacionais brutos (*Raw/Bronze*), agregação SQL em memória via **DuckDB** (*Silver*) e construção de *features* de domínio financeiro em formato colunar `.parquet` (*Gold*).
- [x] **Fase 2 — *Exploratory Data Analysis & Feature Selection* (MVP & 5 Cs do Crédito):** Análise exploratória orientada a hipóteses, diagnóstico de assimetria e desbalanceamento, poda de multicolinearidade e definição do contrato enxuto do MVP ancorado nos **5 Cs do Crédito**.
- [x] **Fase 3 — *Modeling & Statistical Evaluation* (Torneio de Modelos & Política de Crédito):** Prevenção de vazamento de dados via `Pipeline` + `ColumnTransformer`, competição de algoritmos com validação cruzada estratificada, calibração de limiares de negócio ($F_1$, $F_2$ e KS) e validação de significância estatística (*Bootstrap* e Teste Z).
- [ ] **Fase 4 — *Deployment: Microsserviço REST & Containerização*:** Exposição do pipeline serializado em API REST (**FastAPI**), contratos estritos de validação de payload (**Pydantic**), suíte de testes automatizados (**Pytest**) e empacotamento reprodutível em **Docker**.
- [ ] **Fase 5 — *Business Simulation & Executive Pitch*:** Interface executiva interativa em **Streamlit** consumindo a API para simulação individual de propostas (Motor de Decisão) e avaliação financeira da carteira.

---

## 🏗️ 3. Engenharia de Dados e Arquitetura Medallion (*Data Preparation*)

Para o processamento de mais de $307\text{ mil}$ contratos principais (`application_train.csv`) cruzados com mais de $1,7\text{ milhão}$ de registros históricos de outras instituições financeiras (`bureau.csv`), implementou-se um pipeline analítico de alta performance utilizando **DuckDB** e armazenamento **Apache Parquet**.

### Estrutura das Camadas de Dados

* **Camada Bronze (`data/raw/`):** Dados brutos originais em CSV, preservados de forma imutável.
* **Camada Silver (`sql/create_abt.sql`):** Processamento relacional *in-memory* via DuckDB executando `LEFT JOIN` entre a proposta atual e as agregações históricas do birô por cliente (`SK_ID_CURR`), assegurando **cardinalidade estrita** $1:1$ (ausência de duplicação de linhas).
* **Camada Gold (`data/processed/abt_train.parquet`):** Tabela Analítica Final (*ABT*) devidamente tipada, saneada de anomalias cadastrais (ex.: conversão do marcador `365243` em `DAYS_EMPLOYED` para `NULL`) e enriquecida com variáveis de negócio, comprimida de centenas de megabytes para $\approx 13\text{ MB}$.

### Engenharia de Atributos (*Feature Engineering*) de Domínio Bancário

Além das variáveis originais de cadastro e birô, foram derivadas razões financeiras fundamentais para a análise de capacidade de pagamento:

* `AGE_YEARS` e `YEARS_EMPLOYED`: Conversão da contagem regressiva de dias (`DAYS_BIRTH` e `DAYS_EMPLOYED`) para escala positiva em anos.
* `CREDIT_TO_INCOME_RATIO`: Comprometimento total de renda em relação ao montante financiado ($\frac{\text{AMT\_CREDIT}}{\text{AMT\_INCOME\_TOTAL}}$).
* `ANNUITY_TO_INCOME_RATIO`: Comprometimento mensal da renda com a parcela do empréstimo ($\frac{\text{AMT\_ANNUITY}}{\text{AMT\_INCOME\_TOTAL}}$ — *Debt-to-Income / DTI*).
* `CREDIT_TO_GOODS_RATIO`: Razão de financiamento sobre o valor real do bem ($\frac{\text{AMT\_CREDIT}}{\text{AMT\_GOODS\_PRICE}}$ — *Loan-to-Value / LTV*).
* Agregações do Birô: `BUREAU_TOTAL_LOANS`, `BUREAU_ACTIVE_LOANS`, `BUREAU_TOTAL_CREDIT_SUM`, `BUREAU_TOTAL_DEBT_SUM` e `BUREAU_MAX_OVERDUE`.

---

## 📊 4. Análise Exploratória (EDA) e Design do MVP pelos 5 Cs do Crédito

Toda a arquitetura de pré-processamento e seleção de variáveis do modelo foi orientada por evidências estatísticas obtidas na etapa de EDA (`notebooks/02_model_training_and_evaluation.ipynb`).

### 4.1. Desbalanceamento Severo da Classe Alvo (`TARGET`)

A carteira apresenta uma taxa basal de inadimplência (*Bad Rate*) de $8,07\%$ ($24.825$ inadimplentes) contra $91,93\%$ de adimplentes ($282.686$ contratos), configurando uma razão de desbalanceamento de $11,39 : 1$.

* **Decisão Metodológica:** A métrica de *Acurácia* foi descartada (um classificador ingênuo obteria $91,93\%$ de acerto aprovando $100\%$ da base). Estabeleceu-se amostragem estratificada (`stratify=y`), ponderação de custo na função de perda (`class_weight='balanced'`) e avaliação orientada por **ROC-AUC**, **PR-AUC** e **Estatística KS**.

### 4.2. Assimetria Extrema (*Skewness*) e Imputação Resiliente

A análise descritiva evidenciou a presença de *outliers* severos nas variáveis monetárias — por exemplo, `AMT_INCOME_TOTAL` possui mediana de $\$147.150$, porém valor máximo de $\$117.000.000$ (*skewness* $= 391,56$), enquanto `BUREAU_TOTAL_CREDIT_SUM` apresenta *skewness* de $78,34$.

* **Decisão Metodológica:** Vetou-se o uso de imputação pela média aritmética. Todo o pipeline numérico adota **imputação pela mediana (`SimpleImputer(strategy='median')`)**, assegurando robustez frente a valores extremos.

### 4.3. Poder Discriminante Contínuo e Categórico

A análise bivariada evidenciou contrastes marcantes no perfil dos clientes inadimplentes (`TARGET = 1`):

* Redução expressiva nos scores de birôs externos: a mediana de `EXT_SOURCE_3` sofre redução de $-29,0\%$ ($0,379$ vs. $0,535$) e a de `EXT_SOURCE_2` de $-22,1\%$ ($0,440$ vs. $0,565$).
* Menor maturidade profissional e etária: mediana de tempo de emprego $-27,5\%$ inferior ($3,38$ anos vs. $4,66$ anos) e idade mediana mais baixa ($39,1$ anos vs. $43,8$ anos).
* O dobro de operações ativas no mercado: a mediana de `BUREAU_ACTIVE_LOANS` aumenta em $+100\%$ ($2,0$ vs. $1,0$).
* Gradiente relevante de risco demográfico e socioeconômico nas variáveis categóricas (`NAME_EDUCATION_TYPE`, `NAME_INCOME_TYPE` e `CODE_GENDER`).

### 4.4. Estratégia de MVP e Mapeamento nos 5 Cs do Crédito (12 Features)

Em sistemas produtivos de concessão de crédito, pipelines com excesso de variáveis elevam a latência de rede, fragilizam o contrato do *payload* da API e dificultam auditorias regulatórias.

Optou-se pela estratégia de **MVP (*Minimum Viable Product*)**: após a eliminação de variáveis redundantes via Matriz de Correlação de Pearson (como `AMT_GOODS_PRICE`, que apresentava $r = 0,99$ com `AMT_CREDIT`, e `BUREAU_TOTAL_LOANS`, com $r = 0,74$ frente a `BUREAU_ACTIVE_LOANS`), selecionaram-se **12 atributos de elevado poder discriminante**, estruturados segundo os pilares clássicos dos **5 Cs do Crédito**:

| Pilar dos 5 Cs do Crédito | Significado Bancário | Variáveis Selecionadas no Contrato do MVP |
| :--- | :--- | :--- |
| **1. Caráter (*Character*)** | Histórico de cumprimento de obrigações e reputação de mercado. | `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| **2. Capacidade (*Capacity*)** | Geração de renda, estabilidade profissional e comprometimento financeiro. | `AMT_INCOME_TOTAL`, `ANNUITY_TO_INCOME_RATIO`, `YEARS_EMPLOYED` |
| **3. Capital (*Capital*)** | Grau de endividamento vigente no mercado e alavancagem financeira. | `BUREAU_ACTIVE_LOANS`, `BUREAU_TOTAL_DEBT_SUM` |
| **4. Condições (*Conditions*)** | Condições da operação e enquadramento socioeconômico/demográfico do tomador. | `AMT_CREDIT`, `NAME_CONTRACT_TYPE`, `AGE_YEARS`, `CODE_GENDER`, `NAME_EDUCATION_TYPE` |
| **5. Colateral (*Collateral*)** | Garantias atreladas ao financiamento *(capturado indiretamente pelo valor e modalidade contratada)*. | *(Representado via `AMT_CREDIT` e modalidade `NAME_CONTRACT_TYPE`)* |

---

## 🤖 5. Modelagem Científica, Prevenção de Data Leakage e Seleção do Campeão

### 5.1. Arquitetura Anti-Leakage

A base de dados foi particionada de maneira estratificada em **Treino ($80\%$ — $246.008$ registros)** e **Teste Holdout ($20\%$ — $61.503$ registros)**. Todo o pré-processamento (`SimpleImputer`, `StandardScaler` e `OneHotEncoder(handle_unknown='ignore')`) foi encapsulado em um objeto unificado `Pipeline` + `ColumnTransformer` do Scikit-Learn, garantindo que estatísticas de imputação e escalonamento fossem ajustadas (`.fit()`) exclusivamente nos dados de treino.

### 5.2. Torneio de Modelos (5-Fold Stratified Cross-Validation)

Três famílias de algoritmos com níveis crescentes de complexidade foram avaliadas na base de Treino:

| Modelo Candidato | ROC-AUC (CV Mean $\pm$ Std) | PR-AUC (CV Mean) | Brier Score | Tempo de Fit |
| :--- | :---: | :---: | :---: | :---: |
| **1. Logistic Regression** *(Baseline Linear)* | $0,7263 \pm 0,0047$ | $0,2047$ | $0,2076$ | $0,68\text{ s}$ |
| **2. Random Forest** *(Bagging Paralelo)* | $0,7331 \pm 0,0044$ | $0,2111$ | $0,1924$ | $14,72\text{ s}$ |
| **3. CatBoost Classifier** *(Gradient Boosting)* | $0,7379 \pm 0,0048$ | $0,2174$ | $0,2037$ | $8,84\text{ s}$ |

O **CatBoost Classifier** (`iterations=400`, `depth=4`, `learning_rate=0.05`, `l2_leaf_reg=5.0`) alcançou desempenho superior em todas as métricas de discriminação de risco, capturando interações não-lineares entre os 5 Cs do Crédito com estabilidade entre os *folds* e reduzido *gap* de generalização entre treino e validação ($0,0048$).

---

## 🎯 6. Avaliação no Holdout, Política de Crédito e Rigor Estatístico

No conjunto de Teste isolado ($61.503$ propostas não observadas no treino), o pipeline campeão obteve os seguintes resultados:

* **ROC-AUC:** **`0.7353`**
* **PR-AUC (Average Precision):** **`0.2116`** ($2,62\times$ superior à taxa aleatória de $0,0807$)
* **Estatística KS (Kolmogorov-Smirnov):** **`0.3510`** ($35,10\%$ de separação máxima acumulada entre adimplentes e inadimplentes, patamar classificado como excelente no varejo financeiro).

### 6.1. Política Operacional de Crédito: Semáforo de 3 Faixas

Dado que o treinamento operou com balanceamento de classes (`Balanced`), as probabilidades calculadas funcionam como um **Score Ordinal de Risco** calibrado em torno de $0,50$. Em substituição a uma regra puramente binária, adotaram-se os dois limiares ótimos calculados sobre a curva de teste — o ponto de **Máximo KS / $F_2$-Score (`0.5027`)** e o de **Máximo $F_1$-Score (`0.6400`)** — para estruturar uma política de concessão em 3 faixas:

| Faixa de Score (`prob`) | Classificação de Risco | Ação Recomendada na API | Racional de Negócio e Impacto na Carteira |
| :---: | :---: | :---: | :--- |
| **`< 0.5027`** | 🟢 **BAIXO RISCO** | **Aprovação Automática** (*Fast-Track*) | Preserva $71,0\%$ dos clientes adimplentes e reduz a taxa de inadimplência da carteira aprovada de $8,07\%$ **para $4,19\%$** ($-48,1\%$ **de redução relativa de perdas**). |
| **`0.5027` a `0.6399`** | 🟡 **MÉDIO RISCO** | **Análise Manual / Limite Reduzido** (*Zona Cinzenta*) | Casos limítrofes passíveis de solicitação de garantias, documentação complementar ou redução de limite concedido. |
| **`>= 0.6400`** | 🔴 **ALTO RISCO** | **Reprovação Automática** (*Política Restritiva*) | Bloqueio do estrato superior de risco onde a concentração de inadimplência é crítica, prevenindo baixas a prejuízo (*write-offs*). |

### 6.2. Comprovação de Significância Estatística (Bootstrap e Teste Z)

Para mitigar o risco de conclusões atreladas a variações amostrais, executaram-se dois procedimentos de validação formal sobre o conjunto de Teste:

1. **Pareamento via Bootstrap ($500$ réplicas):** A superioridade do CatBoost (`0.7353`, IC 95%: `[0.7285, 0.7423]`) sobre a Regressão Logística (`0.7244`, IC 95%: `[0.7172, 0.7316]`) resultou em ganho médio de $\Delta\text{AUC} = +0,0109$ com $p\text{-value} < 0,0001$ (superior em $100\%$ das reamostragens).
2. **Teste A/B de Impacto de Carteira (Teste Z de Proporções):** A contração da taxa de inadimplência de $8,07\%$ para $4,19\%$ registrou estatística $Z = -21,34$ ($p\text{-value} < 0,0001$), evidenciando significância estatística na contenção de perdas financeiras.

### 6.3. Explicabilidade e Conformidade Regulatória (BACEN / LGPD)

A análise de importância de atributos corrobora a aderência do modelo aos **5 Cs do Crédito**: o pilar de **Caráter** (`EXT_SOURCE_2` e `EXT_SOURCE_3`) concentra $62,0\%$ da influência decisória, complementado pelas **Condições e Capacidade** do proponente (`AMT_CREDIT` com $9,3\%$, `AGE_YEARS` com $6,7\%$, escolaridade com $5,5\%$ e tempo de emprego `YEARS_EMPLOYED` com $4,7\%$).

---

## 📂 7. Estrutura Atual do Repositório

```text
├── data/
│   ├── raw/                        # Camada Bronze: CSVs originais (Home Credit) [Ignorado no Git]
│   └── processed/                  # Camada Gold: abt_train.parquet [Ignorado no Git]
├── models/
│   ├── credit_risk_pipeline.joblib # Pipeline completo serializado (Pré-processamento + CatBoost - 120 KB)
│   └── model_metrics.json          # Metadados de auditoria, métricas de teste e limiares (0.5027 / 0.6400)
├── notebooks/
│   ├── 01_eda_and_abt_validation.ipynb        # Validação da ABT e sanidade da camada Gold
│   └── 02_model_training_and_evaluation.ipynb # EDA profunda, MVP 5 Cs, modelagem e testes estatísticos
├── outputs/                        # Gráficos de alta resolução gerados na Fase 2/3 (300 DPI)
├── sql/
│   └── create_abt.sql              # Query analítica DuckDB (Camada Silver -> Gold)
├── src/
│   └── data_prep.py                # Script de orquestração do pipeline de dados
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação técnica e executiva (CRISP-DM)
```

---

## 🚀 8. Instruções para Reprodução Local

1. **Clonagem do repositório e criação do ambiente virtual:**
   ```bash
   git clone <URL_DO_SEU_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS (ou .venv\Scripts\activate no Windows)
   pip install -r requirements.txt
   ```

2. **Execução do Pipeline de Engenharia de Dados (Camada Gold):**
   Com os arquivos `application_train.csv` e `bureau.csv` posicionados em `data/raw/`, executar:
   ```bash
   python src/data_prep.py
   ```

3. **Inspeção do Treinamento e Testes Estatísticos:**
   Abrir o notebook `notebooks/02_model_training_and_evaluation.ipynb` para verificação das tabelas descritivas, gráficos diagnósticos e exportação dos artefatos em `models/`.
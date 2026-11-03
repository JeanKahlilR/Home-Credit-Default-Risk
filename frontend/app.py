"""Mesa de Crédito | Home Credit Risk — Front-End Streamlit.

Consome a API FastAPI (/health, /metadata, /predict, /predict/batch).
"""
import io
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Mesa de Crédito | Home Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Constantes de negócio
# ----------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_APPROVE = 0.5027
DEFAULT_REJECT = 0.64

EDUCACAO_PT_EN = {
    "Ensino Fundamental": "Lower secondary",
    "Ensino Médio / Técnico": "Secondary / secondary special",
    "Superior Incompleto": "Incomplete higher",
    "Ensino Superior": "Higher education",
    "Pós-graduação / Acadêmico": "Academic degree",
}
RENDA_PT_EN = {
    "Assalariado (CLT/Formal)": "Working",
    "Empresário / Autônomo": "Commercial associate",
    "Empreendedor / Dono de Negócio": "Businessman",
    "Servidor Público": "State servant",
    "Aposentado / Pensionista": "Pensioner",
    "Estudante": "Student",
    "Desempregado": "Unemployed",
    "Licença Maternidade": "Maternity leave",
}
EDUCACAO_EN_PT = {v: k for k, v in EDUCACAO_PT_EN.items()}
RENDA_EN_PT = {v: k for k, v in RENDA_PT_EN.items()}

FAIXAS = {
    "APROVADO_AUTOMATICO": {
        "emoji": "🟢",
        "titulo": "APROVADO AUTOMÁTICO",
        "curto": "Aprovado",
        "bg": "#e6f4ea",
        "borda": "#1e8e3e",
        "texto": "#0d5c22",
        "parecer": (
            "Proposta dentro do apetite de risco. **Recomendação:** seguir com a "
            "formalização do contrato nas condições solicitadas, mantendo o "
            "monitoramento padrão da carteira."
        ),
    },
    "ANALISE_ADICIONAL_OU_CONDICIONAL": {
        "emoji": "🟡",
        "titulo": "ANÁLISE ADICIONAL / CONDICIONAL",
        "curto": "Mesa Manual",
        "bg": "#fef7e0",
        "borda": "#f9ab00",
        "texto": "#7a5200",
        "parecer": (
            "Proposta na zona cinzenta. **Recomendação:** encaminhar ao comitê de "
            "crédito para análise manual; considerar comprovação adicional de renda, "
            "redução do valor/prazo, entrada maior ou garantia complementar."
        ),
    },
    "REPROVADO_AUTOMATICO": {
        "emoji": "🔴",
        "titulo": "REPROVADO AUTOMÁTICO",
        "curto": "Reprovado",
        "bg": "#fce8e6",
        "borda": "#d93025",
        "texto": "#8c1d18",
        "parecer": (
            "Risco de inadimplência acima do limite tolerado. **Recomendação:** "
            "recusar a proposta nas condições atuais; pode-se orientar o cliente "
            "a reapresentar com valor menor, comprovação de renda ou garantias."
        ),
    },
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "baixo": dict(
        idade=48, anos_emprego=15.0, sem_vinculo=False,
        escolaridade="Ensino Superior", vinculo="Servidor Público",
        renda=300000.0, credito=150000.0, dti=10.0,
        score2=850, score3=0.85, sem_birô=False,
        emprestimos=0, divida=0.0, carro="Sim",
    ),
    "moderado": dict(
        idade=34, anos_emprego=3.0, sem_vinculo=False,
        escolaridade="Ensino Médio / Técnico", vinculo="Assalariado (CLT/Formal)",
        renda=120000.0, credito=500000.0, dti=35.0,
        score2=500, score3=0.50, sem_birô=False,
        emprestimos=2, divida=60000.0, carro="Não",
    ),
    "alto": dict(
        idade=21, anos_emprego=0.2, sem_vinculo=False,
        escolaridade="Ensino Fundamental", vinculo="Assalariado (CLT/Formal)",
        renda=40000.0, credito=1200000.0, dti=90.0,
        score2=50, score3=0.05, sem_birô=False,
        emprestimos=8, divida=500000.0, carro="Não",
    ),
}

# Dicionário de dados (12 variáveis do contrato da API)
DICIONARIO = pd.DataFrame(
    [
        ("EXT_SOURCE_2", "Score de Crédito Externo 2", "Capital / Condições", "Pontuação normalizada de birô externo (0 a 1; na tela, 0 a 1000). Quanto maior, melhor o perfil.", "Numérico (float)"),
        ("EXT_SOURCE_3", "Score de Crédito Externo 3", "Capital / Condições", "Segunda pontuação de birô externo (0 a 1). Pode ser nula para clientes sem histórico.", "Numérico (float, opcional)"),
        ("AMT_CREDIT", "Valor do Crédito Solicitado", "Capacidade", "Montante do crédito pleiteado, em R$.", "Numérico (float)"),
        ("YEARS_EMPLOYED", "Tempo de Emprego", "Caráter", "Anos no vínculo atual. Nulo para aposentados / sem vínculo.", "Numérico (float, opcional)"),
        ("AGE_YEARS", "Idade", "Caráter", "Idade do proponente, em anos (18 a 120).", "Numérico (float)"),
        ("DTI_RATIO", "Comprometimento da Renda Mensal — DTI (%)", "Capacidade", "Percentual da renda mensal do cliente comprometido com a parcela do financiamento (ex.: 30% significa que a parcela consome 30% do ganho mensal). Enviado à API como fração (0,30). Opcional.", "Numérico (float, opcional)"),
        ("AMT_INCOME_TOTAL", "Renda Total Anual", "Capacidade", "Renda anual total declarada, em R$.", "Numérico (float)"),
        ("BUREAU_TOTAL_CREDIT_DEBT", "Dívida Total Ativa no Birô", "Capital", "Soma da dívida em aberto em outras instituições, em R$.", "Numérico (float)"),
        ("BUREAU_ACTIVE_LOANS", "Empréstimos Ativos no Birô", "Condições / Histórico", "Quantidade de operações de crédito ativas no birô.", "Inteiro"),
        ("FLAG_OWN_CAR", "Possui Veículo Próprio", "Colateral", "1 = possui veículo; 0 = não possui.", "Inteiro (0/1)"),
        ("NAME_EDUCATION_TYPE", "Escolaridade", "Caráter", "Nível de escolaridade do proponente.", "Categórico"),
        ("NAME_INCOME_TYPE", "Tipo de Vínculo / Renda", "Caráter / Capacidade", "Natureza da fonte de renda do proponente.", "Categórico"),
    ],
    columns=["Variável (API)", "Nome de Negócio", "Pilar dos 5 Cs", "Descrição", "Tipo de Dado"],
)

# Mapeamento de cabeçalhos amigáveis do CSV -> campos da API
CSV_COLUNAS = {
    "Idade (anos)": "AGE_YEARS",
    "Tempo de Emprego (anos)": "YEARS_EMPLOYED",
    "Escolaridade": "NAME_EDUCATION_TYPE",
    "Tipo de Vinculo/Renda": "NAME_INCOME_TYPE",
    "Renda Total Anual (R$)": "AMT_INCOME_TOTAL",
    "Valor do Credito (R$)": "AMT_CREDIT",
    "Comprometimento da Renda Mensal - DTI (%)": "DTI_RATIO",
    "Score Externo 2 (0-1000)": "EXT_SOURCE_2",
    "Score Externo 3 (0-1000)": "EXT_SOURCE_3",
    "Emprestimos Ativos no Biro": "BUREAU_ACTIVE_LOANS",
    "Divida Total Ativa (R$)": "BUREAU_TOTAL_CREDIT_DEBT",
    "Possui Veiculo (Sim/Nao)": "FLAG_OWN_CAR",
}
CSV_OPCIONAIS = {"Tempo de Emprego (anos)", "Comprometimento da Renda Mensal - DTI (%)", "Score Externo 3 (0-1000)"}

MODELO_CSV = pd.DataFrame(
    [
        [48, 15, "Servidor Público", "Ensino Superior", 300000, 150000, 10, 850, 850, 0, 0, "Sim"],
        [34, 3, "Assalariado (CLT/Formal)", "Ensino Médio / Técnico", 120000, 500000, 35, 500, 500, 2, 60000, "Não"],
        [21, 0.2, "Assalariado (CLT/Formal)", "Ensino Fundamental", 40000, 1200000, 90, 50, 50, 8, 500000, "Não"],
        [67, "", "Aposentado / Pensionista", "Ensino Médio / Técnico", 60000, 80000, 15, 600, "", 1, 5000, "Não"],
    ],
    columns=[
        "Idade (anos)", "Tempo de Emprego (anos)", "Tipo de Vinculo/Renda", "Escolaridade",
        "Renda Total Anual (R$)", "Valor do Credito (R$)", "Comprometimento da Renda Mensal - DTI (%)",
        "Score Externo 2 (0-1000)", "Score Externo 3 (0-1000)",
        "Emprestimos Ativos no Biro", "Divida Total Ativa (R$)", "Possui Veiculo (Sim/Nao)",
    ],
)[list(CSV_COLUNAS.keys())]


# ----------------------------------------------------------------------------
# Estilo institucional
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1400px;}
    h1, h2, h3 {color: #14315c; font-weight: 600;}
    .app-header {border-bottom: 3px solid #14315c; padding-bottom: .4rem; margin-bottom: 1rem;}
    .app-header h1 {margin: 0; font-size: 1.8rem;}
    .app-header span {color: #5f6b7a; font-size: .95rem;}
    .badge {display:inline-block; padding: .25rem .8rem; border-radius: 999px;
            color:#fff; font-weight:600; font-size:.85rem;}
    .badge-on {background:#1e8e3e;} .badge-off {background:#d93025;}
    .risk-card {border-left: 10px solid; border-radius: 8px; padding: 1.1rem 1.4rem; margin: .6rem 0;}
    .risk-card h2 {margin: 0; font-size: 1.6rem; color: inherit !important;}
    .risk-card p {margin: .3rem 0 0 0; font-size: .95rem; color: inherit !important;}
    div[data-testid="stMetric"] {background:#f5f7fa; border:1px solid #e1e6ed;
            border-radius: 8px; padding: .7rem 1rem;}
    div[data-testid="stMetric"] *,
    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {color: #14315c !important;}
    .policy-box {background:#fff4e5; border:1px solid #f9ab00; border-left: 8px solid #e37400;
            border-radius: 8px; padding: .9rem 1.2rem; margin: .6rem 0; color:#5c3b00;}
    .policy-box h4 {margin:0 0 .4rem 0; color:#5c3b00 !important;}
    .policy-box ul {margin:.3rem 0 0 1.2rem; padding:0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Cliente da API
# ----------------------------------------------------------------------------
def api_get(base: str, path: str, timeout: float = 4.0) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{base.rstrip('/')}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_post(base: str, path: str, payload: Dict[str, Any], timeout: float = 60.0):
    """Retorna (json, erro)."""
    try:
        r = requests.post(f"{base.rstrip('/')}{path}", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return None, f"Falha de comunicação com a API: {exc}"
    if r.status_code == 200:
        return r.json(), None
    if r.status_code == 422:
        try:
            detalhes = "; ".join(
                f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in r.json()["detail"][:8]
            )
        except Exception:
            detalhes = r.text
        return None, f"Dados inválidos rejeitados pela API — {detalhes}"
    return None, f"A API respondeu com erro {r.status_code}: {r.text[:300]}"


@st.cache_data(ttl=300, show_spinner=False)
def get_metadata(base: str) -> Optional[Dict[str, Any]]:
    return api_get(base, "/metadata")


# ----------------------------------------------------------------------------
# Política Institucional de Crédito (Hard Rules)
# ----------------------------------------------------------------------------
SEM_VINCULO_EN = {"Unemployed", "Pensioner", "Student"}
SEM_VINCULO_PT = {k for k, v in RENDA_PT_EN.items() if v in SEM_VINCULO_EN}
GRAVIDADE = {"APROVADO_AUTOMATICO": 0, "ANALISE_ADICIONAL_OU_CONDICIONAL": 1, "REPROVADO_AUTOMATICO": 2}

MOTIVO_DESEMPREGADO = "Bloqueio de Política: Proponente desempregado sem fonte de renda ativa comprovada."
MOTIVO_ALAVANCAGEM = (
    "Bloqueio de Política (Capacidade): Valor solicitado excede o teto máximo de 10x a renda anual declarada."
)
MOTIVO_DTI_CRITICO = (
    "Bloqueio de Política (Capacidade): Comprometimento da renda mensal (DTI) superior ao limite prudencial de 50%."
)
MOTIVO_DTI_ALERTA = (
    "Alerta de Política (Capacidade): Comprometimento da renda mensal (DTI) entre 35% e 50% — "
    "aprovação automática não permitida, requer análise do comitê."
)


def aplicar_politica(payload: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica as travas de política sobre o resultado do modelo.

    A política só pode agravar a decisão (nunca converte reprovação em aprovação).
    Retorna a predição com `faixa_modelo`, `faixa_risco` (final), `travas` e `politica_acionada`.
    """
    faixa_modelo = pred["faixa_risco"]
    final = faixa_modelo
    travas: List[str] = []

    def escalar(alvo: str) -> None:
        nonlocal final
        if GRAVIDADE[alvo] > GRAVIDADE[final]:
            final = alvo

    if payload.get("NAME_INCOME_TYPE") == "Unemployed":
        travas.append(MOTIVO_DESEMPREGADO)
        escalar("REPROVADO_AUTOMATICO")

    renda, credito = payload.get("AMT_INCOME_TOTAL"), payload.get("AMT_CREDIT")
    if renda is not None and credito is not None and credito > 10 * renda:
        travas.append(MOTIVO_ALAVANCAGEM)
        escalar("REPROVADO_AUTOMATICO")

    dti = payload.get("DTI_RATIO")
    if dti is not None:
        if dti > 0.50:
            travas.append(MOTIVO_DTI_CRITICO)
            escalar("REPROVADO_AUTOMATICO")
        elif dti > 0.35 and faixa_modelo == "APROVADO_AUTOMATICO":
            travas.append(MOTIVO_DTI_ALERTA)
            escalar("ANALISE_ADICIONAL_OU_CONDICIONAL")

    out = dict(pred)
    out["faixa_modelo"] = faixa_modelo
    out["faixa_risco"] = final
    out["travas"] = travas
    out["politica_acionada"] = bool(travas)
    out["politica_sobrepos"] = final != faixa_modelo
    return out


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 Mesa de Crédito")
    st.caption("Home Credit Risk · Motor de decisão de crédito")
    api_url = st.text_input(
        "URL da API",
        value=DEFAULT_API_URL,
        help="Endereço base da API FastAPI. Altere apenas se o serviço estiver em outro host/porta.",
    ).strip()
    health = api_get(api_url, "/health")
    online = bool(health and health.get("model_loaded"))
    if online:
        st.markdown('<span class="badge badge-on">● Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-off">● Offline</span>', unsafe_allow_html=True)
        st.warning("API indisponível ou modelo não carregado. Verifique a URL e o serviço.")
    if st.button("🔄 Verificar conexão", width="stretch"):
        get_metadata.clear()
        st.rerun()
    st.divider()
    metadata = get_metadata(api_url) if online else None
    bands = (metadata or {}).get("decision_thresholds", {}).get("traffic_light_bands", {})
    LIM_APROV = float(bands.get("green_auto_approve_below", DEFAULT_APPROVE))
    LIM_REPROV = float(bands.get("red_auto_reject_above_or_equal", DEFAULT_REJECT))
    st.markdown("**Política de risco vigente**")
    st.markdown(
        f"🟢 Score < {LIM_APROV:.4f}  \n"
        f"🟡 {LIM_APROV:.4f} ≤ Score < {LIM_REPROV:.4f}  \n"
        f"🔴 Score ≥ {LIM_REPROV:.4f}"
    )
    if metadata:
        st.caption(f"Modelo v{metadata.get('model_version', '?')}")

st.markdown(
    '<div class="app-header"><h1>Mesa de Crédito</h1>'
    "<span>Simulação e concessão de crédito baseada em risco de inadimplência</span></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    [
        "👤 Análise de Proposta Individual",
        "📊 Análise de Lotes (Batch via CSV)",
        "📖 Dicionário de Dados & Política de Risco",
    ]
)


# ----------------------------------------------------------------------------
# Helpers de apresentação
# ----------------------------------------------------------------------------
def render_risk_card(pred: Dict[str, Any]) -> None:
    f = FAIXAS[pred["faixa_risco"]]
    st.markdown(
        f"""<div class="risk-card" style="background:{f['bg']};border-color:{f['borda']};color:{f['texto']}">
        <h2>{f['emoji']} {f['titulo']}</h2></div>""",
        unsafe_allow_html=True,
    )


def render_politica(pred: Dict[str, Any]) -> None:
    """Aviso de Travas de Política: score puro do modelo x política institucional."""
    if not pred.get("politica_acionada"):
        return
    fm = FAIXAS[pred["faixa_modelo"]]
    ff = FAIXAS[pred["faixa_risco"]]
    itens = "".join(f"<li>{t}</li>" for t in pred["travas"])
    if pred.get("politica_sobrepos"):
        expl = (
            f"A <b>Política Institucional de Crédito</b> sobrepôs o score estatístico: o modelo indicava "
            f"{fm['emoji']} <b>{fm['titulo']}</b>, mas a decisão final é {ff['emoji']} <b>{ff['titulo']}</b>. "
            "Travas de política refletem limites prudenciais do banco e prevalecem sobre a probabilidade "
            "calculada pelo modelo, que não é capaz de capturar essas restrições isoladamente."
        )
    else:
        expl = (
            "As travas abaixo foram acionadas, mas a decisão do modelo já era igual ou mais restritiva; "
            "a classificação final foi mantida."
        )
    st.markdown(
        f"""<div class="policy-box"><h4>⚠️ Travas de Política Acionadas</h4>
        <p><b>Score Estatístico Puro do Modelo:</b> {pred['probabilidade_default']:.4f}
        ({pred['probabilidade_default']:.2%}) — {fm['emoji']} {fm['titulo']}</p>
        <p>{expl}</p><b>Travas acionadas:</b><ul>{itens}</ul></div>""",
        unsafe_allow_html=True,
    )


def brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ----------------------------------------------------------------------------
# Aba 1 — Individual
# ----------------------------------------------------------------------------
def aplicar_preset(nome: str) -> None:
    for k, v in PRESETS[nome].items():
        st.session_state[f"f_{k}"] = v


def sincronizar_vinculo() -> None:
    """Sem tempo de emprego para Desempregado / Aposentado / Estudante."""
    st.session_state["f_sem_vinculo"] = st.session_state["f_vinculo"] in SEM_VINCULO_PT


# Estado inicial (preset moderado como padrão neutro)
for _k, _v in PRESETS["moderado"].items():
    st.session_state.setdefault(f"f_{_k}", _v)

with tab1:
    st.subheader("Simulação rápida")
    p1, p2, p3 = st.columns(3)
    p1.button("🟢 Exemplo: Baixo Risco (Prime)", on_click=aplicar_preset, args=("baixo",), width="stretch")
    p2.button("🟡 Exemplo: Risco Moderado (Zona Cinzenta)", on_click=aplicar_preset, args=("moderado",), width="stretch")
    p3.button("🔴 Exemplo: Alto Risco (Alavancado)", on_click=aplicar_preset, args=("alto",), width="stretch")

    st.subheader("Dados da proposta — 5 Cs do Crédito")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            st.markdown("**👤 Perfil & Estabilidade** *(Caráter)*")
            st.number_input("Idade (anos)", 18, 100, key="f_idade", help="Idade do proponente na data da análise.")
            sem_vinculo = st.checkbox(
                "Aposentado / Sem vínculo", key="f_sem_vinculo",
                disabled=st.session_state.f_vinculo in SEM_VINCULO_PT,
                help="Marque se não há tempo de emprego a informar. Será enviado como valor nulo. "
                     "Marcado automaticamente para Desempregado, Aposentado / Pensionista e Estudante.",
            )
            st.number_input(
                "Tempo de Emprego (anos)", 0.0, 60.0, step=0.5, key="f_anos_emprego",
                disabled=sem_vinculo, help="Tempo no vínculo empregatício atual.",
            )
            st.selectbox("Escolaridade", list(EDUCACAO_PT_EN), key="f_escolaridade",
                         help="Maior nível de escolaridade concluído.")
            st.selectbox("Tipo de Vínculo / Renda", list(RENDA_PT_EN), key="f_vinculo",
                         on_change=sincronizar_vinculo,
                         help="Natureza da principal fonte de renda. Desempregado, Aposentado / Pensionista "
                              "e Estudante não possuem tempo de emprego (enviado como nulo).")
    with c2:
        with st.container(border=True):
            st.markdown("**💰 Capacidade Financeira** *(Capacidade)*")
            st.number_input("Renda Total Anual (R$)", 1.0, 100_000_000.0, step=5000.0, key="f_renda",
                            help="Renda bruta anual total declarada.")
            st.number_input("Valor do Crédito Solicitado (R$)", 1.0, 100_000_000.0, step=5000.0, key="f_credito",
                            help="Montante total pleiteado na proposta.")
            st.slider("Comprometimento da Renda Mensal — DTI (%)", 0.0, 150.0, step=1.0, key="f_dti",
                      help="Percentual da renda mensal do cliente comprometido com a parcela do "
                           "financiamento. Ex.: 30% significa que a parcela consome 30% do ganho mensal.")
    with c3:
        with st.container(border=True):
            st.markdown("**🏛️ Histórico de Mercado** *(Capital / Condições)*")
            st.number_input("Score de Crédito Externo 2 (0–1000)", 0, 1000, step=10, key="f_score2",
                            help="Pontuação do birô, de 0 a 1000. Convertida internamente para 0,0–1,0.")
            sem_birô = st.checkbox(
                "Cliente sem histórico no birô", key="f_sem_birô",
                help="Marque se não existe Score Externo 3. Será enviado como valor nulo.",
            )
            st.slider("Score Externo 3 (0,0–1,0)", 0.0, 1.0, step=0.01, key="f_score3", disabled=sem_birô,
                      help="Segunda pontuação de birô, já na escala de 0 a 1.")
            st.number_input("Empréstimos Ativos no Birô", 0, 100, key="f_emprestimos",
                            help="Quantidade de operações de crédito ativas em outras instituições.")
            st.number_input("Dívida Total Ativa (R$)", 0.0, 100_000_000.0, step=5000.0, key="f_divida",
                            help="Soma do saldo devedor ativo no birô.")
    with c4:
        with st.container(border=True):
            st.markdown("**🚗 Garantias** *(Colateral)*")
            st.radio("Possui Veículo Próprio?", ["Sim", "Não"], key="f_carro", horizontal=True,
                     help="Bem passível de oferta como garantia.")

    analisar = st.button("⚖️ Analisar Proposta", type="primary", width="stretch", disabled=not online)
    if not online:
        st.info("Conecte-se à API para habilitar a análise.")

    if analisar:
        s = st.session_state
        payload = {
            "EXT_SOURCE_2": round(s.f_score2 / 1000.0, 6),
            "EXT_SOURCE_3": None if s.f_sem_birô else float(s.f_score3),
            "AMT_CREDIT": float(s.f_credito),
            "YEARS_EMPLOYED": None if (s.f_sem_vinculo or s.f_vinculo in SEM_VINCULO_PT) else float(s.f_anos_emprego),
            "AGE_YEARS": float(s.f_idade),
            "DTI_RATIO": round(s.f_dti / 100.0, 6),
            "AMT_INCOME_TOTAL": float(s.f_renda),
            "BUREAU_TOTAL_CREDIT_DEBT": float(s.f_divida),
            "BUREAU_ACTIVE_LOANS": int(s.f_emprestimos),
            "FLAG_OWN_CAR": 1 if s.f_carro == "Sim" else 0,
            "NAME_EDUCATION_TYPE": EDUCACAO_PT_EN[s.f_escolaridade],
            "NAME_INCOME_TYPE": RENDA_PT_EN[s.f_vinculo],
        }
        with st.spinner("Consultando o motor de risco..."):
            pred, erro = api_post(api_url, "/predict", payload)
        if erro:
            st.error(erro)
        else:
            pred = aplicar_politica(payload, pred)
            st.divider()
            st.subheader("Painel de Decisão")
            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilidade de Inadimplência", f"{pred['probabilidade_default']:.2%}",
                      help="Score estatístico puro do modelo: probabilidade estimada de default.")
            m2.metric("Score do Modelo", f"{pred['probabilidade_default']:.4f}")
            m3.metric("Limiar Oficial Aplicado", f"{pred['threshold_utilizado']:.4f}",
                      help=f"Reprovação automática a partir de {LIM_REPROV:.4f}.")
            render_risk_card(pred)
            render_politica(pred)
            st.markdown("#### 📝 Parecer de Negócio")
            st.markdown(FAIXAS[pred["faixa_risco"]]["parecer"])
            with st.expander("Ver payload enviado à API"):
                st.json(payload)


# ----------------------------------------------------------------------------
# Aba 2 — Lote
# ----------------------------------------------------------------------------
def _num(v: Any, escala: float = 1.0) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    if isinstance(v, str):
        v = v.strip().replace("R$", "").replace(" ", "")
        if "," in v:
            v = v.replace(".", "").replace(",", ".")
    return float(v) / escala


def csv_para_payloads(df: pd.DataFrame) -> List[Dict[str, Any]]:
    faltantes = [c for c in CSV_COLUNAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes no arquivo: {', '.join(faltantes)}")
    payloads: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        linha = i + 2  # considera o cabeçalho
        try:
            edu = str(row["Escolaridade"]).strip()
            vin = str(row["Tipo de Vinculo/Renda"]).strip()
            if edu not in EDUCACAO_PT_EN and edu not in EDUCACAO_EN_PT:
                raise ValueError(f"Escolaridade desconhecida '{edu}'")
            if vin not in RENDA_PT_EN and vin not in RENDA_EN_PT:
                raise ValueError(f"Tipo de Vínculo/Renda desconhecido '{vin}'")
            carro = str(row["Possui Veiculo (Sim/Nao)"]).strip().lower()
            if carro not in ("sim", "não", "nao", "1", "0", "1.0", "0.0"):
                raise ValueError(f"Valor inválido para veículo '{carro}'")
            payloads.append(
                {
                    "EXT_SOURCE_2": _num(row["Score Externo 2 (0-1000)"], 1000.0),
                    "EXT_SOURCE_3": _num(row["Score Externo 3 (0-1000)"], 1000.0),
                    "AMT_CREDIT": _num(row["Valor do Credito (R$)"]),
                    "YEARS_EMPLOYED": None if RENDA_PT_EN.get(vin, vin) in SEM_VINCULO_EN
                    else _num(row["Tempo de Emprego (anos)"]),
                    "AGE_YEARS": _num(row["Idade (anos)"]),
                    "DTI_RATIO": _num(row["Comprometimento da Renda Mensal - DTI (%)"], 100.0),
                    "AMT_INCOME_TOTAL": _num(row["Renda Total Anual (R$)"]),
                    "BUREAU_TOTAL_CREDIT_DEBT": _num(row["Divida Total Ativa (R$)"]),
                    "BUREAU_ACTIVE_LOANS": int(_num(row["Emprestimos Ativos no Biro"])),
                    "FLAG_OWN_CAR": 1 if carro in ("sim", "1", "1.0") else 0,
                    "NAME_EDUCATION_TYPE": EDUCACAO_PT_EN.get(edu, edu),
                    "NAME_INCOME_TYPE": RENDA_PT_EN.get(vin, vin),
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Linha {linha}: {exc}") from exc
    return payloads


def colorir_status(valor: str) -> str:
    cores = {
        "Aprovado": "background-color:#e6f4ea;color:#0d5c22;font-weight:600",
        "Mesa Manual": "background-color:#fef7e0;color:#7a5200;font-weight:600",
        "Reprovado": "background-color:#fce8e6;color:#8c1d18;font-weight:600",
    }
    return cores.get(valor, "")


with tab2:
    st.subheader("Processamento em lote")
    st.markdown(
        "1. Baixe a planilha modelo · 2. Preencha uma proposta por linha · "
        "3. Envie o arquivo `.csv` para avaliação."
    )
    st.download_button(
        "⬇️ Baixar Planilha Modelo de Propostas (CSV)",
        data=MODELO_CSV.to_csv(index=False).encode("utf-8-sig"),
        file_name="modelo_propostas.csv",
        mime="text/csv",
    )
    st.caption(
        "Campos opcionais (podem ficar em branco): Tempo de Emprego, Comprometimento da Renda Mensal (DTI) e Score Externo 3. "
        "Scores externos na escala 0–1000; DTI em % da renda mensal comprometida com a parcela (ex.: 30 = 30%). "
        "Para Desempregado, Aposentado / Pensionista e Estudante, o Tempo de Emprego é ignorado (enviado como nulo). "
        "Propostas passam pela Política Institucional de Crédito (travas de desemprego, alavancagem acima de 10x a renda e DTI)."
    )
    arquivo = st.file_uploader("Enviar planilha de propostas (.csv)", type=["csv"])

    if arquivo is not None:
        try:
            raw = arquivo.getvalue()
            df_in = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
            df_in.columns = [str(c).strip() for c in df_in.columns]
            payloads = csv_para_payloads(df_in)
        except Exception as exc:
            st.error(f"Não foi possível ler o arquivo: {exc}")
            payloads = []
        if payloads:
            st.success(f"{len(payloads)} proposta(s) lida(s) com sucesso.")
            with st.expander("Pré-visualizar arquivo enviado"):
                st.dataframe(df_in, width="stretch")
            if st.button("🚀 Processar Lote", type="primary", disabled=not online):
                with st.spinner("Avaliando propostas..."):
                    resp, erro = api_post(api_url, "/predict/batch", {"applications": payloads})
                if erro:
                    st.error(erro)
                else:
                    st.session_state["lote_resultado"] = (df_in.copy(), resp, payloads)
            if not online:
                st.info("Conecte-se à API para processar o lote.")

    if "lote_resultado" in st.session_state:
        df_in, resp, payloads_lote = st.session_state["lote_resultado"]
        finais = [aplicar_politica(pl, pr) for pl, pr in zip(payloads_lote, resp["predictions"])]
        preds = pd.DataFrame(finais)
        total = len(finais)
        cont = {k: int((preds["faixa_risco"] == k).sum()) for k in FAIXAS}
        n_politica = int(preds["politica_sobrepos"].sum())
        st.divider()
        st.subheader("Resultados do lote")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Processado", total)
        k2.metric("🟢 % Aprovado", f"{cont['APROVADO_AUTOMATICO'] / total:.1%}")
        k3.metric("🟡 % Em Mesa Manual", f"{cont['ANALISE_ADICIONAL_OU_CONDICIONAL'] / total:.1%}")
        k4.metric("🔴 % Reprovado", f"{cont['REPROVADO_AUTOMATICO'] / total:.1%}")
        if n_politica:
            st.warning(
                f"⚠️ {n_politica} proposta(s) tiveram a decisão do modelo sobreposta pela Política "
                "Institucional de Crédito (Hard Rules). Veja as colunas **Status do Modelo** e **Travas de Política**."
            )

        resultado = df_in.reset_index(drop=True).copy()
        resultado.insert(0, "Nº", range(1, len(resultado) + 1))
        resultado["Prob. Inadimplência"] = preds["probabilidade_default"].round(4)
        resultado["Status do Modelo"] = preds["faixa_modelo"].map(lambda x: FAIXAS[x]["curto"])
        resultado["Status"] = preds["faixa_risco"].map(lambda x: FAIXAS[x]["curto"])
        resultado["Travas de Política"] = preds["travas"].map(lambda t: " | ".join(t))
        resultado["Decisão Binária"] = (preds["faixa_risco"] != "APROVADO_AUTOMATICO").astype(int)
        resultado["Limiar"] = preds["threshold_utilizado"]

        st.dataframe(
            resultado.style.map(colorir_status, subset=["Status", "Status do Modelo"]).format({"Prob. Inadimplência": "{:.2%}"}),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "⬇️ Baixar Relatório Avaliado (CSV)",
            data=resultado.to_csv(index=False).encode("utf-8-sig"),
            file_name="relatorio_propostas_avaliadas.csv",
            mime="text/csv",
        )


# ----------------------------------------------------------------------------
# Aba 3 — Dicionário & Metadados
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Dicionário de Dados — 12 variáveis do modelo")
    st.dataframe(DICIONARIO, width="stretch", hide_index=True)

    st.subheader("Política de Risco — Semáforo")
    st.table(
        pd.DataFrame(
            {
                "Faixa": ["🟢 Aprovado Automático", "🟡 Análise Adicional / Condicional", "🔴 Reprovado Automático"],
                "Regra de Score": [
                    f"Score < {LIM_APROV:.4f}",
                    f"{LIM_APROV:.4f} ≤ Score < {LIM_REPROV:.4f}",
                    f"Score ≥ {LIM_REPROV:.4f}",
                ],
                "Ação Operacional": [
                    "Formalização direta",
                    "Encaminhar ao comitê de crédito",
                    "Recusa nas condições atuais",
                ],
            }
        )
    )

    st.subheader("Metadados Oficiais do Modelo")
    if not metadata:
        st.warning("Metadados indisponíveis — API offline.")
    else:
        teste = metadata.get("test_holdout_metrics", {})
        val = metadata.get("validation_metrics", {})
        ci = teste.get("roc_auc_ci95_bootstrap", [None, None])
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC (Teste)", f"{teste.get('roc_auc', 0):.4f}")
        b.metric("KS (Teste)", f"{teste.get('ks_statistic_pct', 0):.2f}%")
        c.metric("ROC-AUC (Validação)", f"{val.get('roc_auc', 0):.4f}")
        d.metric("Algoritmo", metadata.get("champion_algorithm", "—"))
        if ci and ci[0] is not None:
            st.caption(f"IC 95% (bootstrap) do ROC-AUC de teste: [{ci[0]:.4f}, {ci[1]:.4f}]")

        col_h, col_i = st.columns(2)
        with col_h:
            st.markdown("**Hiperparâmetros do modelo campeão**")
            st.dataframe(
                pd.DataFrame(
                    [(k, str(v)) for k, v in metadata.get("best_hyperparameters", {}).items()],
                    columns=["Hiperparâmetro", "Valor"],
                ),
                hide_index=True, width="stretch",
            )
        with col_i:
            st.markdown("**Importância das variáveis (%)**")
            imp = pd.Series(metadata.get("feature_importance_contract_pct", {})).sort_values()
            st.bar_chart(imp, horizontal=True)

        with st.expander("Metadados completos (JSON)"):
            st.json(metadata)

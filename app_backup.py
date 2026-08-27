import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import skfuzzy as fuzz

from skfuzzy import control as ctrl


# ============================================================
# PAGE CONFIGURATION
# ============================================================
# (UNCHANGED)

st.set_page_config(
    page_title="Financial Fraud Detection",
    layout="wide"
)


# ============================================================
# GLOBAL STYLE
# ============================================================
# FRONTEND ONLY: styling rewritten for a cleaner, more
# "human-designed" academic/industry prototype look.
# No backend logic lives in this block.

st.markdown("""
<style>

    /* ---------- Base ---------- */

    .stApp {
        background-color: #f6f7f9;
        color: #1f2430;
    }

    .main .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont,
                      Helvetica, Arial, sans-serif;
    }


    /* ---------- Header ---------- */

    .app-header {
        border-bottom: 1px solid #e2e5eb;
        padding-bottom: 16px;
        margin-bottom: 28px;
    }

    .app-title {
        font-size: 26px;
        font-weight: 700;
        color: #14213d;
        letter-spacing: -0.2px;
        margin: 0 0 4px 0;
    }

    .app-subtitle {
        font-size: 14.5px;
        color: #667085;
        margin: 0;
    }


    /* ---------- Section headings ---------- */

    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #14213d;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 6px;
        margin-bottom: 14px;
        padding-bottom: 6px;
        border-bottom: 2px solid #14213d;
        display: inline-block;
    }

    .section-wrap {
        margin-bottom: 30px;
    }


    /* ---------- Dataset info panel (replaces st.info) ---------- */

    .dataset-panel {
        background-color: #ffffff;
        border: 1px solid #e2e5eb;
        border-radius: 4px;
        padding: 14px 18px;
        height: 100%;
    }

    .dataset-panel .d-row {
        display: flex;
        justify-content: space-between;
        font-size: 13.5px;
        padding: 3px 0;
        color: #475069;
    }

    .dataset-panel .d-row span:last-child {
        font-weight: 600;
        color: #14213d;
    }


    /* ---------- Transaction summary (plain typography) ---------- */

    .summary-item {
        padding: 2px 0 10px 0;
    }

    .summary-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8a93a6;
        margin-bottom: 2px;
    }

    .summary-value {
        font-size: 20px;
        font-weight: 600;
        color: #14213d;
    }

    .summary-value.tag-legit {
        color: #1b7a3d;
    }

    .summary-value.tag-fraud {
        color: #b3261e;
    }


    /* ---------- Metric / result cards ---------- */

    .result-card {
        background-color: #ffffff;
        border: 1px solid #e2e5eb;
        border-radius: 4px;
        padding: 16px 14px;
        text-align: left;
        min-height: 92px;
    }

    .result-label {
        color: #8a93a6;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 8px;
    }

    .result-value {
        color: #14213d;
        font-size: 24px;
        font-weight: 700;
        line-height: 1.1;
    }


    /* ---------- Classification banner ---------- */

    .status-banner {
        border-radius: 4px;
        padding: 16px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid transparent;
    }

    .status-banner .status-icon {
        font-size: 20px;
        font-weight: 700;
        line-height: 1;
    }

    .status-banner .status-text-title {
        font-weight: 700;
        font-size: 14.5px;
        letter-spacing: 0.3px;
        margin-bottom: 2px;
    }

    .status-banner .status-text-sub {
        font-size: 13.5px;
        opacity: 0.9;
    }

    .status-legit {
        background-color: #edf7f0;
        border-color: #c7e6d1;
        color: #1b7a3d;
    }

    .status-fraud {
        background-color: #fbecec;
        border-color: #f0c6c4;
        color: #b3261e;
    }


    /* ---------- Risk assessment panel ---------- */

    .risk-panel {
        border-radius: 4px;
        padding: 18px 20px;
        border: 1px solid transparent;
    }

    .risk-panel-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 10px;
    }

    .risk-level-tag {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 3px 10px;
        border-radius: 3px;
        color: #ffffff;
    }

    .risk-score-line {
        font-size: 13.5px;
        color: #475069;
        margin-bottom: 8px;
    }

    .risk-action-label {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        color: #8a93a6;
        margin-bottom: 2px;
    }

    .risk-action-text {
        font-size: 14px;
        color: #14213d;
    }

    .risk-high {
        background-color: #fbecec;
        border-color: #f0c6c4;
    }
    .risk-high .risk-level-tag { background-color: #b3261e; }

    .risk-medium {
        background-color: #fdf5e5;
        border-color: #f2dfae;
    }
    .risk-medium .risk-level-tag { background-color: #b06d02; }

    .risk-low {
        background-color: #edf7f0;
        border-color: #c7e6d1;
    }
    .risk-low .risk-level-tag { background-color: #1b7a3d; }


    /* ---------- Streamlit widget overrides ---------- */

    [data-testid="stNumberInput"] label {
        color: #14213d !important;
        font-weight: 600;
        font-size: 13.5px;
    }

    [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #14213d !important;
        border: 1px solid #ccd1db !important;
        border-radius: 4px !important;
    }

    .stButton > button {
        background-color: #14213d !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 11px 22px !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        letter-spacing: 0.2px;
        transition: background-color 0.15s ease-in-out;
    }

    .stButton > button p {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #1f3563 !important;
    }

    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e5eb !important;
        border-radius: 4px !important;
    }

    [data-testid="stExpander"] summary {
        color: #14213d !important;
        font-weight: 600;
        font-size: 13.5px;
    }

    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
    }

    .feature-note {
        font-size: 12.5px;
        color: #8a93a6;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    hr {
        border-color: #e2e5eb !important;
    }

    /* ---------- Footer ---------- */

    .app-footer {
        text-align: center;
        color: #98a1b3;
        font-size: 12px;
        padding-top: 18px;
        line-height: 1.6;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
# FRONTEND ONLY: compact header, no large hero section.

st.markdown(
    '<div class="app-header">'
    '<p class="app-title">AI Financial Fraud Detection</p>'
    '<p class="app-subtitle">Transaction Risk Assessment System '
    '&mdash; Machine Learning classification with fuzzy logic risk scoring</p>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================
# (UNCHANGED - BACKEND)

@st.cache_resource
def load_models():

    model = joblib.load(
        "models/optimized_xgboost.pkl"
    )

    model_config = joblib.load(
        "models/model_config.pkl"
    )

    anomaly_scaler = joblib.load(
        "models/anomaly_scaler.pkl"
    )

    legitimate_centroid = joblib.load(
        "models/legitimate_centroid.pkl"
    )

    legitimate_scaled = joblib.load(
        "models/legitimate_scaled.pkl"
    )

    return (
        model,
        model_config,
        anomaly_scaler,
        legitimate_centroid,
        legitimate_scaled
    )


(
    optimized_xgb,
    model_config,
    anomaly_scaler,
    legitimate_centroid,
    legitimate_scaled
) = load_models()


# ============================================================
# LOAD DATASET
# ============================================================
# (UNCHANGED - BACKEND)

@st.cache_data
def load_dataset():

    data = pd.read_csv(
        "creditcard.csv"
    )

    data["Log_Amount"] = np.log1p(
        data["Amount"]
    )

    data["Hour"] = (
        data["Time"] / 3600
    ) % 24

    return data


df = load_dataset()


# ============================================================
# FUZZY SYSTEM
# ============================================================
# (UNCHANGED - BACKEND)

fraud_prob = ctrl.Antecedent(
    np.arange(0, 101, 1),
    "fraud_probability"
)

amount_risk = ctrl.Antecedent(
    np.arange(0, 101, 1),
    "amount_risk"
)

anomaly_score = ctrl.Antecedent(
    np.arange(0, 101, 1),
    "anomaly_score"
)

risk = ctrl.Consequent(
    np.arange(0, 101, 1),
    "risk"
)


# ============================================================
# MEMBERSHIP FUNCTIONS
# ============================================================
# (UNCHANGED - BACKEND)

fraud_prob["low"] = fuzz.trapmf(
    fraud_prob.universe,
    [0, 0, 20, 40]
)

fraud_prob["medium"] = fuzz.trimf(
    fraud_prob.universe,
    [25, 50, 75]
)

fraud_prob["high"] = fuzz.trapmf(
    fraud_prob.universe,
    [60, 80, 100, 100]
)


amount_risk["low"] = fuzz.trapmf(
    amount_risk.universe,
    [0, 0, 20, 40]
)

amount_risk["medium"] = fuzz.trimf(
    amount_risk.universe,
    [25, 50, 75]
)

amount_risk["high"] = fuzz.trapmf(
    amount_risk.universe,
    [60, 80, 100, 100]
)


anomaly_score["low"] = fuzz.trapmf(
    anomaly_score.universe,
    [0, 0, 20, 40]
)

anomaly_score["medium"] = fuzz.trimf(
    anomaly_score.universe,
    [25, 50, 75]
)

anomaly_score["high"] = fuzz.trapmf(
    anomaly_score.universe,
    [60, 80, 100, 100]
)


risk["low"] = fuzz.trapmf(
    risk.universe,
    [0, 0, 20, 40]
)

risk["medium"] = fuzz.trimf(
    risk.universe,
    [30, 50, 70]
)

risk["high"] = fuzz.trapmf(
    risk.universe,
    [60, 80, 100, 100]
)


# ============================================================
# FUZZY RULES
# ============================================================
# (UNCHANGED - BACKEND)

rules = [

    ctrl.Rule(
        fraud_prob["low"] &
        amount_risk["low"] &
        anomaly_score["low"],
        risk["low"]
    ),

    ctrl.Rule(
        fraud_prob["low"] &
        amount_risk["medium"] &
        anomaly_score["low"],
        risk["low"]
    ),

    ctrl.Rule(
        fraud_prob["low"] &
        amount_risk["low"] &
        anomaly_score["medium"],
        risk["medium"]
    ),

    ctrl.Rule(
        fraud_prob["medium"] &
        amount_risk["low"] &
        anomaly_score["low"],
        risk["medium"]
    ),

    ctrl.Rule(
        fraud_prob["medium"] &
        amount_risk["medium"] &
        anomaly_score["low"],
        risk["medium"]
    ),

    ctrl.Rule(
        fraud_prob["low"] &
        amount_risk["high"] &
        anomaly_score["low"],
        risk["medium"]
    ),

    ctrl.Rule(
        fraud_prob["low"] &
        amount_risk["medium"] &
        anomaly_score["medium"],
        risk["medium"]
    ),

    ctrl.Rule(
        fraud_prob["medium"] &
        amount_risk["low"] &
        anomaly_score["medium"],
        risk["medium"]
    ),

    ctrl.Rule(
        fraud_prob["high"],
        risk["high"]
    ),

    ctrl.Rule(
        fraud_prob["medium"] &
        anomaly_score["high"],
        risk["high"]
    ),

    ctrl.Rule(
        fraud_prob["medium"] &
        amount_risk["high"],
        risk["high"]
    ),

    ctrl.Rule(
        amount_risk["high"] &
        anomaly_score["high"],
        risk["high"]
    ),

    ctrl.Rule(
        fraud_prob["low"] &
        amount_risk["high"] &
        anomaly_score["medium"],
        risk["high"]
    ),

    ctrl.Rule(
        fraud_prob["low"] &
        amount_risk["medium"] &
        anomaly_score["high"],
        risk["high"]
    ),

    ctrl.Rule(
        fraud_prob["low"] &
        amount_risk["high"] &
        anomaly_score["high"],
        risk["high"]
    )
]


risk_control = ctrl.ControlSystem(rules)


# ============================================================
# TRANSACTION ANALYSIS (BACKEND FUNCTION)
# ============================================================
# (UNCHANGED - BACKEND)
# Function name, logic, calculations and returned dictionary
# keys are all preserved exactly as in the original file.

def analyze_transaction(transaction):

    transaction = transaction.copy()

    feature_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Log_Amount", "Hour"]
    )

    X_transaction = pd.DataFrame(
        [transaction[feature_columns]]
    )

    # XGBoost fraud probability

    fraud_probability = optimized_xgb.predict_proba(
        X_transaction
    )[0, 1]

    threshold = model_config["threshold"]

    if fraud_probability >= threshold:
        ml_decision = "FRAUD"
    else:
        ml_decision = "LEGITIMATE"

    # Amount risk

    amount_risk_value = (
        (df["Amount"] <= transaction["Amount"]).mean()
        * 100
    )

    # Anomaly score

    transaction_v = pd.DataFrame(
        [[
            transaction[f"V{i}"]
            for i in range(1, 29)
        ]],
        columns=[
            f"V{i}"
            for i in range(1, 29)
        ]
    )

    transaction_scaled = anomaly_scaler.transform(
        transaction_v
    )

    anomaly_distance = np.sqrt(
        (
            transaction_scaled
            - legitimate_centroid
        ) ** 2
    ).sum() ** 0.5

    legitimate_distances = np.sqrt(
        (
            legitimate_scaled
            - legitimate_centroid
        ) ** 2
    ).sum(axis=1) ** 0.5

    anomaly_score_value = (
        np.mean(
            legitimate_distances <= anomaly_distance
        ) * 100
    )

    # Fuzzy risk assessment

    simulation = ctrl.ControlSystemSimulation(
        risk_control
    )

    simulation.input[
        "fraud_probability"
    ] = fraud_probability * 100

    simulation.input[
        "amount_risk"
    ] = amount_risk_value

    simulation.input[
        "anomaly_score"
    ] = anomaly_score_value

    try:

        simulation.compute()

        fuzzy_risk_score = (
            simulation.output["risk"]
        )

    except (KeyError, ValueError):

        fuzzy_risk_score = (
            0.60 * fraud_probability * 100
            + 0.20 * amount_risk_value
            + 0.20 * anomaly_score_value
        )

        fuzzy_risk_score = np.clip(
            fuzzy_risk_score,
            0,
            100
        )

    if fuzzy_risk_score < 30:
        risk_level = "LOW"

    elif fuzzy_risk_score < 70:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    return {
        "Fraud Probability": fraud_probability,
        "ML Decision": ml_decision,
        "Amount Risk": amount_risk_value,
        "Anomaly Score": anomaly_score_value,
        "Fuzzy Risk Score": fuzzy_risk_score,
        "Risk Level": risk_level
    }


# ============================================================
# TRANSACTION ANALYSIS SECTION (UI)
# ============================================================
# FRONTEND ONLY: same number_input, same min/max/value/step
# feeding the same transaction_index variable used below.

st.markdown(
    '<div class="section-wrap">'
    '<span class="section-title">Transaction Analysis</span>'
    '</div>',
    unsafe_allow_html=True
)

input_col, info_col = st.columns([1, 1.4])

with input_col:

    transaction_index = st.number_input(
        "Transaction Index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

with info_col:

    st.markdown(
        '<div class="dataset-panel">'
        '<div class="d-row"><span>Dataset</span>'
        f'<span>{len(df):,} Transactions</span></div>'
        '<div class="d-row"><span>Valid Index Range</span>'
        f'<span>0 &ndash; {len(df)-1:,}</span></div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SELECT TRANSACTION
# ============================================================
# (UNCHANGED - BACKEND / DATA ACCESS)

transaction = df.iloc[
    int(transaction_index)
].copy()


# ============================================================
# TRANSACTION SUMMARY (UI)
# ============================================================
# FRONTEND ONLY: replaced st.metric cards with lighter
# typography-based summary, as requested. "Dataset Label"
# simply reads the existing "Class" column already present
# in the loaded dataframe (0 = legitimate, 1 = fraud in the
# source data) - no new computation is introduced.

st.markdown(
    '<div class="section-wrap">'
    '<span class="section-title">Transaction Summary</span>'
    '</div>',
    unsafe_allow_html=True
)

summary1, summary2, summary3 = st.columns(3)

transaction_hour = (
    transaction["Time"] / 3600
) % 24

dataset_label = (
    "Fraud" if int(transaction["Class"]) == 1 else "Legitimate"
)

dataset_label_class = (
    "tag-fraud" if int(transaction["Class"]) == 1 else "tag-legit"
)

with summary1:

    st.markdown(
        '<div class="summary-item">'
        '<div class="summary-label">Transaction Amount</div>'
        f'<div class="summary-value">${transaction["Amount"]:,.2f}</div>'
        '</div>',
        unsafe_allow_html=True
    )

with summary2:

    st.markdown(
        '<div class="summary-item">'
        '<div class="summary-label">Transaction Hour</div>'
        f'<div class="summary-value">{transaction_hour:.2f}</div>'
        '</div>',
        unsafe_allow_html=True
    )

with summary3:

    st.markdown(
        '<div class="summary-item">'
        '<div class="summary-label">Dataset Classification</div>'
        f'<div class="summary-value {dataset_label_class}">{dataset_label}</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# ANALYZE BUTTON (UI)
# ============================================================
# FRONTEND ONLY: same st.button call/behavior, styled via CSS
# above (navy, subtle hover) instead of Streamlit's default red.

st.write("")

analyze = st.button(
    "Analyze Transaction",
    type="primary",
    use_container_width=True
)


if analyze:

    result = analyze_transaction(
        transaction
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ========================================================
    # DETECTION RESULTS (UI)
    # ========================================================
    # FRONTEND ONLY: same result dict values, restyled cards.

    st.markdown(
        '<div class="section-wrap">'
        '<span class="section-title">Detection Results</span>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">Fraud Probability</div>'
            f'<div class="result-value">'
            f'{result["Fraud Probability"] * 100:.2f}%'
            '</div></div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">Amount Risk</div>'
            f'<div class="result-value">'
            f'{result["Amount Risk"]:.2f}'
            '</div></div>',
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">Anomaly Score</div>'
            f'<div class="result-value">'
            f'{result["Anomaly Score"]:.2f}'
            '</div></div>',
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">Fuzzy Risk Score</div>'
            f'<div class="result-value">'
            f'{result["Fuzzy Risk Score"]:.2f}/100'
            '</div></div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # FRAUD CLASSIFICATION (UI)
    # ========================================================
    # FRONTEND ONLY: replaced st.error/st.success with a
    # custom, subtler status banner. Decision text is driven
    # entirely by result["ML Decision"], unchanged.

    st.markdown(
        '<div class="section-wrap" style="margin-top:28px;">'
        '<span class="section-title">Fraud Classification</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if result["ML Decision"] == "FRAUD":

        st.markdown(
            '<div class="status-banner status-fraud">'
            '<div class="status-icon">&#33;</div>'
            '<div>'
            '<div class="status-text-title">FRAUDULENT TRANSACTION</div>'
            '<div class="status-text-sub">Transaction classified as '
            'potentially fraudulent by the machine learning model.</div>'
            '</div></div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="status-banner status-legit">'
            '<div class="status-icon">&#10003;</div>'
            '<div>'
            '<div class="status-text-title">LEGITIMATE TRANSACTION</div>'
            '<div class="status-text-sub">Transaction classified as '
            'legitimate by the machine learning model.</div>'
            '</div></div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # FUZZY RISK ASSESSMENT (UI)
    # ========================================================
    # FRONTEND ONLY: same risk_level / fuzzy_risk_score values,
    # restyled panel with a clear level tag.

    st.markdown(
        '<div class="section-wrap" style="margin-top:28px;">'
        '<span class="section-title">Fuzzy Risk Assessment</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if result["Risk Level"] == "HIGH":

        st.markdown(
            f'<div class="risk-panel risk-high">'
            f'<div class="risk-panel-header">'
            f'<span class="risk-level-tag">HIGH</span>'
            f'</div>'
            f'<div class="risk-score-line">Fuzzy Risk Score: '
            f'{result["Fuzzy Risk Score"]:.2f} / 100</div>'
            f'<div class="risk-action-label">Recommended Action</div>'
            f'<div class="risk-action-text">Transaction should undergo '
            f'immediate review or additional verification.</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    elif result["Risk Level"] == "MEDIUM":

        st.markdown(
            f'<div class="risk-panel risk-medium">'
            f'<div class="risk-panel-header">'
            f'<span class="risk-level-tag">MEDIUM</span>'
            f'</div>'
            f'<div class="risk-score-line">Fuzzy Risk Score: '
            f'{result["Fuzzy Risk Score"]:.2f} / 100</div>'
            f'<div class="risk-action-label">Recommended Action</div>'
            f'<div class="risk-action-text">Additional verification is '
            f'recommended before processing.</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="risk-panel risk-low">'
            f'<div class="risk-panel-header">'
            f'<span class="risk-level-tag">LOW</span>'
            f'</div>'
            f'<div class="risk-score-line">Fuzzy Risk Score: '
            f'{result["Fuzzy Risk Score"]:.2f} / 100</div>'
            f'<div class="risk-action-label">Recommended Action</div>'
            f'<div class="risk-action-text">Transaction can proceed '
            f'under normal monitoring.</div>'
            f'</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # RISK INDICATOR CHART (UI)
    # ========================================================
    # FRONTEND ONLY: same four values plotted, restyled for the
    # light theme (colors/fonts only - no data transformation).

    st.markdown(
        '<div class="section-wrap" style="margin-top:28px;">'
        '<span class="section-title">Risk Indicators</span>'
        '</div>',
        unsafe_allow_html=True
    )

    indicators = pd.DataFrame({
        "Indicator": [
            "Fraud Probability",
            "Amount Risk",
            "Anomaly Score",
            "Fuzzy Risk Score"
        ],

        "Score": [
            result["Fraud Probability"] * 100,
            result["Amount Risk"],
            result["Anomaly Score"],
            result["Fuzzy Risk Score"]
        ]
    })

    fig = px.bar(
        indicators,
        x="Indicator",
        y="Score",
        range_y=[0, 100],
        text="Score"
    )

    fig.update_traces(
        marker_color="#14213d",
        marker_line_width=0,
        texttemplate="%{text:.2f}",
        textposition="outside",
        textfont=dict(color="#14213d", size=12),
        width=0.5
    )

    fig.update_layout(
        height=360,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(
            color="#475069",
            family="Segoe UI, Helvetica, Arial, sans-serif",
            size=12.5
        ),
        xaxis_title="",
        yaxis_title="Risk Score",
        yaxis=dict(
            range=[0, 100],
            gridcolor="#eef0f4",
            zerolinecolor="#e2e5eb"
        ),
        xaxis=dict(
            showgrid=False
        ),
        bargap=0.45,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # ========================================================
    # TECHNICAL FEATURES (UI)
    # ========================================================
    # FRONTEND ONLY: same underlying data (transaction row minus
    # Class/Log_Amount/Hour), presented in a compact 2-column
    # layout instead of a single tall dataframe.

    with st.expander(
        "View Technical Transaction Features"
    ):

        st.markdown(
            '<div class="feature-note">V1&ndash;V28 are anonymized '
            'PCA-transformed attributes provided by the dataset.</div>',
            unsafe_allow_html=True
        )

        technical_features = transaction.drop(
            ["Class", "Log_Amount", "Hour"]
        )

        feature_items = list(technical_features.items())
        midpoint = (len(feature_items) + 1) // 2
        left_items = feature_items[:midpoint]
        right_items = feature_items[midpoint:]

        left_df = pd.DataFrame(
            left_items, columns=["Feature", "Value"]
        ).set_index("Feature")

        right_df = pd.DataFrame(
            right_items, columns=["Feature", "Value"]
        ).set_index("Feature")

        feat_col1, feat_col2 = st.columns(2)

        with feat_col1:
            st.dataframe(
                left_df,
                use_container_width=True,
                height=420
            )

        with feat_col2:
            st.dataframe(
                right_df,
                use_container_width=True,
                height=420
            )


# ============================================================
# FOOTER
# ============================================================
# FRONTEND ONLY: subtle footer, no extra branding.

st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown(
    '<div class="app-footer">'
    'AI-Based Financial Fraud Detection and Risk Assessment<br/>'
    'Academic Project Prototype'
    '</div>',
    unsafe_allow_html=True
)
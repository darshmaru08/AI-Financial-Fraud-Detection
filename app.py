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

# Keep uploaded results available across Streamlit reruns.
# This prevents NameError before a CSV has been analyzed.
uploaded_results_df = st.session_state.get(
    "uploaded_results_df",
    None
)


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
# TRANSACTION ANALYSIS SECTION
# ============================================================

st.markdown(
    '<div class="section-wrap">'
    '<span class="section-title">Transaction Analysis</span>'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# ANALYSIS MODE
# ============================================================

analysis_tab1, analysis_tab2 = st.tabs(
    ["Existing Dataset", "Upload CSV"]
)

# ============================================================
# OPTION 1 — EXISTING DATASET
# ============================================================

with analysis_tab1:

    st.markdown("### Analyze Existing Transaction")

    input_col, info_col = st.columns([1, 1.4])

    with input_col:

        transaction_index = st.number_input(
            "Transaction Index",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
            key="existing_transaction_index"
        )

    with info_col:

        st.markdown(
            '<div class="dataset-panel">'
            '<div class="d-row">'
            '<span>Dataset</span>'
            f'<span>{len(df):,} Transactions</span>'
            '</div>'
            '<div class="d-row">'
            '<span>Valid Index Range</span>'
            f'<span>0 &ndash; {len(df)-1:,}</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # SELECT TRANSACTION
    # --------------------------------------------------------

    transaction = df.iloc[
        int(transaction_index)
    ].copy()

    # --------------------------------------------------------
    # TRANSACTION SUMMARY
    # --------------------------------------------------------

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
        "Fraud"
        if int(transaction["Class"]) == 1
        else "Legitimate"
    )

    dataset_label_class = (
        "tag-fraud"
        if int(transaction["Class"]) == 1
        else "tag-legit"
    )

    with summary1:

        st.markdown(
            '<div class="summary-item">'
            '<div class="summary-label">Transaction Amount</div>'
            f'<div class="summary-value">'
            f'${transaction["Amount"]:,.2f}'
            '</div></div>',
            unsafe_allow_html=True
        )

    with summary2:

        st.markdown(
            '<div class="summary-item">'
            '<div class="summary-label">Transaction Hour</div>'
            f'<div class="summary-value">'
            f'{transaction_hour:.2f}'
            '</div></div>',
            unsafe_allow_html=True
        )

    with summary3:

        st.markdown(
            '<div class="summary-item">'
            '<div class="summary-label">Dataset Classification</div>'
            f'<div class="summary-value {dataset_label_class}">'
            f'{dataset_label}'
            '</div></div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # ANALYZE EXISTING TRANSACTION
    # --------------------------------------------------------

    st.write("")

    analyze = st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
        key="analyze_existing"
    )

    if analyze:

        result = analyze_transaction(
            transaction
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ====================================================
        # DETECTION RESULTS
        # ====================================================

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
                f'{result["Fuzzy Risk Score"]:.2f} / 100'
                '</div></div>',
                unsafe_allow_html=True
            )

        # ====================================================
        # ML DECISION
        # ====================================================

        if result["ML Decision"] == "FRAUD":

            st.error(
                "⚠ FRAUDULENT TRANSACTION\n\n"
                "The machine learning model classified "
                "this transaction as potentially fraudulent."
            )

        else:

            st.success(
                "✓ LEGITIMATE TRANSACTION\n\n"
                "The machine learning model classified "
                "this transaction as legitimate."
            )

        # ====================================================
        # RISK LEVEL
        # ====================================================

        risk_level = result["Risk Level"]

        if risk_level == "HIGH":

            st.error(
                f"Risk Level: HIGH\n\n"
                f"Fuzzy Risk Score: "
                f'{result["Fuzzy Risk Score"]:.2f} / 100'
            )

        elif risk_level == "MEDIUM":

            st.warning(
                f"Risk Level: MEDIUM\n\n"
                f"Fuzzy Risk Score: "
                f'{result["Fuzzy Risk Score"]:.2f} / 100'
            )

        else:

            st.success(
                f"Risk Level: LOW\n\n"
                f"Fuzzy Risk Score: "
                f'{result["Fuzzy Risk Score"]:.2f} / 100'
            )


# ============================================================
# OPTION 2 — UPLOAD CSV
# ============================================================

with analysis_tab2:

    st.markdown("### Analyze Your Own CSV")

    st.markdown(
        """
        Upload a CSV containing transaction records.
        
        The uploaded file must contain the same transaction
        features used by the trained fraud detection model.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="fraud_csv_uploader"
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"CSV loaded successfully — "
                f"{len(uploaded_df):,} transactions found."
            )

            # ------------------------------------------------
            # REQUIRED FEATURES
            # ------------------------------------------------

            required_columns = [
                "Time",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
                "V9",
                "V10",
                "V11",
                "V12",
                "V13",
                "V14",
                "V15",
                "V16",
                "V17",
                "V18",
                "V19",
                "V20",
                "V21",
                "V22",
                "V23",
                "V24",
                "V25",
                "V26",
                "V27",
                "V28",
                "Amount"
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in uploaded_df.columns
            ]

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if missing_columns:

                st.error(
                    "The uploaded CSV is missing the "
                    "following required columns:"
                )

                st.code(
                    ", ".join(missing_columns)
                )

                st.info(
                    "Required columns are: "
                    + ", ".join(required_columns)
                )

            else:

                st.success(
                    "CSV format validated successfully."
                )
                uploaded_df["Log_Amount"] = np.log1p(
                    uploaded_df["Amount"]
                )
                uploaded_df["Hour"] = (
                    uploaded_df["Time"] / 3600
                ) % 24

                # --------------------------------------------
                # FILE INFORMATION
                # --------------------------------------------

                info1, info2, info3 = st.columns(3)

                with info1:

                    st.metric(
                        "Transactions",
                        f"{len(uploaded_df):,}"
                    )

                with info2:

                    st.metric(
                        "Features",
                        len(required_columns)
                    )

                with info3:

                    st.metric(
                        "File Type",
                        "CSV"
                    )

                # --------------------------------------------
                # PREVIEW
                # --------------------------------------------

                with st.expander(
                    "Preview Uploaded Data"
                ):

                    st.dataframe(
                        uploaded_df.head(10),
                        use_container_width=True
                    )

                st.write("")

                # --------------------------------------------
                # ANALYZE BUTTON
                # --------------------------------------------

                analyze_uploaded = st.button(
                    "Analyze Uploaded Transactions",
                    type="primary",
                    use_container_width=True,
                    key="analyze_uploaded"
                )

                if analyze_uploaded:

                    results = []

                    progress_bar = st.progress(0)

                    status_text = st.empty()

                    total_rows = len(uploaded_df)

                    for index, row in uploaded_df.iterrows():

                        try:

                            transaction_result = (
                                analyze_transaction(
                                    row.copy()
                                )
                            )

                            results.append({
                                "Transaction": index + 1,
                                "Fraud Probability":
                                    transaction_result[
                                        "Fraud Probability"
                                    ],
                                "ML Decision":
                                    transaction_result[
                                        "ML Decision"
                                    ],
                                "Amount Risk":
                                    transaction_result[
                                        "Amount Risk"
                                    ],
                                "Anomaly Score":
                                    transaction_result[
                                        "Anomaly Score"
                                    ],
                                "Fuzzy Risk Score":
                                    transaction_result[
                                        "Fuzzy Risk Score"
                                    ],
                                "Risk Level":
                                    transaction_result[
                                        "Risk Level"
                                    ]
                            })

                        except Exception as e:

                            results.append({
                                "Transaction": index + 1,
                                "Fraud Probability": None,
                                "ML Decision": "ERROR",
                                "Amount Risk": None,
                                "Anomaly Score": None,
                                "Fuzzy Risk Score": None,
                                "Risk Level": "ERROR"
                            })

                        progress_bar.progress(
                            (index + 1) / total_rows
                        )

                        status_text.text(
                            f"Analyzing transaction "
                            f"{index + 1:,} of "
                            f"{total_rows:,}"
                        )

                    status_text.empty()
                    progress_bar.empty()

                    # ----------------------------------------
                    # RESULTS DATAFRAME
                    # ----------------------------------------

                    uploaded_results_df = pd.DataFrame(
                        results
                    )

                    st.session_state[
                        "uploaded_results_df"
                    ] = uploaded_results_df

                    st.success(
                        "CSV analysis completed successfully."
                    )

                # --------------------------------------------
                # DISPLAY RESULTS
                # --------------------------------------------

                if (
                    "uploaded_results_df"
                    in st.session_state
                ):

                    uploaded_results_df = (
                        st.session_state[
                            "uploaded_results_df"
                        ]
                    )

                    st.markdown(
                        '<div class="section-wrap">'
                        '<span class="section-title">'
                        'Uploaded Transaction Results'
                        '</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    display_df = (
                        uploaded_results_df.copy()
                    )

                    display_df[
                        "Fraud Probability"
                    ] = (
                        display_df[
                            "Fraud Probability"
                        ] * 100
                    ).round(2)

                    display_df[
                        "Fraud Probability"
                    ] = (
                        display_df[
                            "Fraud Probability"
                        ].astype(str)
                        + "%"
                    )

                    display_df[
                        "Amount Risk"
                    ] = display_df[
                        "Amount Risk"
                    ].round(2)

                    display_df[
                        "Anomaly Score"
                    ] = display_df[
                        "Anomaly Score"
                    ].round(2)

                    display_df[
                        "Fuzzy Risk Score"
                    ] = display_df[
                        "Fuzzy Risk Score"
                    ].round(2)

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ----------------------------------------
                    # SUMMARY
                    # ----------------------------------------
                    # ----------------------------------------
                    # UPLOAD ANALYSIS SUMMARY
                    # ----------------------------------------

                    st.markdown(
                        '<div class="section-wrap">'
                        '<span class="section-title">'
                        'Upload Analysis Summary'
                        '</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    valid_results = uploaded_results_df[
                        uploaded_results_df["ML Decision"] != "ERROR"
                    ]

                    total_count = len(valid_results)

                    fraud_count = (
                        valid_results["ML Decision"] == "FRAUD"
                    ).sum()

                    legitimate_count = (
                        valid_results["ML Decision"] == "LEGITIMATE"
                    ).sum()

                    high_count = (
                        valid_results["Risk Level"] == "HIGH"
                    ).sum()

                    medium_count = (
                        valid_results["Risk Level"] == "MEDIUM"
                    ).sum()

                    low_count = (
                        valid_results["Risk Level"] == "LOW"
                    ).sum()

                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:
                        st.metric(
                            "Transactions Analyzed",
                            f"{total_count:,}"
                        )

                    with summary_col2:
                        st.metric(
                            "Fraud Detected",
                            f"{fraud_count:,}"
                        )

                    with summary_col3:
                        st.metric(
                            "Legitimate",
                            f"{legitimate_count:,}"
                        )

                    risk_col1, risk_col2, risk_col3 = st.columns(3)

                    with risk_col1:
                        st.metric(
                            "High Risk",
                            f"{high_count:,}"
                        )

                    with risk_col2:
                        st.metric(
                            "Medium Risk",
                            f"{medium_count:,}"
                        )

                    with risk_col3:
                        st.metric(
                            "Low Risk",
                            f"{low_count:,}"
                        )

                    # ----------------------------------------
                    # DOWNLOAD RESULTS
                    # ----------------------------------------

                    st.markdown(
                        '<div class="section-wrap">'
                        '<span class="section-title">'
                        'Export Results'
                        '</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    download_df = uploaded_results_df.copy()

                    download_df[
                        "Fraud Probability"
                    ] = (
                        download_df[
                            "Fraud Probability"
                        ] * 100
                    ).round(4)

                    download_df[
                        "Amount Risk"
                    ] = download_df[
                        "Amount Risk"
                    ].round(4)

                    download_df[
                        "Anomaly Score"
                    ] = download_df[
                        "Anomaly Score"
                    ].round(4)

                    download_df[
                        "Fuzzy Risk Score"
                    ] = download_df[
                        "Fuzzy Risk Score"
                    ].round(4)

                    csv_results = download_df.to_csv(
                        index=False
                    )

                    st.download_button(
                        label="Download Analysis Results",
                        data=csv_results,
                        file_name="fraud_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    st.caption(
                        "Download the complete fraud detection "
                        "and risk assessment results as a CSV file."
                    )

        except Exception as e:

            st.error(
                "Unable to read the uploaded CSV."
            )

            st.exception(e)

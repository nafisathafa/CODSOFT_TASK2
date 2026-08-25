import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FraudGuard | AI Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
# Design goal: calm, professional "fintech" look — one accent color,
# generous whitespace, quiet shadows, restrained typography.

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #0f172a;
    --ink-soft: #475569;
    --muted: #64748b;
    --border: #e5e9f0;
    --accent: #4f46e5;
    --accent-soft: #eef2ff;
    --danger: #dc2626;
    --danger-soft: #fef2f2;
    --success: #16a34a;
    --success-soft: #f0fdf4;
    --surface: #ffffff;
    --bg: #f7f8fb;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

/* App background */
html, body {
    background: var(--bg);
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stAppViewContainer"] > .main,
.stApp {
    background: var(--bg) !important;
    height: auto !important;
    min-height: 100vh;
    overflow: visible !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 880px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    color: var(--ink-soft) !important;
}
section[data-testid="stSidebar"] h3 {
    color: var(--ink) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 1.1rem 0 !important;
}
section[data-testid="stSidebar"] ol,
section[data-testid="stSidebar"] p {
    font-size: 13.5px !important;
    line-height: 1.55;
}

/* Hero header */
.hero {
    padding: 4px 4px 30px 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--accent);
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: var(--ink);
    margin: 0;
    letter-spacing: -0.6px;
}
.hero-subtitle {
    color: var(--muted);
    font-size: 15px;
    margin-top: 6px;
    font-weight: 400;
}

/* Meta row (replaces heavy metric pills) */
.meta-row {
    display: flex;
    gap: 28px;
    margin-top: 22px;
    flex-wrap: wrap;
}
.meta-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.meta-item .value {
    color: var(--ink);
    font-size: 14.5px;
    font-weight: 700;
}
.meta-item .label {
    color: var(--muted);
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin: 0 0 14px 2px;
}
.section-header .step {
    color: var(--accent);
    font-size: 12.5px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.section-header h3 {
    font-size: 16px;
    font-weight: 700;
    color: var(--ink);
    margin: 0;
}

/* Card container */
.card {
    background-color: var(--surface);
    padding: 26px 28px;
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    margin-bottom: 28px;
}

/* Streamlit input tweaks */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label,
.stDateInput label {
    font-weight: 600 !important;
    color: var(--ink-soft) !important;
    font-size: 13px !important;
    margin-bottom: 2px !important;
}

div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    font-size: 14px !important;
}

div[data-testid="stNumberInput"] input:focus,
div[data-baseweb="select"]:focus-within > div {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* Analyze button */
div.stButton > button {
    background: var(--accent);
    color: white;
    font-weight: 600;
    font-size: 15px;
    padding: 12px 0;
    border-radius: 10px;
    border: none;
    letter-spacing: 0.2px;
    box-shadow: none;
    transition: background 0.15s ease;
}
div.stButton > button:hover {
    background: #4338ca;
    color: white;
}
div.stButton > button:active {
    background: #3730a3;
    color: white;
}

/* Result cards */
.result-fraud,
.result-safe {
    padding: 30px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid var(--border);
}

.result-fraud {
    background: var(--danger-soft);
    border-color: #fecaca;
}

.result-safe {
    background: var(--success-soft);
    border-color: #bbf7d0;
}

.result-icon {
    font-size: 32px;
    margin-bottom: 8px;
}

.result-title {
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 4px;
    letter-spacing: -0.2px;
}
.result-fraud .result-title { color: var(--danger); }
.result-safe .result-title { color: var(--success); }

.result-desc {
    color: var(--ink-soft);
    font-size: 14px;
    margin-bottom: 18px;
}

.confidence-label {
    font-size: 11.5px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
}
.confidence-value {
    font-size: 26px;
    font-weight: 800;
}
.result-fraud .confidence-value { color: var(--danger); }
.result-safe .confidence-value { color: var(--success); }

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background-color: var(--accent) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 2rem 0 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: var(--muted);
    margin-top: 48px;
    font-size: 12.5px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
}
.footer b {
    color: var(--ink-soft);
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")


model = load_model()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("### 💳 FraudGuard")
    st.markdown(
        "An AI-powered credit card fraud detection tool that analyzes "
        "transaction details in real time using a trained **Random Forest** "
        "classifier."
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Enter the transaction details\n"
        "2. Fill in location & timing info\n"
        "3. Click **Analyze Transaction**\n"
        "4. Get an instant fraud risk verdict"
    )
    st.markdown("---")
    st.markdown("### Notes")
    st.markdown(
        "This tool is for demonstration purposes and should not be used "
        "as the sole basis for real financial decisions."
    )


# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">⚡ Real-time AI detection</div>
        <div class="hero-title">FraudGuard</div>
        <div class="hero-subtitle">AI-powered credit card fraud detection using a Random Forest classifier</div>
        <div class="meta-row">
            <div class="meta-item">
                <div class="value">Random Forest</div>
                <div class="label">Model</div>
            </div>
            <div class="meta-item">
                <div class="value">27</div>
                <div class="label">Features</div>
            </div>
            <div class="meta-item">
                <div class="value">Fraud / Legit</div>
                <div class="label">Output</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TRANSACTION DETAILS
# --------------------------------------------------

st.markdown(
    '<div class="section-header"><span class="step">01</span><h3>Transaction details</h3></div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    amount = st.number_input(
        "Transaction amount ($)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

with col_b:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col_c:
    category = st.selectbox(
        "Transaction category",
        [
            "entertainment",
            "food_dining",
            "gas_transport",
            "grocery_net",
            "grocery_pos",
            "health_fitness",
            "home",
            "kids_pets",
            "misc_net",
            "misc_pos",
            "personal_care",
            "shopping_net",
            "shopping_pos",
            "travel"
        ]
    )

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# LOCATION DETAILS
# --------------------------------------------------

st.markdown(
    '<div class="section-header"><span class="step">02</span><h3>Location details</h3></div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    zip_code = st.number_input(
        "ZIP code",
        min_value=0,
        value=10001
    )

    latitude = st.number_input(
        "Customer latitude",
        value=40.7128,
        format="%.6f"
    )

    longitude = st.number_input(
        "Customer longitude",
        value=-74.0060,
        format="%.6f"
    )

with col2:
    city_population = st.number_input(
        "City population",
        min_value=0,
        value=100000
    )

    merchant_latitude = st.number_input(
        "Merchant latitude",
        value=40.7128,
        format="%.6f"
    )

    merchant_longitude = st.number_input(
        "Merchant longitude",
        value=-74.0060,
        format="%.6f"
    )

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# DATE AND TIME
# --------------------------------------------------

st.markdown(
    '<div class="section-header"><span class="step">03</span><h3>Transaction time</h3></div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

transaction_datetime = st.datetime_input(
    "Transaction date & time",
    value=datetime.now()
)

st.markdown("</div>", unsafe_allow_html=True)

hour = transaction_datetime.hour
day = transaction_datetime.day
month = transaction_datetime.month
day_of_week = transaction_datetime.weekday()

unix_time = int(transaction_datetime.timestamp())


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

analyze_clicked = st.button("Analyze transaction", use_container_width=True)

if analyze_clicked:

    # Gender encoding
    # F = 0
    # M = 1
    gender_value = 0 if gender == "Female" else 1

    # Create all 27 features
    input_data = {
        "amt": amount,
        "gender": gender_value,
        "zip": zip_code,
        "lat": latitude,
        "long": longitude,
        "city_pop": city_population,
        "unix_time": unix_time,
        "merch_lat": merchant_latitude,
        "merch_long": merchant_longitude,
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,

        "category_entertainment": 0,
        "category_food_dining": 0,
        "category_gas_transport": 0,
        "category_grocery_net": 0,
        "category_grocery_pos": 0,
        "category_health_fitness": 0,
        "category_home": 0,
        "category_kids_pets": 0,
        "category_misc_net": 0,
        "category_misc_pos": 0,
        "category_personal_care": 0,
        "category_shopping_net": 0,
        "category_shopping_pos": 0,
        "category_travel": 0
    }

    # Activate selected category
    category_column = "category_" + category

    if category_column in input_data:
        input_data[category_column] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure exact feature order
    feature_order = [
        "amt",
        "gender",
        "zip",
        "lat",
        "long",
        "city_pop",
        "unix_time",
        "merch_lat",
        "merch_long",
        "hour",
        "day",
        "month",
        "day_of_week",
        "category_entertainment",
        "category_food_dining",
        "category_gas_transport",
        "category_grocery_net",
        "category_grocery_pos",
        "category_health_fitness",
        "category_home",
        "category_kids_pets",
        "category_misc_net",
        "category_misc_pos",
        "category_personal_care",
        "category_shopping_net",
        "category_shopping_pos",
        "category_travel"
    ]

    input_df = input_df[feature_order]

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get probability
    probability = model.predict_proba(input_df)[0]

    confidence = probability[prediction] * 100

    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.markdown(
        '<div class="section-header" style="margin-top: 32px;"><span class="step">04</span><h3>Analysis result</h3></div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.markdown(
            f"""
            <div class="result-fraud">
                <div class="result-icon">🚨</div>
                <div class="result-title">Fraudulent transaction</div>
                <div class="result-desc">This transaction has been flagged as potentially fraudulent.</div>
                <div class="confidence-label">Model confidence</div>
                <div class="confidence-value">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title">Legitimate transaction</div>
                <div class="result-desc">This transaction appears to be legitimate.</div>
                <div class="confidence-label">Model confidence</div>
                <div class="confidence-value">{confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(int(confidence))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Built with <b>Python</b> • <b>Scikit-learn</b> • <b>Random Forest</b> • <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True
)
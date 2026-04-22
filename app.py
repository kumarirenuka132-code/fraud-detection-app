import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection", layout="wide")

# ===============================
# CSS (FIXED)
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b132b, #1c2541);
    color: white;
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
}

/* Sidebar */
.sidebar-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Result */
.safe-box {
    background: #064e3b;
    padding: 10px;
    border-radius: 10px;
}
.fraud-box {
    background: #7f1d1d;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# MODEL LOAD
# ===============================
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# ===============================
# SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_score" not in st.session_state:
    st.session_state.last_score = "N/A"

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Fraud Detection System</div>', unsafe_allow_html=True)

    st.write("### Features")
    st.success("Real-time Fraud Check")
    st.success("Risk Score")
    st.success("History")
    st.success("CSV Download")

    st.write("### Model")
    st.info("Random Forest")

    st.write("### Status")
    st.success("Running")

    if st.button("Clear History"):
        st.session_state.history = []

# ===============================
# MAIN TITLE
# ===============================
st.title("Fraud Detection Dashboard")
st.write("Detect suspicious transactions using AI")

# ===============================
# METRICS
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("System Status", "Active")
col2.metric("Model", "Random Forest")
col3.metric("Predictions", len(st.session_state.history))
col4.metric("Last Risk", st.session_state.last_score)

# ===============================
# INPUT
# ===============================
st.subheader("Transaction Input")

col1, col2 = st.columns(2)

amt = col1.number_input("Amount", value=100.0)

category = col2.selectbox("Category", [
    "personal_care", "health_fitness", "misc_pos", "travel",
    "gas_transport", "food_dining", "shopping_net"
])

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Fraud"):
    input_data = pd.DataFrame({
        'cc_num': [1],
        'merchant': [1],
        'category': [1],
        'amt': [amt],
        'gender': [1],
        'street': [1],
        'city': [1],
        'state': [1],
        'zip': [1],
        'lat': [1],
        'long': [1],
        'city_pop': [1],
        'job': [1],
        'unix_time': [1],
        'merch_lat': [1],
        'merch_long': [1],
        'trans_date_trans_time_year': [2020],
        'trans_date_trans_time_month': [1],
        'trans_date_trans_time_day': [1],
        'dob_year': [1990],
        'dob_month': [1],
        'dob_day': [1]
    })

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    risk = prob * 100
    st.session_state.last_score = f"{risk:.2f}%"

    result = "Fraud" if pred == 1 else "Legit"

    st.session_state.history.append({
        "Time": datetime.now(),
        "Amount": amt,
        "Result": result,
        "Risk": risk
    })

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("Result")

    st.metric("Risk Score", f"{risk:.2f}%")

    if pred == 1:
        st.markdown('<div class="fraud-box">Fraud Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="safe-box">Safe Transaction</div>', unsafe_allow_html=True)

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig)

# ===============================
# HISTORY
# ===============================
st.subheader("Prediction History")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv)

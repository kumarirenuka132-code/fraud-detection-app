import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Sentinel AI | Fraud Detection",
    layout="wide"
)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except:
        return None

model = load_model()

# ==============================
# ICONS + CSS
# ==============================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

<style>
body {
    background: linear-gradient(135deg,#020617,#0f172a);
    color:white;
}
.header {
    font-size:32px;
    font-weight:bold;
}
.card {
    padding:20px;
    border-radius:12px;
    background:#111827;
    margin-bottom:10px;
}
.metric {
    font-size:20px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="header">
<i class="fa-solid fa-shield-halved"></i> Fraud Detection Dashboard
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model file missing ❌")
    st.stop()

# ==============================
# SESSION
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# INPUT
# ==============================
st.markdown("### <i class='fa-solid fa-magnifying-glass'></i> Transaction Input", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", 0.0, 100000.0, 1000.0)

with col2:
    category = st.selectbox("Category", [
        'shopping_net','travel','food_dining',
        'gas_transport','entertainment'
    ])

# ==============================
# ANALYZE BUTTON
# ==============================
if st.button("Analyze Transaction"):

    category_map = {
        'shopping_net':0,
        'travel':1,
        'food_dining':2,
        'gas_transport':3,
        'entertainment':4
    }

    input_df = pd.DataFrame({
        'cc_num':[1],
        'merchant':[1],
        'category':[category_map[category]],
        'amt':[amount],
        'gender':[1],
        'street':[1],
        'city':[1],
        'state':[1],
        'zip':[1],
        'lat':[1],
        'long':[1],
        'city_pop':[10000],
        'job':[1],
        'unix_time':[1],
        'merch_lat':[1],
        'merch_long':[1],
        'trans_date_trans_time_year':[2020],
        'trans_date_trans_time_month':[1],
        'trans_date_trans_time_day':[1],
        'dob_year':[1990],
        'dob_month':[1],
        'dob_day':[1]
    })

    prob = model.predict_proba(input_df)[0][1]
    risk = prob * 100

    decision = "Fraud" if risk > 50 else "Safe"

    st.session_state.history.append({
        "time":datetime.now(),
        "amount":amount,
        "category":category,
        "risk":risk,
        "decision":decision
    })

# ==============================
# RESULT
# ==============================
if len(st.session_state.history) > 0:

    last = st.session_state.history[-1]

    st.markdown("### <i class='fa-solid fa-chart-line'></i> Result", unsafe_allow_html=True)

    if last["decision"] == "Fraud":
        st.markdown(f"""
        <div class="card" style="color:red;">
        <i class="fa-solid fa-triangle-exclamation"></i>
        FRAUD DETECTED <br>
        Risk Score: {last['risk']:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card" style="color:lightgreen;">
        <i class="fa-solid fa-circle-check"></i>
        SAFE TRANSACTION <br>
        Risk Score: {last['risk']:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ==============================
    # GAUGE CHART
    # ==============================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=last['risk'],
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if last['risk']>50 else "green"}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# ANALYTICS
# ==============================
if len(st.session_state.history) > 0:

    df = pd.DataFrame(st.session_state.history)

    st.markdown("### <i class='fa-solid fa-chart-pie'></i> Analytics", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="risk", color="decision")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(df, names="decision")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df)

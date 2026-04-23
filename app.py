import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Fraud Detection",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl","rb"))

model = load_model()

# -------------------------
# HEADER (UPDATED NAME)
# -------------------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<div style="padding:20px;
background:linear-gradient(90deg,#2563eb,#7c3aed);
border-radius:15px;
color:white;">
<h2><i class="fa-solid fa-shield-halved"></i> Fraud Detection</h2>
<p>AI Powered Transaction Monitoring System</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# SESSION STATE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# SIDEBAR
# -------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Check Transaction", "Analytics"]
)

# -------------------------
# DASHBOARD
# -------------------------
if menu == "Dashboard":

    total = len(st.session_state.history)
    fraud = len([x for x in st.session_state.history if x["decision"]=="Fraud"])
    safe = total - fraud

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Detected", fraud)
    col3.metric("Safe Transactions", safe)

    if total > 0:
        df = pd.DataFrame(st.session_state.history)

        fig = px.pie(df, names="decision", title="Fraud vs Safe")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# CHECK TRANSACTION
# -------------------------
elif menu == "Check Transaction":

    amount = st.number_input("Amount", value=1000.0)
    category = st.selectbox("Category", ["shopping","travel","food","gas"])

    if st.button("Check Fraud"):

        input_data = pd.DataFrame({
            "amt":[amount],
            "category":[0]
        })

        prob = model.predict_proba(input_data)[0][1]
        risk = prob * 100

        decision = "Fraud" if risk > 50 else "Safe"

        st.session_state.history.append({
            "time":str(datetime.now()),
            "amount":amount,
            "risk":risk,
            "decision":decision
        })

        st.subheader("Result")

        if decision == "Fraud":
            st.error(f"🚨 Fraud Detected | Risk: {risk:.2f}%")
        else:
            st.success(f"✅ Safe Transaction | Risk: {risk:.2f}%")

        st.progress(int(risk))

# -------------------------
# ANALYTICS
# -------------------------
elif menu == "Analytics":

    if len(st.session_state.history) == 0:
        st.warning("No data yet")
    else:
        df = pd.DataFrame(st.session_state.history)

        fig1 = px.histogram(df, x="risk", title="Risk Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(df, y="risk", title="Risk Trend")
        st.plotly_chart(fig2, use_container_width=True)

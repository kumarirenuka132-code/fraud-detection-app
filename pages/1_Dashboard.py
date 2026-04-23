import streamlit as st
import pandas as pd

st.title("Dashboard")

if "history" not in st.session_state:
    st.session_state.history = []

if len(st.session_state.history) == 0:
    st.warning("No data available yet")
else:
    df = pd.DataFrame(st.session_state.history)

    st.subheader("Risk Trend")
    st.line_chart(df["Risk Score"])

    st.subheader("Fraud vs Legit")
    st.bar_chart(df["Prediction"].value_counts())

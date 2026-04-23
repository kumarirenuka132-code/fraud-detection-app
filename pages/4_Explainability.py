import streamlit as st

st.title("Explainability")

st.write("Why was this transaction flagged?")

st.subheader("Rules Used")

st.write("""
- High amount → higher fraud risk
- Suspicious category → higher fraud risk
- Pattern mismatch → flagged
""")

st.subheader("Model Explanation")

st.info("Random Forest model analyzes transaction patterns to detect anomalies.")

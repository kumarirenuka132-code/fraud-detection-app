import streamlit as st

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("Fraud Detection System")
st.write("AI-based system for detecting fraudulent transactions.")

st.info("Use the navigation menu to open different sections of the system.")

st.subheader("Quick Navigation")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Dashboard.py", label="Open Dashboard")
    st.page_link("pages/2_Risk_Scoring.py", label="Open Risk Scoring")

with col2:
    st.page_link("pages/3_History.py", label="Open History")
    st.page_link("pages/4_Explainability.py", label="Open Explainability")

st.markdown("""
### Features
- Fraud Prediction
- Risk Score
- History Tracking
- Explainable AI
- Dashboard Analytics
""")

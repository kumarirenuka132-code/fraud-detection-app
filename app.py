import streamlit as st

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("Fraud Detection System")
st.write("AI-based system for detecting fraudulent transactions.")

st.info("Open the left sidebar menu to navigate between pages.")

st.markdown("""
### Available Sections
- Dashboard
- Risk Scoring
- History
- Explainability
""")

st.write("If pages are not visible, make sure the `pages` folder and page files are created correctly in GitHub.")

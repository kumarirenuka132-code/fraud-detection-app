import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    st.markdown("### Features")
    st.success("Real-time Fraud Prediction")
    st.success("Risk Score")
    st.success("Prediction History")
    st.success("CSV Download")
    st.success("Audit Summary")

    st.markdown("### Model")
    st.info("Random Forest Classifier")

    st.markdown("### Status")
    st.success("App deployed successfully")

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.session_state.latest_result = None
        st.rerun()

# -------------------------------
# HEADER
# -------------------------------
st.title("💳 Fraud Detection Dashboard")
st.write("AI-based system for detecting suspicious transactions")

if model is None:
    st.error("model.pkl missing!")
    st.stop()

# -------------------------------
# TOP METRICS
# -------------------------------
pred_count = len(st.session_state.history)

last_risk = "N/A"
if pred_count > 0:
    last_risk = f"{st.session_state.history[-1]['Risk Score (%)']:.2f}%"

c1, c2, c3, c4 = st.columns(4)

c1.metric("System Status", "Active")
c2.metric("Model Used", "Random Forest")
c3.metric("Predictions Made", pred_count)
c4.metric("Last Risk Score", last_risk)

# -------------------------------
# INPUT
# -------------------------------
st.subheader("Transaction Input")

col1, col2 = st.columns(2)

with col1:
    amt = st.number_input("Transaction Amount", value=100.0)

with col2:
    category_options = [
        'personal_care','health_fitness','misc_pos','travel',
        'gas_transport','food_dining','shopping_net','groceries',
        'home','entertainment','kids_pets','hotels_motels',
        'shopping_mall','utility'
    ]
    category_input = st.selectbox("Category", category_options)

category_map = {cat:i for i,cat in enumerate(category_options)}
encoded_category = category_map[category_input]

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚨 Predict Fraud", use_container_width=True):

    input_data = pd.DataFrame({
        'cc_num':[1],'merchant':[1],'category':[encoded_category],
        'amt':[amt],'gender':[1],'street':[1],'city':[1],
        'state':[1],'zip':[1],'lat':[1],'long':[1],
        'city_pop':[1],'job':[1],'unix_time':[1],
        'merch_lat':[1],'merch_long':[1],
        'trans_date_trans_time_year':[2020],
        'trans_date_trans_time_month':[6],
        'trans_date_trans_time_day':[21],
        'dob_year':[1968],'dob_month':[3],'dob_day':[19]
    })

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    risk = prob * 100

    risk_level = "Low"
    if risk > 70:
        risk_level = "High"
    elif risk > 30:
        risk_level = "Medium"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # SAVE HISTORY
    st.session_state.history.append({
        "Time":timestamp,
        "Amount":amt,
        "Category":category_input,
        "Prediction":"Fraud" if pred==1 else "Legitimate",
        "Fraud Probability":round(prob,4),
        "Risk Score (%)":round(risk,2),
        "Risk Level":risk_level
    })

    st.session_state.latest_result = {
        "prediction":pred,
        "prob":prob,
        "risk":risk,
        "level":risk_level
    }

    st.rerun()

# -------------------------------
# SHOW RESULT
# -------------------------------
if st.session_state.latest_result:

    res = st.session_state.latest_result

    st.subheader("Prediction Output")

    r1,r2,r3 = st.columns(3)
    r1.metric("Fraud Probability", f"{res['prob']:.4f}")
    r2.metric("Risk Score", f"{res['risk']:.2f}%")
    r3.metric("Risk Level", res['level'])

    st.progress(int(res['risk']))

    if res["prediction"] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction is Legitimate")

# -------------------------------
# HISTORY + CHARTS
# -------------------------------
if len(st.session_state.history) > 0:

    st.subheader("Prediction History")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "history.csv"
    )

    st.subheader("Risk Trend")
    st.line_chart(df["Risk Score (%)"])

    st.subheader("Category Risk")
    st.bar_chart(df.groupby("Category")["Risk Score (%)"].mean())

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit + ML")

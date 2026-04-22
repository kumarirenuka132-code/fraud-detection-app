import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="shield",
    layout="wide"
)

# -------------------------------
# LOAD FONT AWESOME + CSS
# -------------------------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 95%;
}
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 1rem;
    margin-bottom: 0.7rem;
}
.icon-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 1rem;
    margin-bottom: 0.7rem;
}
.metric-card {
    background: linear-gradient(145deg, #172033, #1e293b);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #243047;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.metric-label {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 8px;
}
.metric-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 700;
}
.panel-card {
    background: #111827;
    border: 1px solid #253046;
    border-radius: 18px;
    padding: 18px;
}
.safe-box {
    background: linear-gradient(90deg, #08361d, #0f5132);
    color: #dcfce7;
    padding: 16px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    border-left: 5px solid #22c55e;
}
.fraud-box {
    background: linear-gradient(90deg, #3b0b0b, #7f1d1d);
    color: #fee2e2;
    padding: 16px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
    border-left: 5px solid #ef4444;
}
.note-box {
    color: #94a3b8;
    font-size: 13px;
    padding: 10px 0 0 0;
}
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
}
.small-icon-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}
.custom-divider {
    border-top: 1px solid #243047;
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
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
    st.markdown('<div class="sidebar-title"><i class="fa-solid fa-sliders"></i> Control Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="small-icon-title"><i class="fa-solid fa-circle-info"></i> About Project</div>', unsafe_allow_html=True)
    st.write("This app predicts whether a transaction is fraudulent using a trained Random Forest model.")

    st.markdown('<div class="small-icon-title"><i class="fa-solid fa-layer-group"></i> Features</div>', unsafe_allow_html=True)
    st.info("Real-time Fraud Prediction")
    st.info("Risk Score Analysis")
    st.info("Prediction History")
    st.info("CSV Download")
    st.info("Audit Summary")
    st.info("Risk Gauge Dashboard")

    st.markdown('<div class="small-icon-title"><i class="fa-solid fa-microchip"></i> Model</div>', unsafe_allow_html=True)
    st.success("Random Forest Classifier")

    st.markdown('<div class="small-icon-title"><i class="fa-solid fa-signal"></i> Status</div>', unsafe_allow_html=True)
    st.success("Application is running")

    if st.button("Clear Prediction History", use_container_width=True):
        st.session_state.history = []
        st.session_state.latest_result = None
        st.rerun()

# -------------------------------
# MAIN HEADER
# -------------------------------
st.markdown('<div class="main-title">Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-based audit interface for detecting suspicious financial transactions with dynamic risk scoring.</div>',
    unsafe_allow_html=True
)

if model is None:
    st.error("model.pkl not found. Please upload model.pkl in the same folder as app.py.")
    st.stop()

# -------------------------------
# METRICS
# -------------------------------
pred_count = len(st.session_state.history)
last_risk = "N/A"
if pred_count > 0:
    last_risk = f"{st.session_state.history[-1]['Risk Score (%)']:.2f}%"

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">System Status</div>
        <div class="metric-value">Active</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Used</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predictions Made</div>
        <div class="metric-value">{pred_count}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Risk Score</div>
        <div class="metric-value">{last_risk}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "History & Charts", "About"])

# ===============================
# TAB 1
# ===============================
with tab1:
    st.markdown('<div class="icon-title"><i class="fa-solid fa-credit-card"></i> Transaction Input</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        amt = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)

    with c2:
        category_options = [
            'personal_care', 'health_fitness', 'misc_pos', 'travel',
            'gas_transport', 'food_dining', 'shopping_net', 'groceries',
            'home', 'entertainment', 'kids_pets', 'hotels_motels',
            'shopping_mall', 'utility'
        ]
        category_input = st.selectbox("Transaction Category", options=category_options)

    category_mapping = {cat: i for i, cat in enumerate(category_options)}
    encoded_category = category_mapping[category_input]

    st.markdown('<div class="note-box">Some features are internally prefilled for demonstration because the trained model expects the complete training feature structure.</div>', unsafe_allow_html=True)

    if st.button("Run Fraud Check", use_container_width=True):
        try:
            input_data = pd.DataFrame({
                'cc_num': [2291163933867240],
                'merchant': [319],
                'category': [encoded_category],
                'amt': [amt],
                'gender': [1],
                'street': [351],
                'city': [123],
                'state': [45],
                'zip': [3033],
                'lat': [33.9659],
                'long': [-80.9355],
                'city_pop': [333497],
                'job': [123],
                'unix_time': [1371816865],
                'merch_lat': [33.986391],
                'merch_long': [-81.200714],
                'trans_date_trans_time_year': [2020],
                'trans_date_trans_time_month': [6],
                'trans_date_trans_time_day': [21],
                'dob_year': [1968],
                'dob_month': [3],
                'dob_day': [19]
            })

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            risk_score = float(probability * 100)

            risk_level = (
                "Low Risk" if risk_score < 30 else
                "Medium Risk" if risk_score < 70 else
                "High Risk"
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            history_row = {
                "Time": timestamp,
                "Amount": amt,
                "Category": category_input,
                "Prediction": "Fraud" if prediction == 1 else "Legitimate",
                "Fraud Probability": round(float(probability), 4),
                "Risk Score (%)": round(risk_score, 2),
                "Risk Level": risk_level
            }

            st.session_state.history.append(history_row)

            st.session_state.latest_result = {
                "timestamp": timestamp,
                "amount": amt,
                "category": category_input,
                "prediction": prediction,
                "probability": probability,
                "risk_score": risk_score,
                "risk_level": risk_level
            }

            st.rerun()

        except Exception as e:
            st.error(f"Prediction error: {e}")

    if st.session_state.latest_result:
        res = st.session_state.latest_result

        st.markdown('<div class="icon-title"><i class="fa-solid fa-chart-column"></i> Prediction Output</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Fraud Probability", f"{res['probability']:.4f}")
        with r2:
            st.metric("Risk Score", f"{res['risk_score']:.2f}%")
        with r3:
            st.metric("Risk Level", res["risk_level"])

        st.progress(min(int(res["risk_score"]), 100))

        if res["prediction"] == 1:
            st.markdown('<div class="fraud-box">Fraudulent Transaction Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">Transaction is Likely Legitimate</div>', unsafe_allow_html=True)

        st.markdown('<div class="icon-title"><i class="fa-solid fa-gauge-high"></i> Risk Gauge Meter</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=res["risk_score"],
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'bar': {'color': "darkred"}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="icon-title"><i class="fa-solid fa-magnifying-glass-chart"></i> Audit Explanation</div>', unsafe_allow_html=True)

        exp1, exp2 = st.columns(2)

        with exp1:
            st.write("**Amount Analysis**")
            if res["amount"] > 5000:
                st.warning("High transaction amount may increase fraud suspicion.")
            elif res["amount"] > 1000:
                st.info("Moderately high amount. This may require additional audit review.")
            else:
                st.success("Amount appears in a lower-risk range.")

        with exp2:
            st.write("**Category Analysis**")
            if res["category"] in ["travel", "shopping_net", "misc_pos"]:
                st.warning(f"Category '{res['category']}' may show relatively riskier patterns in demo analysis.")
            else:
                st.success(f"Category '{res['category']}' appears normal in this demo context.")

        st.markdown('<div class="icon-title"><i class="fa-solid fa-file-shield"></i> Final Audit Summary</div>', unsafe_allow_html=True)
        st.write(f"- **Timestamp:** {res['timestamp']}")
        st.write(f"- **Transaction Amount:** {res['amount']}")
        st.write(f"- **Selected Category:** {res['category']}")
        st.write(f"- **Prediction:** {'Fraud' if res['prediction'] == 1 else 'Legitimate'}")
        st.write(f"- **Risk Score:** {res['risk_score']:.2f}%")
        st.write(f"- **Model Used:** Random Forest")

        report = f"""
Fraud Detection Report
======================

Timestamp: {res['timestamp']}
Transaction Amount: {res['amount']}
Category: {res['category']}
Prediction: {'Fraud' if res['prediction'] == 1 else 'Legitimate'}
Fraud Probability: {res['probability']:.4f}
Risk Score: {res['risk_score']:.2f}%
Risk Level: {res['risk_level']}
Model Used: Random Forest
"""
        st.download_button(
            "Download Result Report",
            report,
            file_name="fraud_report.txt",
            mime="text/plain",
            use_container_width=True
        )

# ===============================
# TAB 2
# ===============================
with tab2:
    st.markdown('<div class="icon-title"><i class="fa-solid fa-clock-rotate-left"></i> Prediction History</div>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No prediction history yet. Make at least one prediction to see history and charts.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction History as CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown('<div class="icon-title"><i class="fa-solid fa-chart-line"></i> Risk Score Trend</div>', unsafe_allow_html=True)
        trend_df = hist_df[["Time", "Risk Score (%)"]].copy()
        trend_df = trend_df.set_index("Time")
        st.line_chart(trend_df)

        st.markdown('<div class="icon-title"><i class="fa-solid fa-chart-simple"></i> Category-wise Average Risk</div>', unsafe_allow_html=True)
        cat_chart = hist_df.groupby("Category", as_index=True)["Risk Score (%)"].mean()
        st.bar_chart(cat_chart)

        st.markdown('<div class="icon-title"><i class="fa-solid fa-chart-pie"></i> Prediction Distribution</div>', unsafe_allow_html=True)
        pred_count_chart = hist_df["Prediction"].value_counts()
        st.bar_chart(pred_count_chart)

# ===============================
# TAB 3
# ===============================
with tab3:
    st.markdown('<div class="icon-title"><i class="fa-solid fa-diagram-project"></i> Project Overview</div>', unsafe_allow_html=True)
    st.write("""
This fraud detection dashboard is built using **Streamlit** and a **Random Forest Classifier**.
It predicts whether a transaction is suspicious and assigns a **dynamic risk score**.

### Main Components
- Data preprocessing and model training in Python
- Fraud classification using machine learning
- Dynamic fraud probability and risk score
- Real-time deployed UI using Streamlit
- GitHub-based deployment

### Deployment Stack
- **Frontend/UI:** Streamlit
- **Backend Logic:** Python
- **Model:** Scikit-learn Random Forest
- **Deployment:** GitHub + Streamlit Cloud
    """)

st.markdown("---")
st.caption("Built with Streamlit, Plotly, and Scikit-learn.")

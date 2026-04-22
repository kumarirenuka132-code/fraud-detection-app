import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------
# PREMIUM CSS + ICONS
# -------------------------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
/* Global */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1730 45%, #111827 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 96%;
}

/* Titles */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.1rem;
    letter-spacing: 0.3px;
}
.sub-title {
    font-size: 16px;
    color: #a5b4c7;
    margin-bottom: 1.4rem;
}
.icon-title {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 1rem;
    margin-bottom: 0.9rem;
}
.icon-badge {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    background: linear-gradient(135deg, rgba(59,130,246,0.20), rgba(168,85,247,0.20));
    border: 1px solid rgba(255,255,255,0.10);
    color: #60a5fa;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

/* Cards */
.metric-card {
    background: rgba(17, 24, 39, 0.62);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.28);
}
.metric-label {
    color: #93a4bb;
    font-size: 14px;
    margin-bottom: 8px;
}
.metric-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
    line-height: 1.1;
}
.panel-card {
    background: rgba(17, 24, 39, 0.55);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

/* Result boxes */
.safe-box {
    background: linear-gradient(90deg, rgba(22,163,74,0.22), rgba(34,197,94,0.14));
    border: 1px solid rgba(34,197,94,0.35);
    color: #dcfce7;
    padding: 16px;
    border-radius: 16px;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
}
.fraud-box {
    background: linear-gradient(90deg, rgba(220,38,38,0.22), rgba(239,68,68,0.14));
    border: 1px solid rgba(239,68,68,0.35);
    color: #fee2e2;
    padding: 16px;
    border-radius: 16px;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
}

/* Helper text */
.note-box {
    color: #98a8bc;
    font-size: 13px;
    padding-top: 8px;
    padding-bottom: 8px;
}

/* Sidebar */
.sidebar-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin-bottom: 12px;
}
.sidebar-icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: linear-gradient(135deg, rgba(14,165,233,0.22), rgba(59,130,246,0.22));
    color: #38bdf8;
    border: 1px solid rgba(255,255,255,0.10);
}
.sidebar-section {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 10px;
    margin-bottom: 8px;
}
.sidebar-divider {
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 14px 0 18px 0;
}

/* Tab style helpers */
.tab-subtle {
    color: #9db0c6;
    font-size: 14px;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 700;
    padding: 0.7rem 1rem;
    box-shadow: 0 10px 24px rgba(37,99,235,0.25);
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    color: white;
    border: none;
}

/* Download buttons */
div.stDownloadButton > button {
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 700;
}

/* Make dataframe area nicer */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

/* Metrics spacing */
[data-testid="metric-container"] {
    background: rgba(17, 24, 39, 0.40);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px;
    border-radius: 18px;
}

/* Hide default streamlit footer */
footer {visibility: hidden;}
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
    st.markdown("""
    <div class="sidebar-heading">
        <div class="sidebar-icon"><i class="fa-solid fa-building-columns"></i></div>
        <div>Banking Panel</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <i class="fa-solid fa-circle-info" style="color:#38bdf8;"></i>
        <span>About Project</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("This application predicts suspicious transactions using a trained Random Forest fraud model.")

    st.markdown("""
    <div class="sidebar-section">
        <i class="fa-solid fa-bolt" style="color:#a78bfa;"></i>
        <span>Features</span>
    </div>
    """, unsafe_allow_html=True)
    st.success("Real-time Fraud Check")
    st.success("Dynamic Risk Scoring")
    st.success("Prediction History")
    st.success("CSV Export")
    st.success("Audit Summary")
    st.success("Risk Gauge Meter")

    st.markdown("""
    <div class="sidebar-section">
        <i class="fa-solid fa-microchip" style="color:#f59e0b;"></i>
        <span>Model</span>
    </div>
    """, unsafe_allow_html=True)
    st.info("Random Forest Classifier")

    st.markdown("""
    <div class="sidebar-section">
        <i class="fa-solid fa-signal" style="color:#22c55e;"></i>
        <span>Status</span>
    </div>
    """, unsafe_allow_html=True)
    st.success("Application Running")

    if st.button("Clear Prediction History", use_container_width=True):
        st.session_state.history = []
        st.session_state.latest_result = None
        st.rerun()

# -------------------------------
# MAIN HEADER
# -------------------------------
st.markdown('<div class="main-title">Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Premium audit interface for monitoring suspicious financial transactions with explainable risk scoring.</div>',
    unsafe_allow_html=True
)

if model is None:
    st.error("model.pkl not found. Please keep model.pkl in the same folder as app.py.")
    st.stop()

# -------------------------------
# TOP METRICS
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

st.markdown("")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "History & Charts", "About"])

# ===============================
# TAB 1 - PREDICTION
# ===============================
with tab1:
    st.markdown("""
    <div class="icon-title">
        <div class="icon-badge"><i class="fa-solid fa-credit-card"></i></div>
        <div>Transaction Input</div>
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown(
        '<div class="note-box">Some training features are internally prefilled because the current demo model expects the full processed feature structure.</div>',
        unsafe_allow_html=True
    )

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

        st.markdown("""
        <div class="icon-title">
            <div class="icon-badge"><i class="fa-solid fa-chart-column"></i></div>
            <div>Prediction Output</div>
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown("""
        <div class="icon-title">
            <div class="icon-badge"><i class="fa-solid fa-gauge-high"></i></div>
            <div>Risk Gauge Meter</div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=res["risk_score"],
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "#14532d"},
                    {'range': [30, 70], 'color': "#a16207"},
                    {'range': [70, 100], 'color': "#991b1b"}
                ],
                'bar': {'color': "#ef4444"}
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#e5e7eb", 'family': "Segoe UI"},
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="icon-title">
            <div class="icon-badge"><i class="fa-solid fa-magnifying-glass-chart"></i></div>
            <div>Audit Explanation</div>
        </div>
        """, unsafe_allow_html=True)

        exp1, exp2 = st.columns(2)

        with exp1:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.write("**Amount Analysis**")
            if res["amount"] > 5000:
                st.warning("High transaction amount may increase fraud suspicion.")
            elif res["amount"] > 1000:
                st.info("Moderately high amount. This may require additional audit review.")
            else:
                st.success("Amount appears in a lower-risk range.")
            st.markdown('</div>', unsafe_allow_html=True)

        with exp2:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.write("**Category Analysis**")
            if res["category"] in ["travel", "shopping_net", "misc_pos"]:
                st.warning(f"Category '{res['category']}' may show relatively riskier patterns in demo analysis.")
            else:
                st.success(f"Category '{res['category']}' appears normal in this demo context.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="icon-title">
            <div class="icon-badge"><i class="fa-solid fa-file-shield"></i></div>
            <div>Final Audit Summary</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.write(f"- **Timestamp:** {res['timestamp']}")
        st.write(f"- **Transaction Amount:** {res['amount']}")
        st.write(f"- **Selected Category:** {res['category']}")
        st.write(f"- **Prediction:** {'Fraud' if res['prediction'] == 1 else 'Legitimate'}")
        st.write(f"- **Risk Score:** {res['risk_score']:.2f}%")
        st.write(f"- **Model Used:** Random Forest")
        st.markdown('</div>', unsafe_allow_html=True)

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
# TAB 2 - HISTORY
# ===============================
with tab2:
    st.markdown("""
    <div class="icon-title">
        <div class="icon-badge"><i class="fa-solid fa-clock-rotate-left"></i></div>
        <div>Prediction History</div>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No prediction history yet. Make at least one prediction to view analytics.")
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

        st.markdown("""
        <div class="icon-title">
            <div class="icon-badge"><i class="fa-solid fa-chart-line"></i></div>
            <div>Risk Score Trend</div>
        </div>
        """, unsafe_allow_html=True)

        trend_df = hist_df[["Time", "Risk Score (%)"]].copy()
        trend_df = trend_df.set_index("Time")
        st.line_chart(trend_df)

        st.markdown("""
        <div class="icon-title">
            <div class="icon-badge"><i class="fa-solid fa-chart-simple"></i></div>
            <div>Category-wise Average Risk</div>
        </div>
        """, unsafe_allow_html=True)

        cat_chart = hist_df.groupby("Category", as_index=True)["Risk Score (%)"].mean()
        st.bar_chart(cat_chart)

        st.markdown("""
        <div class="icon-title">
            <div class="icon-badge"><i class="fa-solid fa-chart-pie"></i></div>
            <div>Prediction Distribution</div>
        </div>
        """, unsafe_allow_html=True)

        pred_count_chart = hist_df["Prediction"].value_counts()
        st.bar_chart(pred_count_chart)

# ===============================
# TAB 3 - ABOUT
# ===============================
with tab3:
    st.markdown("""
    <div class="icon-title">
        <div class="icon-badge"><i class="fa-solid fa-diagram-project"></i></div>
        <div>Project Overview</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.write("""
This fraud detection dashboard is built using **Streamlit** and a **Random Forest Classifier**.

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
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit, Plotly, and Scikit-learn.")

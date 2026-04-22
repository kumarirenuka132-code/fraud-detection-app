import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #081120 0%, #0f172a 45%, #111827 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 96%;
}

/* ---------- HEADINGS ---------- */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.15rem;
    letter-spacing: 0.3px;
}
.sub-title {
    font-size: 16px;
    color: #9fb0c7;
    margin-bottom: 1.4rem;
}
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 1rem;
    margin-bottom: 0.9rem;
}

/* ---------- CARDS ---------- */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}

.metric-label {
    color: #97a8bf;
    font-size: 14px;
    margin-bottom: 8px;
}
.metric-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
    line-height: 1.1;
}

/* ---------- RESULT BOXES ---------- */
.safe-box {
    background: linear-gradient(90deg, rgba(22,163,74,0.22), rgba(34,197,94,0.12));
    border: 1px solid rgba(34,197,94,0.35);
    color: #dcfce7;
    padding: 16px;
    border-radius: 16px;
    font-size: 18px;
    font-weight: 700;
}
.fraud-box {
    background: linear-gradient(90deg, rgba(220,38,38,0.22), rgba(239,68,68,0.12));
    border: 1px solid rgba(239,68,68,0.35);
    color: #fee2e2;
    padding: 16px;
    border-radius: 16px;
    font-size: 18px;
    font-weight: 700;
}

/* ---------- PANELS ---------- */
.info-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px;
}

/* ---------- SIDEBAR ---------- */
.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}
.sidebar-sub {
    font-size: 17px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 12px;
    margin-bottom: 8px;
}

/* ---------- BUTTON ---------- */
div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 700;
    padding: 0.72rem 1rem;
    box-shadow: 0 8px 22px rgba(37,99,235,0.25);
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    color: white;
    border: none;
}

/* ---------- DOWNLOAD BUTTON ---------- */
div.stDownloadButton > button {
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 700;
}

/* ---------- SMALL TEXT ---------- */
.small-note {
    color: #9db0c5;
    font-size: 13px;
    margin-top: 6px;
}

/* ---------- FOOTER ---------- */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

model = load_model()

# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def rule_based_score(amount: float, category: str) -> float:
    score = 0.0

    if amount >= 5000:
        score += 35
    elif amount >= 1000:
        score += 18
    else:
        score += 5

    risky_categories = {
        "travel": 22,
        "shopping_net": 20,
        "misc_pos": 18,
        "gas_transport": 10
    }

    score += risky_categories.get(category, 5)
    return min(score, 100)


def hybrid_score(ml_prob: float, amount: float, category: str) -> float:
    ml_score = ml_prob * 100
    rules_score = rule_based_score(amount, category)

    # Weighted hybrid score
    final_score = (0.7 * ml_score) + (0.3 * rules_score)
    return min(max(final_score, 0), 100)


def risk_label(score: float) -> str:
    if score < 30:
        return "Low"
    elif score < 70:
        return "Medium"
    return "High"


def final_decision(score: float) -> str:
    return "Fraud" if score >= 50 else "Legitimate"


def build_model_input(amount: float, encoded_category: int) -> pd.DataFrame:
    # Keep the same feature structure that your model expects
    return pd.DataFrame({
        'cc_num': [2291163933867240],
        'merchant': [319],
        'category': [encoded_category],
        'amt': [amount],
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


def risk_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%"},
        title={"text": "Hybrid Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ef4444"},
            "steps": [
                {"range": [0, 30], "color": "#14532d"},
                {"range": [30, 70], "color": "#854d0e"},
                {"range": [70, 100], "color": "#991b1b"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb", "family": "Segoe UI"},
        margin=dict(l=20, r=20, t=50, b=10),
        height=300
    )
    return fig


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Fraud Detection System</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-sub">About</div>', unsafe_allow_html=True)
    st.write("Premium fraud monitoring dashboard built with Streamlit, machine learning, and hybrid risk scoring.")

    st.markdown('<div class="sidebar-sub">Features</div>', unsafe_allow_html=True)
    st.success("Real-time Fraud Checker")
    st.success("Hybrid Risk Scoring")
    st.success("Prediction History")
    st.success("CSV Export")
    st.success("Explainable Output")
    st.success("Risk Gauge")

    st.markdown('<div class="sidebar-sub">Model</div>', unsafe_allow_html=True)
    st.info("Random Forest + Rules")

    st.markdown('<div class="sidebar-sub">Status</div>', unsafe_allow_html=True)
    st.success("Application Running")

    if st.button("Clear Prediction History", use_container_width=True):
        st.session_state.history = []
        st.session_state.latest_result = None
        st.rerun()

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown('<div class="main-title">Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Industry-style fraud monitoring dashboard with explainable AI, hybrid scoring, and live transaction analysis.</div>',
    unsafe_allow_html=True
)

if model is None:
    st.error("model.pkl not found. Please keep model.pkl in the same folder as app.py.")
    st.stop()

# =========================================================
# METRICS
# =========================================================
history_df_for_metrics = pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame()

total_predictions = len(st.session_state.history)
fraud_count = 0
last_risk = "N/A"

if total_predictions > 0:
    fraud_count = int((history_df_for_metrics["Final Decision"] == "Fraud").sum())
    last_risk = f"{history_df_for_metrics.iloc[-1]['Hybrid Risk Score']:.2f}%"

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">System Status</div>
        <div class="metric-value">Active</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Total Predictions</div>
        <div class="metric-value">{total_predictions}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Fraud Count</div>
        <div class="metric-value">{fraud_count}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Last Risk Score</div>
        <div class="metric-value">{last_risk}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Prediction", "History & Analytics", "About"])

# =========================================================
# TAB 1 - PREDICTION
# =========================================================
with tab1:
    st.markdown('<div class="section-title">Transaction Input</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)

    category_options = [
        'personal_care', 'health_fitness', 'misc_pos', 'travel',
        'gas_transport', 'food_dining', 'shopping_net', 'groceries',
        'home', 'entertainment', 'kids_pets', 'hotels_motels',
        'shopping_mall', 'utility'
    ]

    with right:
        category = st.selectbox("Transaction Category", options=category_options)

    category_mapping = {cat: i for i, cat in enumerate(category_options)}
    encoded_category = category_mapping[category]

    st.markdown(
        '<div class="small-note">Some model features are internally prefilled because the trained model expects the full processed feature structure.</div>',
        unsafe_allow_html=True
    )

    if st.button("Run Fraud Check", use_container_width=True):
        try:
            input_df = build_model_input(amount, encoded_category)

            ml_prob = float(model.predict_proba(input_df)[0][1])
            hybrid_risk = hybrid_score(ml_prob, amount, category)
            risk_level = risk_label(hybrid_risk)
            decision = final_decision(hybrid_risk)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            row = {
                "Time": timestamp,
                "Amount": amount,
                "Category": category,
                "ML Probability": round(ml_prob, 4),
                "Rules Score": round(rule_based_score(amount, category), 2),
                "Hybrid Risk Score": round(hybrid_risk, 2),
                "Risk Level": risk_level,
                "Final Decision": decision
            }

            st.session_state.history.append(row)

            st.session_state.latest_result = {
                "time": timestamp,
                "amount": amount,
                "category": category,
                "ml_prob": ml_prob,
                "rules_score": rule_based_score(amount, category),
                "hybrid_score": hybrid_risk,
                "risk_level": risk_level,
                "decision": decision
            }

            st.rerun()

        except Exception as e:
            st.error(f"Prediction error: {e}")

    if st.session_state.latest_result:
        res = st.session_state.latest_result

        st.markdown('<div class="section-title">Prediction Output</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("ML Fraud Probability", f"{res['ml_prob']:.4f}")
        with r2:
            st.metric("Hybrid Risk Score", f"{res['hybrid_score']:.2f}%")
        with r3:
            st.metric("Risk Level", res["risk_level"])

        st.progress(min(int(res["hybrid_score"]), 100))

        if res["decision"] == "Fraud":
            st.markdown('<div class="fraud-box">Fraudulent Transaction Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">Transaction is Likely Legitimate</div>', unsafe_allow_html=True)

        g1, g2 = st.columns([1.2, 1])

        with g1:
            st.markdown('<div class="section-title">Risk Gauge Meter</div>', unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(res["hybrid_score"]), use_container_width=True)

        with g2:
            st.markdown('<div class="section-title">Audit Explanation</div>', unsafe_allow_html=True)

            st.markdown('<div class="info-panel">', unsafe_allow_html=True)
            st.write("**Amount Analysis**")
            if res["amount"] >= 5000:
                st.warning("Very high transaction amount increased the rule-based fraud score.")
            elif res["amount"] >= 1000:
                st.info("Moderately high amount contributed to medium audit concern.")
            else:
                st.success("Transaction amount is in a lower-risk range.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("")

            st.markdown('<div class="info-panel">', unsafe_allow_html=True)
            st.write("**Category Analysis**")
            if res["category"] in ["travel", "shopping_net", "misc_pos"]:
                st.warning(f"Category '{res['category']}' is treated as relatively risk-sensitive.")
            else:
                st.success(f"Category '{res['category']}' appears relatively normal in this context.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Final Audit Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-panel">', unsafe_allow_html=True)
        st.write(f"- **Timestamp:** {res['time']}")
        st.write(f"- **Transaction Amount:** {res['amount']}")
        st.write(f"- **Selected Category:** {res['category']}")
        st.write(f"- **ML Probability:** {res['ml_prob']:.4f}")
        st.write(f"- **Rules Score:** {res['rules_score']:.2f}")
        st.write(f"- **Hybrid Risk Score:** {res['hybrid_score']:.2f}%")
        st.write(f"- **Final Decision:** {res['decision']}")
        st.markdown('</div>', unsafe_allow_html=True)

        report = f"""
Fraud Detection Audit Report
============================

Timestamp: {res['time']}
Transaction Amount: {res['amount']}
Category: {res['category']}
ML Probability: {res['ml_prob']:.4f}
Rules Score: {res['rules_score']:.2f}
Hybrid Risk Score: {res['hybrid_score']:.2f}%
Risk Level: {res['risk_level']}
Final Decision: {res['decision']}
Model Stack: Random Forest + Rule Engine
"""

        st.download_button(
            "Download Result Report",
            report,
            file_name="fraud_report.txt",
            mime="text/plain",
            use_container_width=True
        )

# =========================================================
# TAB 2 - HISTORY & ANALYTICS
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No prediction history yet. Run at least one fraud check to unlock analytics.")
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

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">Risk Score Trend</div>', unsafe_allow_html=True)
            trend_df = hist_df[["Time", "Hybrid Risk Score"]].copy().set_index("Time")
            st.line_chart(trend_df)

        with c2:
            st.markdown('<div class="section-title">Fraud vs Legit</div>', unsafe_allow_html=True)
            decision_counts = hist_df["Final Decision"].value_counts()
            st.bar_chart(decision_counts)

        st.markdown('<div class="section-title">Category-wise Risk</div>', unsafe_allow_html=True)
        category_risk = hist_df.groupby("Category")["Hybrid Risk Score"].mean()
        st.bar_chart(category_risk)

# =========================================================
# TAB 3 - ABOUT
# =========================================================
with tab3:
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-panel">', unsafe_allow_html=True)
    st.write("""
This dashboard is designed as a premium fraud-monitoring interface inspired by modern fintech software.

### Core Capabilities
- Random Forest model-based fraud prediction
- Rule-based fraud scoring
- Hybrid final risk score
- Real-time monitoring UI
- Prediction history and analytics
- Exportable audit reports

### Risk Logic
The system combines:
1. **Machine learning probability**
2. **Rule engine score**
3. **Final weighted hybrid score**

### Technology Stack
- **Frontend:** Streamlit
- **Charts:** Plotly
- **Modeling:** Scikit-learn
- **Deployment:** GitHub + Streamlit Cloud
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit, Plotly, and Scikit-learn.")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AI Deep Audit Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #1e1b4b);
    color: white;
}
.block-container {
    padding: 2rem 3rem;
}
.card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.title {
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #94a3b8;
    font-size: 15px;
}
.good { color: #34d399; font-weight: 700; }
.warn { color: #fbbf24; font-weight: 700; }
.bad { color: #f87171; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_bundle():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
    model = bundle["pipeline"]
    model_name = bundle.get("model_name", "ML Model")
    input_columns = bundle.get("input_columns", None)
    model_threshold = bundle.get("threshold", 0.5)
except Exception as e:
    st.error("❌ Model file not found. Put `fraud_model_bundle.pkl` in same folder as app.py")
    st.stop()

# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

if "audit_results" not in st.session_state:
    st.session_state.audit_results = pd.DataFrame()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def risk_level(score):
    if score >= 75:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"

def final_decision(score, review_zone=True):
    if review_zone and 45 <= score < 60:
        return "Review"
    elif score >= 60:
        return "Fraud"
    return "Legitimate"

def rule_score(row):
    score = 0

    amt = row.get("amt", row.get("amount", 0))
    try:
        amt = float(amt)
    except:
        amt = 0

    if amt >= 10000:
        score += 35
    elif amt >= 5000:
        score += 25
    elif amt >= 1000:
        score += 12

    hour = row.get("trans_hour", row.get("hour", None))
    if hour is not None:
        try:
            hour = int(hour)
            if hour <= 4 or hour >= 23:
                score += 20
        except:
            pass

    distance = row.get("distance_km", 0)
    try:
        distance = float(distance)
        if distance >= 100:
            score += 20
    except:
        pass

    category = str(row.get("category", "")).lower()
    risky_categories = ["shopping_net", "travel", "misc_pos", "entertainment"]
    if category in risky_categories:
        score += 15

    return min(score, 100)

def explain_transaction(row, ml_prob, anomaly_score, rules):
    reasons = []

    amt = row.get("amt", row.get("amount", 0))
    try:
        amt = float(amt)
        if amt >= 10000:
            reasons.append("Very high transaction amount")
        elif amt >= 5000:
            reasons.append("High transaction amount")
    except:
        pass

    hour = row.get("trans_hour", row.get("hour", None))
    try:
        hour = int(hour)
        if hour <= 4 or hour >= 23:
            reasons.append("Transaction happened at unusual time")
    except:
        pass

    distance = row.get("distance_km", 0)
    try:
        distance = float(distance)
        if distance >= 100:
            reasons.append("Large distance between user and merchant")
    except:
        pass

    category = str(row.get("category", "")).lower()
    if category in ["shopping_net", "travel", "misc_pos", "entertainment"]:
        reasons.append("Category is historically risky")

    if ml_prob >= 0.7:
        reasons.append("ML model predicted high fraud probability")

    if anomaly_score >= 70:
        reasons.append("Transaction looks anomalous compared to others")

    if rules >= 60:
        reasons.append("Multiple rule-based risk checks triggered")

    if not reasons:
        reasons.append("No strong suspicious pattern detected")

    return "; ".join(reasons)

def prepare_input(df):
    df = df.copy()

    if input_columns is not None:
        for col in input_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[input_columns]

    return df

def run_deep_audit(df):
    original_df = df.copy()
    model_df = prepare_input(df)

    ml_prob = model.predict_proba(model_df)[:, 1]

    numeric_df = model_df.select_dtypes(include=[np.number]).fillna(0)

    if len(numeric_df) >= 20:
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomaly_raw = iso.fit_predict(numeric_df)
        anomaly_score = np.where(anomaly_raw == -1, 80, 20)
    else:
        anomaly_score = np.array([20] * len(model_df))

    rules = original_df.apply(rule_score, axis=1).values

    final_score = (
        (ml_prob * 100 * 0.55) +
        (anomaly_score * 0.20) +
        (rules * 0.25)
    )

    final_score = np.clip(final_score, 0, 100)

    result_df = original_df.copy()
    result_df["ML Probability"] = np.round(ml_prob, 4)
    result_df["Anomaly Score"] = np.round(anomaly_score, 2)
    result_df["Rule Score"] = np.round(rules, 2)
    result_df["Final Risk Score"] = np.round(final_score, 2)
    result_df["Risk Level"] = result_df["Final Risk Score"].apply(risk_level)
    result_df["Final Decision"] = result_df["Final Risk Score"].apply(final_decision)
    result_df["Explanation"] = [
        explain_transaction(original_df.iloc[i], ml_prob[i], anomaly_score[i], rules[i])
        for i in range(len(original_df))
    ]
    result_df["Audit Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return result_df

def create_gauge(score):
    color = "#10b981" if score < 40 else "#f59e0b" if score < 75 else "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%"},
        title={"text": "Final Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "rgba(16,185,129,0.25)"},
                {"range": [40, 75], "color": "rgba(245,158,11,0.25)"},
                {"range": [75, 100], "color": "rgba(239,68,68,0.25)"}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=300
    )
    return fig

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="card">
    <div class="title">🛡️ AI Deep Audit Platform</div>
    <div class="subtitle">
        Fraud Detection with Machine Learning + Risk Scoring + Explainability + False Positive Reduction
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("🛡️ Audit Control")

    st.success("Model Loaded")
    st.write("Model:", model_name)

    st.divider()

    st.subheader("Risk Settings")
    ml_weight = st.slider("ML Weight", 0.0, 1.0, 0.55, 0.05)
    anomaly_weight = st.slider("Anomaly Weight", 0.0, 1.0, 0.20, 0.05)
    rule_weight = round(1 - ml_weight - anomaly_weight, 2)

    if rule_weight < 0:
        st.error("Weights total cannot exceed 1")
        st.stop()

    st.info(f"Rule Weight: {rule_weight}")

    st.divider()

    if st.button("Clear Audit History"):
        st.session_state.history = pd.DataFrame()
        st.session_state.audit_results = pd.DataFrame()
        st.rerun()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Data Input",
    "🤖 Fraud Engine",
    "📊 Dashboard",
    "⚠️ Alerts Panel",
    "🧠 Explainability"
])

# -----------------------------
# TAB 1: DATA INPUT
# -----------------------------
with tab1:
    st.subheader("📥 Data Input Module")

    option = st.radio(
        "Choose input method",
        ["Upload CSV Dataset", "Manual Transaction Entry"],
        horizontal=True
    )

    if option == "Upload CSV Dataset":
        uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = df
            st.success("Dataset uploaded successfully")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Run Deep Audit on Dataset"):
                result_df = run_deep_audit(df)
                st.session_state.audit_results = result_df
                st.session_state.history = pd.concat(
                    [st.session_state.history, result_df],
                    ignore_index=True
                )
                st.success("Deep audit completed successfully")

    else:
        st.info("Enter important transaction details manually.")

        c1, c2, c3 = st.columns(3)

        with c1:
            amt = st.number_input("Amount", min_value=0.0, value=1500.0)
            category = st.selectbox(
                "Category",
                ["shopping_net", "travel", "misc_pos", "gas_transport", "food_dining", "entertainment", "grocery_pos"]
            )

        with c2:
            trans_hour = st.slider("Transaction Hour", 0, 23, 14)
            distance_km = st.number_input("Distance KM", min_value=0.0, value=10.0)

        with c3:
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
            city_pop = st.number_input("City Population", min_value=0, value=50000)

        manual_data = {
            "amt": amt,
            "category": category,
            "trans_hour": trans_hour,
            "distance_km": distance_km,
            "age": age,
            "city_pop": city_pop
        }

        manual_df = pd.DataFrame([manual_data])

        st.dataframe(manual_df, use_container_width=True)

        if st.button("Analyze Manual Transaction"):
            result_df = run_deep_audit(manual_df)
            st.session_state.audit_results = result_df
            st.session_state.history = pd.concat(
                [st.session_state.history, result_df],
                ignore_index=True
            )
            st.success("Manual transaction analyzed successfully")

# -----------------------------
# TAB 2: FRAUD ENGINE
# -----------------------------
with tab2:
    st.subheader("🤖 Fraud Detection Engine")

    if st.session_state.audit_results.empty:
        st.info("Run an audit first from Data Input tab.")
    else:
        latest = st.session_state.audit_results.iloc[-1]

        c1, c2 = st.columns([1, 1])

        with c1:
            st.plotly_chart(create_gauge(latest["Final Risk Score"]), use_container_width=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("ML Probability", f"{latest['ML Probability'] * 100:.2f}%")
            st.metric("Anomaly Score", f"{latest['Anomaly Score']:.2f}%")
            st.metric("Rule Score", f"{latest['Rule Score']:.2f}%")
            st.metric("Final Decision", latest["Final Decision"])
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Latest Audit Result")
        st.dataframe(st.session_state.audit_results.tail(10), use_container_width=True)

# -----------------------------
# TAB 3: DASHBOARD
# -----------------------------
with tab3:
    st.subheader("📊 Audit Dashboard")

    df = st.session_state.history

    if df.empty:
        st.info("No transactions audited yet.")
    else:
        total = len(df)
        fraud = (df["Final Decision"] == "Fraud").sum()
        review = (df["Final Decision"] == "Review").sum()
        legit = (df["Final Decision"] == "Legitimate").sum()
        high_risk = (df["Risk Level"] == "High").sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Transactions", total)
        c2.metric("Fraud Detected", fraud)
        c3.metric("Review Cases", review)
        c4.metric("High-Risk Alerts", high_risk)

        c5, c6 = st.columns(2)

        with c5:
            fig = px.bar(
                df["Final Decision"].value_counts().reset_index(),
                x="Final Decision",
                y="count",
                title="Fraud vs Normal",
                color="Final Decision"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c6:
            fig = px.histogram(
                df,
                x="Final Risk Score",
                nbins=25,
                title="Risk Score Distribution",
                color="Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "Audit Time" in df.columns:
            time_df = df.copy()
            time_df["Audit Time"] = pd.to_datetime(time_df["Audit Time"])
            fig = px.line(
                time_df,
                x="Audit Time",
                y="Final Risk Score",
                color="Final Decision",
                title="Transactions Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 4: ALERTS PANEL
# -----------------------------
with tab4:
    st.subheader("⚠️ Alerts Panel")

    df = st.session_state.history

    if df.empty:
        st.info("No alerts yet.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            filter_level = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])

        with c2:
            min_amount = st.number_input("Minimum Amount", min_value=0.0, value=0.0)

        with c3:
            decision_filter = st.selectbox("Filter by Decision", ["All", "Fraud", "Review", "Legitimate"])

        alert_df = df.copy()

        if filter_level != "All":
            alert_df = alert_df[alert_df["Risk Level"] == filter_level]

        if "amt" in alert_df.columns:
            alert_df = alert_df[alert_df["amt"] >= min_amount]

        if decision_filter != "All":
            alert_df = alert_df[alert_df["Final Decision"] == decision_filter]

        high_alerts = alert_df[
            (alert_df["Final Decision"].isin(["Fraud", "Review"])) |
            (alert_df["Risk Level"] == "High")
        ]

        st.warning(f"{len(high_alerts)} alert transactions found")

        st.dataframe(high_alerts, use_container_width=True)

        csv = high_alerts.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Alerts CSV",
            csv,
            "fraud_alerts.csv",
            "text/csv"
        )

# -----------------------------
# TAB 5: EXPLAINABILITY
# -----------------------------
with tab5:
    st.subheader("🧠 Explainability Section")

    df = st.session_state.audit_results

    if df.empty:
        st.info("Run audit first.")
    else:
        selected_index = st.selectbox(
            "Select transaction",
            options=list(df.index),
            format_func=lambda x: f"Transaction {x}"
        )

        row = df.loc[selected_index]

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.write("### Why was this transaction flagged?")
        st.write(row["Explanation"])

        st.write("### Score Breakdown")
        breakdown = pd.DataFrame({
            "Component": ["ML Probability", "Anomaly Detection", "Rule-Based Checks"],
            "Score": [
                row["ML Probability"] * 100,
                row["Anomaly Score"],
                row["Rule Score"]
            ]
        })

        fig = px.bar(
            breakdown,
            x="Component",
            y="Score",
            color="Component",
            title="Explainable Risk Contribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("### False Positive Reduction")
        if row["Final Decision"] == "Review":
            st.warning("This transaction is in review zone. It is not directly marked fraud to reduce false positives.")
        elif row["Final Decision"] == "Fraud" and row["ML Probability"] < 0.6:
            st.warning("Fraud decision mainly came from rules/anomaly. Manual auditor review recommended.")
        elif row["Final Decision"] == "Legitimate":
            st.success("Low-risk transaction. No strong fraud signal found.")
        else:
            st.error("Strong fraud signal detected from multiple components.")

        st.markdown('</div>', unsafe_allow_html=True)

        report = row.to_string()
        st.download_button(
            "Download Explanation Report",
            report,
            "transaction_explanation.txt",
            "text/plain"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.write("")
st.caption("AI Deep Audit Platform • ML + Anomaly Detection + Rule Engine + Explainability")

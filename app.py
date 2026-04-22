import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #1e293b;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
.metric-label {
    color: #94a3b8;
    font-size: 15px;
}
.metric-value {
    color: white;
    font-size: 26px;
    font-weight: 700;
}
.safe-box {
    background: #052e16;
    color: #bbf7d0;
    padding: 16px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
}
.fraud-box {
    background: #450a0a;
    color: #fecaca;
    padding: 16px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
}
.info-box {
    background: #0f172a;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 14px;
}
.small-text {
    color: #94a3b8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# MODEL LOAD
# -------------------------------
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("### About Project")
    st.write(
        "This app predicts whether a transaction is fraudulent using a trained Random Forest model."
    )

    st.markdown("### Features")
    st.write("✅ Real-time Fraud Prediction")
    st.write("✅ Risk Score")
    st.write("✅ Prediction History")
    st.write("✅ CSV Download")
    st.write("✅ Audit Summary")

    st.markdown("---")
    st.markdown("### Model")
    st.info("Random Forest Classifier")

    st.markdown("### Status")
    st.success("App deployed successfully")

    if st.button("🗑 Clear Prediction History", use_container_width=True):
        st.session_state.history = []
        st.success("History cleared")

# -------------------------------
# MAIN TITLE
# -------------------------------
st.markdown('<div class="main-title">💳 Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An AI-based audit interface for detecting suspicious financial transactions with dynamic risk scoring.</div>',
    unsafe_allow_html=True
)

if model is None:
    st.error("model.pkl not found. Please upload model.pkl in the same folder as app.py.")
    st.stop()

# -------------------------------
# TOP METRIC CARDS
# -------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">System Status</div>
        <div class="metric-value">Active</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model Used</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predictions Made</div>
        <div class="metric-value">{len(st.session_state.history)}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    last_risk = (
        f"{st.session_state.history[-1]['Risk Score (%)']:.2f}%"
        if len(st.session_state.history) > 0 else "N/A"
    )
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
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📜 History & Charts", "📘 About"])

# ===============================
# TAB 1 - PREDICTION
# ===============================
with tab1:
    st.subheader("Transaction Input")

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
        '<div class="small-text">Note: Some features are internally prefilled for demonstration because the trained model expects the complete training feature structure.</div>',
        unsafe_allow_html=True
    )

    st.markdown("")

    if st.button("🚨 Predict Fraud", use_container_width=True):
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

            # Save history
            st.session_state.history.append({
                "Time": timestamp,
                "Amount": amt,
                "Category": category_input,
                "Prediction": "Fraud" if prediction == 1 else "Legitimate",
                "Fraud Probability": round(float(probability), 4),
                "Risk Score (%)": round(risk_score, 2),
                "Risk Level": risk_level
            })

            st.markdown("## 📊 Prediction Output")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Fraud Probability", f"{probability:.4f}")
            with r2:
                st.metric("Risk Score", f"{risk_score:.2f}%")
            with r3:
                st.metric("Risk Level", risk_level)

            st.progress(min(int(risk_score), 100))

            st.markdown("")
            if prediction == 1:
                st.markdown(
                    '<div class="fraud-box">⚠ Fraudulent Transaction Detected</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="safe-box">✅ Transaction is Likely Legitimate</div>',
                    unsafe_allow_html=True
                )

            st.markdown("### 🧠 Audit Explanation")
            exp1, exp2 = st.columns(2)

            with exp1:
                with st.container():
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.write("**Amount Analysis**")
                    if amt > 5000:
                        st.warning("High transaction amount may increase fraud suspicion.")
                    elif amt > 1000:
                        st.info("Moderately high amount. This may need additional audit attention.")
                    else:
                        st.success("Amount appears in a lower-risk range.")
                    st.markdown('</div>', unsafe_allow_html=True)

            with exp2:
                with st.container():
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.write("**Category Analysis**")
                    if category_input in ["travel", "shopping_net", "misc_pos"]:
                        st.warning(f"Category '{category_input}' may show relatively riskier patterns in demo analysis.")
                    else:
                        st.success(f"Category '{category_input}' appears normal in this demo context.")
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 📌 Final Audit Summary")
            st.write(f"- **Timestamp:** {timestamp}")
            st.write(f"- **Transaction Amount:** {amt}")
            st.write(f"- **Selected Category:** {category_input}")
            st.write(f"- **Prediction:** {'Fraud' if prediction == 1 else 'Legitimate'}")
            st.write(f"- **Risk Score:** {risk_score:.2f}%")
            st.write(f"- **Model Used:** Random Forest")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ===============================
# TAB 2 - HISTORY & CHARTS
# ===============================
with tab2:
    st.subheader("Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No prediction history yet. Make at least one prediction to see charts and tables.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        st.dataframe(hist_df, use_container_width=True)

        st.markdown("### 📥 Download History")
        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction History as CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("### 📈 Risk Score Trend")
        trend_df = hist_df[["Time", "Risk Score (%)"]].copy()
        trend_df = trend_df.set_index("Time")
        st.line_chart(trend_df)

        st.markdown("### 📊 Category-wise Risk Score")
        cat_chart = hist_df.groupby("Category", as_index=True)["Risk Score (%)"].mean()
        st.bar_chart(cat_chart)

        st.markdown("### 📌 Prediction Distribution")
        pred_count = hist_df["Prediction"].value_counts()
        st.bar_chart(pred_count)

# ===============================
# TAB 3 - ABOUT
# ===============================
with tab3:
    st.subheader("Project Overview")
    st.write("""
This fraud detection dashboard is built using **Streamlit** and a **Random Forest Classifier**.
It predicts whether a transaction is suspicious and assigns a **dynamic risk score**.

### Main Components
- Data preprocessing and model training in Python
- Fraud classification using machine learning
- Dynamic fraud probability and risk score
- Real-time deployed UI using Streamlit
- GitHub-based deployment

### Current Demo Limitation
This demo uses a simplified input form while internally pre-filling the remaining model features.
For a more advanced production version, all training features and preprocessing encoders should be fully integrated into the UI.

### Deployment Stack
- **Frontend/UI:** Streamlit
- **Backend Logic:** Python
- **Model:** Scikit-learn Random Forest
- **Deployment:** GitHub + Streamlit Cloud
    """)

st.markdown("---")
st.caption("Built with Streamlit and deployed using GitHub + Streamlit Cloud.")

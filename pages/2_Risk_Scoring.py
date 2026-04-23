import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

model = pickle.load(open("model.pkl", "rb"))

st.title("Risk Scoring")

amt = st.number_input("Amount", value=100.0)

category = st.selectbox("Category", [
    "travel", "shopping_net", "misc_pos", "food_dining"
])

if st.button("Check Risk"):

    input_data = pd.DataFrame({
        'cc_num':[1],'merchant':[1],'category':[1],
        'amt':[amt],'gender':[1],'street':[1],
        'city':[1],'state':[1],'zip':[1],
        'lat':[1],'long':[1],'city_pop':[1],
        'job':[1],'unix_time':[1],
        'merch_lat':[1],'merch_long':[1],
        'trans_date_trans_time_year':[2020],
        'trans_date_trans_time_month':[1],
        'trans_date_trans_time_day':[1],
        'dob_year':[1990],'dob_month':[1],'dob_day':[1]
    })

    prob = model.predict_proba(input_data)[0][1]
    risk = prob * 100

    prediction = "Fraud" if risk > 50 else "Legit"

    st.metric("Risk Score", f"{risk:.2f}%")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        gauge={'axis': {'range': [0,100]}}
    ))
    st.plotly_chart(fig)

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Risk Score": risk,
        "Prediction": prediction
    })

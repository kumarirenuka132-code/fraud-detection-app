import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Fraud Detection App", layout="wide")

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

st.title("💳 Fraud Detection Application")
st.write("Enter transaction details to predict whether a transaction is fraudulent.")

if model is None:
    st.error("model.pkl not found. Please upload/save model.pkl in the same folder.")
else:
    st.header("Transaction Input")

    # User inputs
    amt = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=1.0)

    category_options = [
        'personal_care', 'health_fitness', 'misc_pos', 'travel',
        'gas_transport', 'food_dining', 'shopping_net', 'groceries',
        'home', 'entertainment', 'kids_pets', 'hotels_motels',
        'shopping_mall', 'utility'
    ]
    category_input = st.selectbox("Category", options=category_options)

    # Manual mapping for category
    category_mapping = {cat: i for i, cat in enumerate(category_options)}
    encoded_category = category_mapping[category_input]

    st.subheader("Prediction")

    if st.button("Predict Fraud"):
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
            risk_score = probability * 100

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error(f"⚠ Fraudulent Transaction Detected")
            else:
                st.success("✅ Transaction is Likely Legitimate")

            st.write(f"**Fraud Probability:** {probability:.4f}")
            st.write(f"**Risk Score:** {risk_score:.2f}%")
            st.write(f"**Raw Prediction:** {prediction}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
import streamlit as st
import pandas as pd

st.title("Prediction History")

if "history" not in st.session_state:
    st.session_state.history = []

if len(st.session_state.history) == 0:
    st.info("No predictions yet")
else:
    df = pd.DataFrame(st.session_state.history)

    st.dataframe(df)

    csv = df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        "history.csv"
    )

import streamlit as st
import pandas as pd
import joblib

# Load the XGBoost model
model = joblib.load("xgboost_ptb_pipeline.pkl")

# App title
st.title("PTB Score Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file")

# If a file is uploaded, make prediction
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display uploaded data
    st.subheader("Uploaded Data")
    st.write(df)

    # Make predictions
    predictions = model.predict(df)

    # Show predictions
    st.subheader("Predicted PTB Scores")
    st.write(predictions)

    # Optionally, download predictions
    if st.button("Download Predictions"):
        output_df = df.copy()
        output_df["PTB_Score"] = predictions
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "ptb_predictions.csv", "text/csv")

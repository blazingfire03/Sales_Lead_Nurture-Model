import streamlit as st
import pandas as pd
import joblib

# === Load Trained Model ===
@st.cache_resource
def load_model():
    return joblib.load("xgboost_ptb_pipeline.pkl")

model = load_model()

# === Streamlit UI ===
st.title("🎯 PTB Score Predictor")
st.write("Upload a CSV file containing customer data to score and classify leads.")

uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        # Define features expected by the model
        kept_features = [
            'Age', 'Gender', 'Annual Income', 'Income Bracket', 'Marital Status',
            'Employment Status', 'Region', 'Urban/Rural Flag', 'State', 'ZIP Code',
            'Plan Preference Type', 'Web Form Completion Rate', 'Quote Requested',
            'Application Started', 'Behavior Score', 'Application Submitted', 'Application Applied'
        ]

        df = df[df.columns.intersection(kept_features)]

        # Run prediction
        predictions = model.predict(df)
        df['PTB_Score'] = predictions

        # Tier assignment
        def tier(score):
            if score >= 0.8:
                return "Platinum"
            elif score >= 0.6:
                return "Gold"
            elif score >= 0.4:
                return "Silver"
            else:
                return "Bronze"

        df['Lead_Tier'] = df['PTB_Score'].apply(tier)

        st.subheader("✅ Prediction Results")
        st.dataframe(df[['PTB_Score', 'Lead_Tier']].join(df.drop(columns=['PTB_Score', 'Lead_Tier'])))

        # Allow download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Scored Data", csv, "scored_leads.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("Please upload a valid CSV file to continue.")

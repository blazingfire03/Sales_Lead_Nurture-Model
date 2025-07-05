import streamlit as st
import pandas as pd
import joblib
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError

# === Load Model ===
@st.cache_resource
def load_model():
    return joblib.load("xgboost_ptb_pipeline.pkl")

model = load_model()

# === Load Azure Cosmos DB Data ===
@st.cache_data
def fetch_data():
    endpoint = st.secrets["COSMOS_ENDPOINT"]
    key = st.secrets["COSMOS_KEY"]
    database_name = st.secrets["DATABASE_NAME"]
    container_name = st.secrets["INPUT_CONTAINER"]

    try:
        client = CosmosClient(endpoint, credential=key)
        db = client.get_database_client(database_name)
        container = db.get_container_client(container_name)
        items = list(container.read_all_items())
        return pd.DataFrame(items)
    except CosmosHttpResponseError as e:
        st.error(f"Cosmos DB error: {e}")
        return pd.DataFrame()

# === Run App ===
st.title("🚀 PTB Score Predictor (Azure Auto-Fetch)")

with st.spinner("Fetching data from Azure Cosmos DB..."):
    df = fetch_data()

if df.empty:
    st.warning("No customer data found in Cosmos DB.")
else:
    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    kept_features = [
        'Age', 'Gender', 'Annual Income', 'Income Bracket', 'Marital Status',
        'Employment Status', 'Region', 'Urban/Rural Flag', 'State', 'ZIP Code',
        'Plan Preference Type', 'Web Form Completion Rate', 'Quote Requested',
        'Application Started', 'Behavior Score', 'Application Submitted', 'Application Applied'
    ]

    df = df[df.columns.intersection(kept_features)]

    try:
        predictions = model.predict(df)
        df['PTB_Score'] = predictions

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

        st.subheader("✅ Scored Data")
        st.dataframe(df[['PTB_Score', 'Lead_Tier']].join(df.drop(columns=['PTB_Score', 'Lead_Tier'])))

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", csv, "scored_leads.csv", "text/csv")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

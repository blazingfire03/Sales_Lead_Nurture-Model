import streamlit as st
import pandas as pd
import joblib
import uuid
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# === Load Model ===
@st.cache_resource
def load_model():
    return joblib.load("xgboost-api/xgb_clean.pkl")  # MUST be a clean model without local classes

model = load_model()

# === Load Data from Cosmos DB ===
@st.cache_data
def fetch_data():
    try:
        endpoint = st.secrets["COSMOS_ENDPOINT"]
        key = st.secrets["COSMOS_KEY"]
        database_name = st.secrets["DATABASE_NAME"]
        container_name = st.secrets["INPUT_CONTAINER"]

        client = CosmosClient(endpoint, credential=key)
        db = client.get_database_client(database_name)
        container = db.get_container_client(container_name)
        items = list(container.read_all_items())
        return pd.DataFrame(items)

    except Exception as e:
        st.error(f"❌ Failed to fetch data from Cosmos DB: {e}")
        return pd.DataFrame()

# === Upload Scored Data to Cosmos DB ===
def upload_results(df):
    try:
        endpoint = st.secrets["COSMOS_ENDPOINT"]
        key = st.secrets["COSMOS_KEY"]
        database_name = st.secrets["DATABASE_NAME"]
        output_container = st.secrets["OUTPUT_CONTAINER"]

        client = CosmosClient(endpoint, credential=key)
        db = client.get_database_client(database_name)

        # Create output container if it doesn't exist
        container = db.create_container_if_not_exists(
            id=output_container,
            partition_key=PartitionKey(path="/id"),
            offer_throughput=400
        )

        for _, row in df.iterrows():
            record = row.to_dict()
            record["id"] = str(uuid.uuid4())
            container.upsert_item(record)

        st.success(f"✅ Uploaded {len(df)} scored leads to Cosmos DB → '{output_container}'.")

    except Exception as e:
        st.error(f"❌ Failed to upload to Cosmos DB: {e}")

# === App UI ===
st.title("🧠 PTB Score Predictor with Azure Auto Sync")

with st.spinner("Fetching customer data from Cosmos DB..."):
    df = fetch_data()

if df.empty:
    st.warning("⚠️ No data found in Azure Cosmos DB.")
else:
    st.subheader("📄 Input Data (Preview)")
    st.dataframe(df.head())

    # Define required features
    required_features = [
        'Age', 'Gender', 'Annual Income', 'Income Bracket', 'Marital Status',
        'Employment Status', 'Region', 'Urban/Rural Flag', 'State', 'ZIP Code',
        'Plan Preference Type', 'Web Form Completion Rate', 'Quote Requested',
        'Application Started', 'Behavior Score', 'Application Submitted', 'Application Applied'
    ]

    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        st.error(f"❌ Missing required features in data: {missing_features}")
    else:
        try:
            input_df = df[required_features]
            predictions = model.predict(input_df)
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

            st.subheader("✅ Scored Results")
            st.dataframe(df[['PTB_Score', 'Lead_Tier']].join(df.drop(columns=['PTB_Score', 'Lead_Tier'])))

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "scored_leads.csv", "text/csv")

            # Upload option
            if st.button("🚀 Upload to Cosmos DB"):
                upload_results(df)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

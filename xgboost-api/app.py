import streamlit as st
import pandas as pd
import joblib
import uuid
import os
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_ptb_pipeline.pkl")
    return joblib.load(model_path)

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

    # Required columns
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
            proba = model.predict_proba(input_df)[:, 1]
            df['PTB_Score'] = proba * 100  # convert to percentage

            def tier(score):
                if score >= 90:
                    return "Platinum"
                elif score >= 75:
                    return "Gold"
                elif score >= 50:
                    return "Silver"
                else:
                    return "Bronze"

            df['Lead_Tier'] = df['PTB_Score'].apply(tier)

            st.subheader("✅ Scored Results")
            st.dataframe(df[['PTB_Score', 'Lead_Tier']].join(df.drop(columns=['PTB_Score', 'Lead_Tier'])))

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "scored_leads.csv", "text/csv")

            if st.button("🚀 Upload to Cosmos DB"):
                upload_results(df)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

import streamlit as st
import pandas as pd
from azure.cosmos import CosmosClient

# Load secrets
endpoint = st.secrets["COSMOS_ENDPOINT"]
key = st.secrets["COSMOS_KEY"]
database_name = st.secrets["DATABASE_NAME"]
container_name = st.secrets["INPUT_CONTAINER"]

# Cache the Cosmos DB query
@st.cache_data
def fetch_data():
    client = CosmosClient(endpoint, credential=key)
    db = client.get_database_client(database_name)
    container = db.get_container_client(container_name)
    items = list(container.read_all_items())
    return pd.DataFrame(items)

# Load data
st.title("📊 Sales Lead Nurture Dashboard")
df = fetch_data()

if df.empty:
    st.warning("No data found in Cosmos DB.")
else:
    # KPI Metrics
    st.subheader("🔑 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Leads", len(df))
    col2.metric("Policies Purchased", df['Policy Purchased'].sum())
    col3.metric("Conversion Rate (%)", round(100 * df['Policy Purchased'].sum() / len(df), 2))

    # Filter
    st.sidebar.header("🔍 Filter Leads")
    selected_state = st.sidebar.selectbox("Select State", ["All"] + sorted(df['State'].dropna().unique().tolist()))
    if selected_state != "All":
        df = df[df["State"] == selected_state]

    # Charts
    st.subheader("🗺️ Leads by State")
    st.bar_chart(df["State"].value_counts())

    st.subheader("📈 Application Funnel")
    funnel_cols = ['Application Started', 'Application Submitted', 'Policy Purchased']
    funnel_data = df[funnel_cols].sum()
    st.line_chart(funnel_data)

    st.subheader("📂 Full Data Table")
    st.dataframe(df)

    st.download_button("Download CSV", df.to_csv(index=False), "leads_data.csv")


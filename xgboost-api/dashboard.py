import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from azure.cosmos import CosmosClient

# === Page Config ===
st.set_page_config(page_title="PTB Score Dashboard", layout="wide")
st.title("🧠 PTB Score Predictor with Azure Auto Sync")

# === Load Secrets from Streamlit Cloud ===
endpoint = st.secrets["COSMOS_ENDPOINT"]
key = st.secrets["COSMOS_KEY"]
database_name = st.secrets["DATABASE_NAME"]
container_name = st.secrets["INPUT_CONTAINER"]

# === Fetch Data from Cosmos DB ===
@st.cache_data
def fetch_data():
    try:
        client = CosmosClient(endpoint, credential=key)
        db = client.get_database_client(database_name)
        container = db.get_container_client(container_name)
        items = list(container.read_all_items())
        return pd.DataFrame(items)
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        return pd.DataFrame()

# === Upload Data to Cosmos DB ===
def upload_to_cosmos(df):
    try:
        client = CosmosClient(endpoint, credential=key)
        db = client.get_database_client(database_name)
        container = db.get_container_client(container_name)
        for row in df.to_dict(orient="records"):
            container.upsert_item(row)
        st.success("✅ Data uploaded to Cosmos DB")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

# === Load and Display Data ===
df = fetch_data()

if df.empty:
    st.warning("No data found in Cosmos DB.")
else:
    # === Preview Input Data ===
    st.subheader("📋 Input Data (Preview)")
    preview_cols = [col for col in df.columns if col not in ['PTB_Score', 'Lead_Tier']]
    st.dataframe(df[preview_cols].head())

    # === Scored Results Table ===
    if "PTB_Score" in df.columns and "Lead_Tier" in df.columns:
        st.subheader("✅ Scored Results")
        st.dataframe(df)

        # === Chart 1: PTB Score Distribution ===
        st.subheader("📊 PTB Score Distribution")
        fig1, ax1 = plt.subplots()
        df["PTB_Score"].value_counts().sort_index().plot(kind='bar', ax=ax1, color='orange')
        ax1.set_xlabel("PTB Score")
        ax1.set_ylabel("Number of Customers")
        st.pyplot(fig1)

        # === Chart 2: Policy Purchase Outcomes ===
        if "Policy Purchased" in df.columns:
            st.subheader("✅ Policy Purchase Outcomes")
            fig2, ax2 = plt.subplots()
            df["Policy Purchased"].value_counts().rename(index={0: "Not Purchased", 1: "Purchased"}).plot(kind='bar', ax=ax2, color='orange')
            ax2.set_ylabel("Number of Customers")
            st.pyplot(fig2)

        # === Chart 3: Lead Tier Distribution ===
        st.subheader("🏅 Customers by Lead Tier")
        fig3, ax3 = plt.subplots()
        df["Lead_Tier"].value_counts().plot(kind='bar', ax=ax3, color='orange')
        ax3.set_xlabel("Lead Tier")
        ax3.set_ylabel("Number of Customers")
        st.pyplot(fig3)

        # === Download and Upload Buttons ===
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), "scored_results.csv")
        if st.button("🚀 Upload to Cosmos DB"):
            upload_to_cosmos(df)
    else:
        st.warning("⚠️ Required columns 'PTB_Score' or 'Lead_Tier' not found.")

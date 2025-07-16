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
import plotly.express as px
from azure.cosmos import CosmosClient

st.set_page_config(page_title="Lead Conversion Dashboard", layout="wide")

# === Load data from Cosmos DB ===
@st.cache_data(ttl=60)
def load_data():
    endpoint = st.secrets["COSMOS_ENDPOINT"]
    key = st.secrets["COSMOS_KEY"]
    database_name = st.secrets["DATABASE_NAME"]
    container_name = st.secrets["OUTPUT_CONTAINER"]

    client = CosmosClient(endpoint, credential=key)
    db = client.get_database_client(database_name)
    container = db.get_container_client(container_name)

    items = list(container.read_all_items())
    df = pd.DataFrame(items)

    return df

df = load_data()

# === Metrics ===
total_leads = len(df)
policies_purchased = df["Policy Purchased"].sum()
conversion_rate = (policies_purchased / total_leads) * 100

st.title("📊 Lead Conversion Dashboard")
st.markdown("Live dashboard powered by Cosmos DB. Auto-refreshes every 60 seconds.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Leads", f"{total_leads}")
col2.metric("Policies Purchased", f"{int(policies_purchased)}")
col3.metric("Conversion Rate", f"{conversion_rate:.2f}%")

st.divider()

# === 1. Lead Tier vs State ===
st.subheader("1️⃣ Lead Tier Distribution by State")
states = st.multiselect("Filter by State:", options=df["State"].unique())
df1 = df[df["State"].isin(states)] if states else df
fig1 = px.histogram(df1, x="State", color="Lead_Tier", barmode="group", title="Lead Tier by State")
st.plotly_chart(fig1, use_container_width=True)

# === 2. Lead Tier vs Income Bracket ===
st.subheader("2️⃣ Lead Tier vs Income Bracket")
fig2 = px.histogram(df, x="Income Bracket", color="Lead_Tier", barmode="stack", title="Lead Tier by Income Bracket")
st.plotly_chart(fig2, use_container_width=True)

# === 3. Lead Tier vs Age Group ===
st.subheader("3️⃣ Lead Tier vs Age Group")
age_groups = st.multiselect("Filter by Age Group:", options=df["Age Group"].unique())
df3 = df[df["Age Group"].isin(age_groups)] if age_groups else df
fig3 = px.histogram(df3, x="Age Group", color="Lead_Tier", barmode="group", title="Lead Tier by Age Group")
st.plotly_chart(fig3, use_container_width=True)

# === 4. Lead Tier vs Gender (filtered by Employment Status) ===
st.subheader("4️⃣ Lead Tier vs Gender (Filtered by Employment Status)")
employment_status = st.selectbox("Select Employment Status:", options=["All"] + df["Employment Status"].unique().tolist())
df4 = df if employment_status == "All" else df[df["Employment Status"] == employment_status]
fig4 = px.histogram(df4, x="Gender", color="Lead_Tier", barmode="group", title="Lead Tier by Gender")
st.plotly_chart(fig4, use_container_width=True)

# === 5. PTB Score vs Policy Purchased ===
st.subheader("5️⃣ PTB Score vs Policy Purchased")
fig5 = px.box(
    df,
    y="PTB_Score",
    color=df["Policy Purchased"].astype(str),
    labels={"Policy Purchased": "Policy Purchased"},
    title="PTB Score Distribution by Purchase Outcome"
)
st.plotly_chart(fig5, use_container_width=True)

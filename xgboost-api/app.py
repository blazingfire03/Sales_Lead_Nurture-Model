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

# === Load Input Data from Cosmos DB ===
@st.cache_data
def load_input_data():
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
        st.error(f"\u274c Failed to fetch data from Cosmos DB: {e}")
        return pd.DataFrame()

# === Load Output File Locally ===
@st.cache_data
def load_output_file():
    try:
        return pd.read_excel("model_predictions.xlsx")  # Update this path if needed
    except Exception as e:
        st.error(f"\u274c Failed to load output file: {e}")
        return pd.DataFrame()

# === Merge Input and Output on Name ===
input_df = load_input_data()
output_df = load_output_file()

if input_df.empty or output_df.empty:
    st.stop()

# Merge using customer Name
try:
    df = pd.merge(input_df, output_df, on="Name", how="inner")
except Exception as e:
    st.error(f"\u274c Failed to merge data: {e}")
    st.stop()

# === KPI METRICS ===
st.title("\ud83d\udcc8 Sales Lead Nurture Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Leads", f"{len(df)}")
col2.metric("Policies Purchased", f"{df['Policy Purchased'].sum()}")
conversion_rate = (df['Policy Purchased'].sum() / len(df)) * 100
col3.metric("Conversion Rate", f"{conversion_rate:.2f}%")

st.markdown("---")

# === CHARTS IN TABS ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Lead Tier vs State",
    "Lead Tier vs Income Bracket",
    "Lead Tier vs Age Group",
    "Lead Tier vs Gender",
    "PTB Score vs Policy Purchased"
])

# === 1. Lead Tier vs State ===
with tab1:
    st.subheader("Lead Tier Distribution by State")
    selected_states = st.multiselect("Select State(s)", df["State"].unique(), default=df["State"].unique())
    filtered = df[df["State"].isin(selected_states)]
    fig = px.histogram(filtered, x="State", color="Lead Tier", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# === 2. Lead Tier vs Income Bracket ===
with tab2:
    st.subheader("Lead Tier by Income Bracket")
    fig = px.histogram(df, x="Income Bracket", color="Lead Tier", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# === 3. Lead Tier vs Age Group ===
with tab3:
    st.subheader("Lead Tier by Age Group")
    selected_ages = st.multiselect("Select Age Group(s)", df["Age Group"].unique(), default=df["Age Group"].unique())
    filtered_age = df[df["Age Group"].isin(selected_ages)]
    fig = px.histogram(filtered_age, x="Age Group", color="Lead Tier", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# === 4. Lead Tier vs Gender (filter by Employment Status) ===
with tab4:
    st.subheader("Lead Tier by Gender")
    selected_emp = st.selectbox("Employment Status", df["Employment Status"].unique())
    filtered_emp = df[df["Employment Status"] == selected_emp]
    fig = px.histogram(filtered_emp, x="Gender", color="Lead Tier", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# === 5. PTB Score vs Policy Purchased ===
with tab5:
    st.subheader("PTB Score Distribution by Policy Purchase")
    fig = px.box(df, x="Policy Purchased", y="PTB Score", color="Policy Purchased",
                 labels={"Policy Purchased": "Policy Purchased (0 = No, 1 = Yes)", "PTB Score": "PTB Score"})
    st.plotly_chart(fig, use_container_width=True)

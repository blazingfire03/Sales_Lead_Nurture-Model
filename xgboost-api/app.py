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

# Load your data from CosmosDB or local fallback
@st.cache_data
def load_data():
    # Example fallback (replace with Cosmos DB fetch function)
    return pd.read_excel("Final_Dataset_Updated.xlsx")

df = load_data()

# === KPI METRICS ===
st.title("📈 Sales Lead Nurture Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Leads", f"{len(df)}")
col2.metric("Policies Purchased", f"{df['Policy Purchased'].sum()}")
conversion_rate = (df['Policy Purchased'].sum() / len(df)) * 100
col3.metric("Conversion Rate", f"{conversion_rate:.2f}%")

st.markdown("---")

# === TABS FOR DIFFERENT CHARTS ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Lead Tier vs State",
    "Lead Tier vs Income",
    "Lead Tier vs Age Group",
    "Lead Tier vs Gender",
    "PTB Score vs Policy Purchase"
])

# === TAB 1: Lead Tier vs State ===
with tab1:
    selected_states = st.multiselect("Filter by State", df["State"].unique(), default=df["State"].unique())
    filtered_df = df[df["State"].isin(selected_states)]
    fig = px.histogram(filtered_df, x="State", color="Lead Tier", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: Lead Tier vs Income Bracket ===
with tab2:
    fig = px.histogram(df, x="Income Bracket", color="Lead Tier", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# === TAB 3: Lead Tier vs Age Group ===
with tab3:
    selected_age_groups = st.multiselect("Select Age Group", df["Age Group"].unique(), default=df["Age Group"].unique())
    filtered_age_df = df[df["Age Group"].isin(selected_age_groups)]
    fig = px.histogram(filtered_age_df, x="Age Group", color="Lead Tier", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# === TAB 4: Lead Tier vs Gender (filtered by Employment Status) ===
with tab4:
    selected_employment = st.selectbox("Select Employment Status", df["Employment Status"].unique())
    filtered_emp_df = df[df["Employment Status"] == selected_employment]
    fig = px.histogram(filtered_emp_df, x="Gender", color="Lead Tier", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# === TAB 5: PTB Score vs Policy Purchased ===
with tab5:
    fig = px.box(df, x="Policy Purchased", y="Behavior Score", color="Policy Purchased",
                 labels={"Policy Purchased": "Policy Purchased (0/1)", "Behavior Score": "PTB Score"})
    st.plotly_chart(fig, use_container_width=True)


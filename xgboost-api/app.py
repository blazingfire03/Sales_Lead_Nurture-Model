import streamlit as st
import pandas as pd
import joblib
import uuid
import os
from PIL import Image
from azure.cosmos import CosmosClient, PartitionKey
import plotly.express as px

# === GLOBAL CONFIG ===
st.set_page_config(page_title="Sales Lead Nurture Model Dashboard", layout="wide")

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_ptb_pipeline.pkl")
    return joblib.load(model_path)

model = load_model()

# === Display Logo ===
logo_path = os.path.join(os.path.dirname(__file__), "analytics_ai_logo.png")
cols = st.columns([1, 8])
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    cols[0].image(logo, width=120)
cols[1].markdown("""<h2 style='padding-top: 20px;'>Sales Lead Nurture Model Dashboard</h2>""", unsafe_allow_html=True)

# === Horizontal Nav Bar (Static for demo) ===
st.markdown("""
<style>
.navbar {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #eee;
}
.navbar button {
    background-color: #f0f2f6;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: bold;
    cursor: pointer;
}
.navbar button.active {
    background-color: #1677ff;
    color: white;
}
</style>
<div class="navbar">
  <button>🏠 Overview</button>
  <button class="active">🧠 Score & Upload</button>
  <button>📊 KPIs</button>
  <button>📈 Charts</button>
  <button>📤 Export</button>
</div>
""", unsafe_allow_html=True)

# === Load Input Data ===
@st.cache_data
def fetch_data():
    endpoint = st.secrets["COSMOS_ENDPOINT"]
    key = st.secrets["COSMOS_KEY"]
    db_name = st.secrets["DATABASE_NAME"]
    container_name = st.secrets["INPUT_CONTAINER"]
    client = CosmosClient(endpoint, credential=key)
    db = client.get_database_client(db_name)
    container = db.get_container_client(container_name)
    items = list(container.read_all_items())
    return pd.DataFrame(items)

def clear_output_container():
    endpoint = st.secrets["COSMOS_ENDPOINT"]
    key = st.secrets["COSMOS_KEY"]
    db_name = st.secrets["DATABASE_NAME"]
    output_container = st.secrets["OUTPUT_CONTAINER"]
    client = CosmosClient(endpoint, credential=key)
    db = client.get_database_client(db_name)
    container = db.get_container_client(output_container)
    items = list(container.read_all_items())
    for item in items:
        container.delete_item(item=item["id"], partition_key=item["id"])
    st.info(f"🧹 Cleared {len(items)} records from '{output_container}'.")

def upload_results(df):
    endpoint = st.secrets["COSMOS_ENDPOINT"]
    key = st.secrets["COSMOS_KEY"]
    db_name = st.secrets["DATABASE_NAME"]
    output_container = st.secrets["OUTPUT_CONTAINER"]
    client = CosmosClient(endpoint, credential=key)
    db = client.get_database_client(db_name)
    container = db.create_container_if_not_exists(
        id=output_container,
        partition_key=PartitionKey(path="/id"),
        offer_throughput=400
    )
    for _, row in df.iterrows():
        record = row.to_dict()
        record["id"] = str(uuid.uuid4())
        container.upsert_item(record)
    st.success(f"✅ Uploaded {len(df)} leads to '{output_container}'.")

# === Score Section ===
st.subheader("📄 Input Data")
df = fetch_data()
if df.empty:
    st.warning("⚠️ No input data found.")
else:
    st.dataframe(df.head())
    required = ['Age', 'Gender', 'Annual Income', 'Income Bracket', 'Marital Status',
                'Employment Status', 'Region', 'Urban/Rural Flag', 'State', 'ZIP Code',
                'Plan Preference Type', 'Web Form Completion Rate', 'Quote Requested',
                'Application Started', 'Behavior Score', 'Application Submitted', 'Application Applied']

    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
    else:
        input_df = df[required]
        df["PTB_Score"] = model.predict_proba(input_df)[:, 1] * 100
        df["Lead_Tier"] = df["PTB_Score"].apply(lambda s: "Platinum" if s>=90 else "Gold" if s>=75 else "Silver" if s>=50 else "Bronze")
        st.subheader("✅ Scored Results")
        st.dataframe(df[["PTB_Score", "Lead_Tier"]].join(df.drop(columns=["PTB_Score", "Lead_Tier"])))
        if st.button("🚀 Clear & Upload to Cosmos DB"):
            clear_output_container()
            upload_results(df)

# === KPIs Section ===
st.markdown("---")
st.subheader("📊 Key Funnel Metrics")

def load_dashboard_data():
    endpoint = st.secrets["COSMOS_ENDPOINT"]
    key = st.secrets["COSMOS_KEY"]
    db_name = st.secrets["DATABASE_NAME"]
    output_container = st.secrets["OUTPUT_CONTAINER"]
    client = CosmosClient(endpoint, credential=key)
    db = client.get_database_client(db_name)
    container = db.get_container_client(output_container)
    items = list(container.read_all_items())
    return pd.DataFrame(items)

dash_df = load_dashboard_data()

if not dash_df.empty:
    total = len(dash_df)
    purchased = dash_df["Policy Purchased"].sum()
    rate = (purchased / total) * 100

    quote_col = "Quote Requested (website)" if "Quote Requested (website)" in dash_df.columns else "Quote Requested"
    quote_requested = dash_df[quote_col].isin(["1", 1, "Yes", True]).sum()
    quote_rate = (quote_requested / total) * 100

    app_started = dash_df["Application Started"].isin(["1", 1, "Yes", True]).sum()
    app_started_rate = (app_started / total) * 100

    app_submitted = dash_df["Application Submitted"].isin(["1", 1, "Yes", True]).sum()
    app_submitted_rate = (app_submitted / total) * 100

    submitted_df = dash_df[dash_df["Application Submitted"].isin(["1", 1, "Yes", True])]
    submitted_to_purchased = (submitted_df["Policy Purchased"].sum() / len(submitted_df)) * 100 if len(submitted_df) > 0 else 0

    # KPI Card Layout
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Leads", f"{total:,}")
    kpi2.metric("Policies Purchased", int(purchased))
    kpi3.metric("Conversion Rate", f"{rate:.2f}%")
    kpi4.metric("Quote Requested Rate", f"{quote_rate:.2f}%")

    kpi5, kpi6, kpi7 = st.columns([1, 1, 1])
    kpi5.metric("App Started Rate", f"{app_started_rate:.2f}%")
    kpi6.metric("App Submitted Rate", f"{app_submitted_rate:.2f}%")
    kpi7.metric("Submitted → Policy Conversion", f"{submitted_to_purchased:.2f}%")

    st.subheader("🥇 Lead Tier Distribution")
    tier_counts = dash_df["Lead_Tier"].value_counts().to_dict()
    b, s, g, p = [tier_counts.get(t, 0) for t in ["Bronze", "Silver", "Gold", "Platinum"]]

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("🥉 Bronze", b)
    t2.metric("🥈 Silver", s)
    t3.metric("🥇 Gold", g)
    t4.metric("🏆 Platinum", p)
else:
    st.warning("⚠️ No scored data found in output container.")

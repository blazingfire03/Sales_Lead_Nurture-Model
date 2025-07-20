import streamlit as st
import pandas as pd
import joblib
import uuid
import os
from PIL import Image
from azure.cosmos import CosmosClient, PartitionKey
import plotly.express as px

st.set_page_config(page_title="PTB Score Dashboard", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_ptb_pipeline.pkl")
    return joblib.load(model_path)

model = load_model()

logo_path = os.path.join(os.path.dirname(__file__), "analytics_ai_logo.png")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=250)
else:
    st.warning("⚠️ Company logo not found.")

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

st.title("🧠 PTB Score Predictor + Cosmos DB Dashboard")

with st.spinner("Loading input data from Cosmos DB..."):
    df = fetch_data()

if df.empty:
    st.warning("⚠️ No input data found.")
else:
    st.subheader("📄 Input Data")
    st.dataframe(df.head())

    required = [
        'Age', 'Gender', 'Annual Income', 'Income Bracket', 'Marital Status',
        'Employment Status', 'Region', 'Urban/Rural Flag', 'State', 'ZIP Code',
        'Plan Preference Type', 'Web Form Completion Rate', 'Quote Requested',
        'Application Started', 'Behavior Score', 'Application Submitted', 'Application Applied'
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
    else:
        input_df = df[required]
        proba = model.predict_proba(input_df)[:, 1]
        df["PTB_Score"] = proba * 100

        def tier(score):
            if score >= 90:
                return "Platinum"
            elif score >= 75:
                return "Gold"
            elif score >= 50:
                return "Silver"
            else:
                return "Bronze"

        df["Lead_Tier"] = df["PTB_Score"].apply(tier)

        st.subheader("✅ Scored Results")
        df["PTB_Score"] = df["PTB_Score"].round(2)
        st.dataframe(df)

        if st.button("🚀 Clear & Upload to Cosmos DB"):
            clear_output_container()
            upload_results(df)

st.markdown("---")
st.header("📊 Lead Conversion Dashboard")

@st.cache_data
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
    submitted_to_purchased = (
        submitted_df["Policy Purchased"].sum() / len(submitted_df) * 100 if len(submitted_df) > 0 else 0
    )

    st.subheader("📈 Key Funnel Metrics")
    kpi_metrics = {
        "Total Leads": f"{total:,}",
        "Policies Purchased": f"{int(purchased)}",
        "Conversion Rate": f"{rate:.2f}%",
        "Quote Requested Rate": f"{quote_rate:.2f}%",
        "App Started Rate": f"{app_started_rate:.2f}%",
        "App Submitted Rate": f"{app_submitted_rate:.2f}%",
        "Submitted → Policy Conversion": f"{submitted_to_purchased:.2f}%"
    }

    st.markdown("""
    <style>
    .kpi-card {
        border: 1px solid #e1e1e1;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        background-color: #fafafa;
    }
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    </style>
    <div class="kpi-container">
    """ +
    "".join([f"<div class='kpi-card'>{label}<br><span style='font-size:24px'>{value}</span></div>" for label, value in kpi_metrics.items()]) +
    "</div>" , unsafe_allow_html=True)

    st.subheader("🏅 Lead Tier Distribution")
    tier_counts = dash_df["Lead_Tier"].value_counts().to_dict()
    color_map = {"Bronze": "#d17c45", "Silver": "#7c97c4", "Gold": "#f2c94c", "Platinum": "#a97ff0"}

    for tier in ["Bronze", "Silver", "Gold", "Platinum"]:
        count = tier_counts.get(tier, 0)
        bar = f"<div style='background:{color_map.get(tier)};width:{min(count/total*100,100)}%;height:10px;border-radius:4px'></div>"
        st.markdown(f"<div style='display:flex;justify-content:space-between'><b>{tier}</b><span>{count}</span></div>{bar}<br>", unsafe_allow_html=True)

else:
    st.warning("⚠️ No scored data found in output container.")

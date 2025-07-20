import streamlit as st
import pandas as pd
import joblib
import uuid
import os
from PIL import Image
from azure.cosmos import CosmosClient, PartitionKey
import plotly.express as px

# === PAGE CONFIG ===
st.set_page_config(page_title="Sales Lead Nurture Model Dashboard", layout="wide")

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_ptb_pipeline.pkl")
    return joblib.load(model_path)

model = load_model()

# === Display Logo and Title ===
logo_path = os.path.join(os.path.dirname(__file__), "analytics_ai_logo.png")
col1, col2 = st.columns([1, 6])
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    col1.image(logo, width=100)
col2.markdown("""
    <h1 style='padding-top: 10px; padding-bottom: 5px;'>PTB Score Predictor + Cosmos DB Dashboard</h1>
    """, unsafe_allow_html=True)

# === Navigation Bar (Static) ===
st.markdown("""
    <style>
        .navbar {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        .navbar button {
            background-color: #f0f2f6;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
        }
        .navbar button.active {
            background-color: #1677ff;
            color: white;
        }
    </style>
    <div class="navbar">
        <button>🧠 S-core & Upload</button>
        <button class="active">📊 KPIs</button>
        <button>📈 Charts</button>
        <button>📤 Export</button>
    </div>
""", unsafe_allow_html=True)

# === Load Dashboard Data ===
@st.cache_data
def load_dashboard_data():
    endpoint = st.secrets["COSMOS_ENDPOINT"]
    key = st.secrets["COSMOS_KEY"]
    db_name = st.secrets["DATABASE_NAME"]
    container_name = st.secrets["OUTPUT_CONTAINER"]
    client = CosmosClient(endpoint, credential=key)
    db = client.get_database_client(db_name)
    container = db.get_container_client(container_name)
    items = list(container.read_all_items())
    return pd.DataFrame(items)

dash_df = load_dashboard_data()

if dash_df.empty:
    st.warning("⚠️ No scored data found in output container.")
else:
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

    # === KPI Card Layout ===
    st.markdown("## Key Funnel Metrics")

    row1 = st.columns(4)
    row1[0].metric("Total Leads", f"{total:,}")
    row1[1].metric("Policies Purchased", f"{int(purchased):,}")
    row1[2].metric("Conversion Rate", f"{rate:.2f}%")
    row1[3].metric("Quote Requested Rate", f"{quote_rate:.2f}%")

    row2 = st.columns(3)
    row2[0].metric("App Started Rate", f"{app_started_rate:.2f}%")
    row2[1].metric("App Submitted Rate", f"{app_submitted_rate:.2f}%")
    row2[2].metric("Submitted + Policy Conversion", f"{submitted_to_purchased:.2f}%")

    # === Tier Distribution ===
    st.markdown("## Lead Tier Distribution")

    tier_counts = dash_df["Lead_Tier"].value_counts().to_dict()
    bronze = tier_counts.get("Bronze", 0)
    silver = tier_counts.get("Silver", 0)
    gold = tier_counts.get("Gold", 0)
    platinum = tier_counts.get("Platinum", 0)

    col_tier1, col_tier2, col_tier3, col_tier4 = st.columns(4)
    col_tier1.metric("🥉 Bronze", bronze)
    col_tier2.metric("🥈 Silver", silver)
    col_tier3.metric("🥇 Gold", gold)
    col_tier4.metric("🏆 Platinum", platinum)

import streamlit as st
import pandas as pd
import joblib
import uuid
import os
from PIL import Image
from azure.cosmos import CosmosClient, PartitionKey
import plotly.express as px

st.set_page_config(page_title="Sales Lead Nurture Model Dashboard", layout="wide")

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

# === TABS ===
tabs = st.tabs(["🏠 Overview", "🤖 S-core & Upload", "📊 KPIs", "📈 Charts", "📤 Export"])

with tabs[1]:
    st.title("Sales Lead Nurture Model Dashboard")
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
            display_df = df.copy()
            display_df["PTB_Score"] = display_df["PTB_Score"].round(2).astype(str) + "%"
            st.dataframe(display_df)

            if st.button("🚀 Clear & Upload to Cosmos DB"):
                clear_output_container()
                upload_results(df)

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

with tabs[4]:
    dash_df = load_dashboard_data()
    if not dash_df.empty:
        st.title("📤 Export Scored Data")
        st.download_button(
            label="📥 Download Scored Leads CSV",
            data=dash_df.to_csv(index=False),
            file_name="scored_leads.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No data available to export.")

with tabs[2]:
    dash_df = load_dashboard_data()
    if not dash_df.empty:
        st.title("📊 KPI Dashboard")
        st.markdown("**Key Funnel Metrics**")

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
        submitted_count = len(submitted_df)
        submitted_to_purchased = (
            submitted_df["Policy Purchased"].sum() / submitted_count * 100
            if submitted_count > 0 else 0
        )

        kpi_values = [
            ("Total Leads", f"{total}"),
            ("Policies Purchased", f"{int(purchased)}"),
            ("Conversion Rate", f"{rate:.2f}%"),
            ("Quote Requested Rate", f"{quote_rate:.2f}%"),
            ("App Started Rate", f"{app_started_rate:.2f}%"),
            ("App Submitted Rate", f"{app_submitted_rate:.2f}%"),
            ("Submitted + Policy Conversion", f"{submitted_to_purchased:.2f}%")
        ]

        kpi_html = "".join([
            f"""
            <div style='flex: 1; min-width: 180px; max-width: 250px; border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin: 8px; background: #fff;'>
                <div style='font-size: 13px; font-weight: 500; color: #333;'>{title}</div>
                <div style='font-size: 28px; font-weight: 700; margin-top: 6px; color: #111;'>{value}</div>
            </div>
            """ for title, value in kpi_values
        ])

        st.markdown(
            f"""
            <div style='display: flex; flex-wrap: wrap; justify-content: space-between;'>
                {kpi_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("**Lead Tier Distribution**")

        tier_counts = dash_df["Lead_Tier"].value_counts().to_dict()

        def render_bar(label, count, color):
            return f"""
            <div style='margin-bottom: 12px;'>
                <strong>{label}</strong>
                <div style='background-color: #eee; border-radius: 5px; overflow: hidden;'>
                    <div style='background-color: {color}; width: {count}px; height: 16px;'></div>
                </div>
                <div style='text-align: right; font-weight: bold;'>{count}</div>
            </div>
            """

        bar_html = ""
        bar_html += render_bar("🥉 Bronze", tier_counts.get("Bronze", 0), "#d97c40")
        bar_html += render_bar("🥈 Silver", tier_counts.get("Silver", 0), "#608cb6")
        bar_html += render_bar("🥇 Gold", tier_counts.get("Gold", 0), "#f2c84b")
        bar_html += render_bar("🏆 Platinum", tier_counts.get("Platinum", 0), "#bb83f2")

        st.markdown(bar_html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No scored data found in output container.")

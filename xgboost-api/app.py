import streamlit as st
import pandas as pd
import joblib
import uuid
import os
from PIL import Image
from azure.cosmos import CosmosClient, PartitionKey
import plotly.express as px

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_ptb_pipeline.pkl")
    return joblib.load(model_path)

model = load_model()

# === Display Logo ===
st.markdown("""
    <div style='display: flex; justify-content: center;'>
        <img src='https://raw.githubusercontent.com/blazingfire03/sales_lead_nurture-model/main/xgboost-api/analytics_ai_logo.png' width='200'/>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center;'>Sales Lead Nurture Model Dashboard</h1>
""", unsafe_allow_html=True)

# === Cosmos DB Utilities ===
@st.cache_data
def fetch_data():
    try:
        endpoint = st.secrets["COSMOS_ENDPOINT"]
        key = st.secrets["COSMOS_KEY"]
        db_name = st.secrets["DATABASE_NAME"]
        container_name = st.secrets["INPUT_CONTAINER"]

        client = CosmosClient(endpoint, credential=key)
        db = client.get_database_client(db_name)
        container = db.get_container_client(container_name)
        items = list(container.read_all_items())
        return pd.DataFrame(items)
    except Exception as e:
        st.error(f"❌ Failed to fetch input data: {e}")
        return pd.DataFrame()

def clear_output_container():
    try:
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
    except Exception as e:
        st.error(f"❌ Failed to clear output container: {e}")

def upload_results(df):
    try:
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
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

# === Load and Score Data ===
df = fetch_data()

tabs = st.tabs(["🏠 Overview", "🧠 Score & Upload", "📊 KPIs", "📈 Charts", "📤 Export"])

with tabs[1]:
    st.header("🧠 PTB Score Prediction")
    if df.empty:
        st.warning("⚠️ No input data found.")
    else:
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
            df["PTB_Score"] = model.predict_proba(input_df)[:, 1] * 100
            df["Lead_Tier"] = df["PTB_Score"].apply(lambda x: "Platinum" if x >= 90 else "Gold" if x >= 75 else "Silver" if x >= 50 else "Bronze")
            st.dataframe(df.head())
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

with tabs[2]:
    dash_df = load_dashboard_data()
    if not dash_df.empty:
        st.subheader("📉 Key Funnel Metrics")

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
            submitted_df["Policy Purchased"].sum() / len(submitted_df) * 100 if not submitted_df.empty else 0
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Leads", total)
        k2.metric("Policies Purchased", int(purchased))
        k3.metric("Conversion Rate", f"{rate:.2f}%")
        k4.metric("Quote Requested Rate", f"{quote_rate:.2f}%")
        k5.metric("App Started Rate", f"{app_started_rate:.2f}%")

        k6, k7 = st.columns(2)
        k6.metric("App Submitted Rate", f"{app_submitted_rate:.2f}%")
        k7.metric("Submitted → Policy Conversion", f"{submitted_to_purchased:.2f}%")

        st.subheader("🥇 Lead Tier Distribution")
        tier_counts = dash_df["Lead_Tier"].value_counts().to_dict()
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("🥉 Bronze", tier_counts.get("Bronze", 0))
        t2.metric("🥈 Silver", tier_counts.get("Silver", 0))
        t3.metric("🥇 Gold", tier_counts.get("Gold", 0))
        t4.metric("🏆 Platinum", tier_counts.get("Platinum", 0))
    else:
        st.warning("⚠️ No scored data found in output container.")

with tabs[3]:
    dash_df = load_dashboard_data()
    if not dash_df.empty:
        st.subheader("📊 Charts")

        st.markdown("### Lead Tier by State")
        states = st.multiselect("State Filter", dash_df["State"].dropna().unique())
        filtered1 = dash_df[dash_df["State"].isin(states)] if states else dash_df
        fig1 = px.histogram(filtered1, x="State", color="Lead_Tier", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### Quote Requested vs Purchase Channel")
        genders = st.multiselect("Gender", dash_df["Gender"].dropna().unique(), default=dash_df["Gender"].dropna().unique())
        incomes = st.multiselect("Income Bracket", dash_df["Income Bracket"].dropna().unique(), default=dash_df["Income Bracket"].dropna().unique())
        quote_vals = st.multiselect("Quote Requested", dash_df[quote_col].dropna().unique(), default=dash_df[quote_col].dropna().unique())
        filtered5 = dash_df[
            dash_df["Gender"].isin(genders) &
            dash_df["Income Bracket"].isin(incomes) &
            dash_df[quote_col].isin(quote_vals)
        ]
        fig5 = px.histogram(filtered5, x="Purchase Channel", color=quote_col, barmode="group")
        st.plotly_chart(fig5, use_container_width=True)

with tabs[4]:
    dash_df = load_dashboard_data()
    if not dash_df.empty:
        st.download_button(
            label="📥 Download Scored Leads as CSV",
            data=dash_df.to_csv(index=False).encode("utf-8"),
            file_name="scored_leads.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No data to export.")

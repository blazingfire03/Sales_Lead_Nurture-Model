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

with tabs[3]:
    dash_df = load_dashboard_data()
    if not dash_df.empty:
        st.subheader("1️⃣ Lead Tier by State")
        states = st.multiselect("Filter by State:", dash_df["State"].dropna().unique())
        filtered1 = dash_df[dash_df["State"].isin(states)] if states else dash_df
        fig1 = px.histogram(filtered1, x="State", color="Lead_Tier", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("2️⃣ Lead Tier by Income Bracket")
        fig2 = px.histogram(dash_df, x="Income Bracket", color="Lead_Tier", barmode="stack")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("3️⃣ Lead Tier by Age Group")
        ages = st.multiselect("Filter by Age Group:", dash_df["Age Group"].dropna().unique())
        filtered3 = dash_df[dash_df["Age Group"].isin(ages)] if ages else dash_df
        fig3 = px.histogram(filtered3, x="Age Group", color="Lead_Tier", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("4️⃣ Lead Tier by Gender (Filtered by Employment)")
        jobs = ["All"] + dash_df["Employment Status"].dropna().unique().tolist()
        emp_filter = st.selectbox("Employment Status:", jobs)
        filtered4 = dash_df if emp_filter == "All" else dash_df[dash_df["Employment Status"] == emp_filter]
        fig4 = px.histogram(filtered4, x="Gender", color="Lead_Tier", barmode="group")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("5️⃣ Quote Requested vs Purchase Channel")
        quote_col = "Quote Requested (website)" if "Quote Requested (website)" in dash_df.columns else "Quote Requested"
        gender_options = dash_df["Gender"].dropna().unique().tolist()
        selected_genders = st.multiselect("Filter by Gender:", gender_options, default=gender_options)

        income_options = dash_df["Income Bracket"].dropna().unique().tolist()
        selected_incomes = st.multiselect("Filter by Income Bracket:", income_options, default=income_options)

        quote_options = dash_df[quote_col].dropna().unique().tolist()
        selected_quotes = st.multiselect("Filter by Quote Requested:", quote_options, default=quote_options)

        filtered5 = dash_df[
            (dash_df["Gender"].isin(selected_genders)) &
            (dash_df["Income Bracket"].isin(selected_incomes)) &
            (dash_df[quote_col].isin(selected_quotes))
        ]

        fig5 = px.histogram(
            filtered5,
            x="Purchase Channel",
            color=quote_col,
            barmode="group",
            title="Quote Requested vs Purchase Channel"
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("⚠️ No scored data found in output container.")

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

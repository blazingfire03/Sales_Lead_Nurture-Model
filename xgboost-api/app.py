import streamlit as st
import pandas as pd
import joblib
import uuid
import os
from azure.cosmos import CosmosClient, PartitionKey

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_ptb_pipeline.pkl")
    return joblib.load(model_path)

model = load_model()

# === Load Input Data ===
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

# === Clear Output Container ===
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

# === Upload New Scored Leads ===
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

# === App UI ===
st.title("🧠 PTB Score Predictor + Cosmos DB Dashboard")

with st.spinner("Loading input data from Cosmos DB..."):
    df = fetch_data()

if df.empty:
    st.warning("⚠️ No input data found.")
else:
    st.subheader("📄 Input Data")
    st.dataframe(df.head())

    # Required features
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
        try:
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
            st.dataframe(df[["PTB_Score", "Lead_Tier"]].join(df.drop(columns=["PTB_Score", "Lead_Tier"])))

            if st.button("🚀 Clear & Upload to Cosmos DB"):
                clear_output_container()
                upload_results(df)

        except Exception as e:
            st.error(f"❌ Scoring failed: {e}")

# === DASHBOARD SECTION ===
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

    # Quote Requested Rate
    quote_col = "Quote Requested (website)" if "Quote Requested (website)" in dash_df.columns else "Quote Requested"
    quote_requested = dash_df[quote_col].isin(["1", 1, "Yes", True]).sum()
    quote_rate = (quote_requested / total) * 100

    # Application Started Rate
    app_started = dash_df["Application Started"].isin(["1", 1, "Yes", True]).sum()
    app_started_rate = (app_started / total) * 100

    # Application Submitted Rate
    app_submitted = dash_df["Application Submitted"].isin(["1", 1, "Yes", True]).sum()
    app_submitted_rate = (app_submitted / total) * 100

    # Policy Conversion from Submitted Applications
    submitted_df = dash_df[dash_df["Application Submitted"].isin(["1", 1, "Yes", True])]
    submitted_count = len(submitted_df)
    submitted_to_purchased = (
        submitted_df["Policy Purchased"].sum() / submitted_count * 100
        if submitted_count > 0 else 0
    )

    # === Display KPIs in two rows ===
    st.subheader("📈 Key Funnel Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Leads", total)
    c2.metric("Policies Purchased", int(purchased))
    c3.metric("Conversion Rate", f"{rate:.2f}%")

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Quote Requested Rate", f"{quote_rate:.2f}%")
    c5.metric("App Started Rate", f"{app_started_rate:.2f}%")
    c6.metric("App Submitted Rate", f"{app_submitted_rate:.2f}%")
    c7.metric("Submitted → Policy Conversion", f"{submitted_to_purchased:.2f}%")
    
    total = len(dash_df)
    purchased = dash_df["Policy Purchased"].sum()
    rate = (purchased / total) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Leads", total)
    c2.metric("Policies Purchased", int(purchased))
    c3.metric("Conversion Rate", f"{rate:.2f}%")

    st.divider()

    import plotly.express as px

    # Chart 1: Lead Tier vs State
    st.subheader("1️⃣ Lead Tier by State")
    states = st.multiselect("Filter by State:", dash_df["State"].dropna().unique())
    filtered1 = dash_df[dash_df["State"].isin(states)] if states else dash_df
    fig1 = px.histogram(filtered1, x="State", color="Lead_Tier", barmode="group")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Lead Tier vs Income Bracket
    st.subheader("2️⃣ Lead Tier by Income Bracket")
    fig2 = px.histogram(dash_df, x="Income Bracket", color="Lead_Tier", barmode="stack")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Lead Tier vs Age Group
    st.subheader("3️⃣ Lead Tier by Age Group")
    ages = st.multiselect("Filter by Age Group:", dash_df["Age Group"].dropna().unique())
    filtered3 = dash_df[dash_df["Age Group"].isin(ages)] if ages else dash_df
    fig3 = px.histogram(filtered3, x="Age Group", color="Lead_Tier", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Lead Tier vs Gender filtered by Employment
    st.subheader("4️⃣ Lead Tier by Gender (Filtered by Employment)")
    jobs = ["All"] + dash_df["Employment Status"].dropna().unique().tolist()
    emp_filter = st.selectbox("Employment Status:", jobs)
    filtered4 = dash_df if emp_filter == "All" else dash_df[dash_df["Employment Status"] == emp_filter]
    fig4 = px.histogram(filtered4, x="Gender", color="Lead_Tier", barmode="group")
    st.plotly_chart(fig4, use_container_width=True)

            # Chart 5: Quote Requested vs Purchase Channel (Filtered by Gender, Income Bracket, Quote Requested)
    st.subheader("5️⃣ Quote Requested vs Purchase Channel")

    # Filter options
    gender_options = dash_df["Gender"].dropna().unique().tolist()
    selected_genders = st.multiselect("Filter by Gender:", gender_options, default=gender_options)

    income_options = dash_df["Income Bracket"].dropna().unique().tolist()
    selected_incomes = st.multiselect("Filter by Income Bracket:", income_options, default=income_options)

    quote_col = "Quote Requested (website)" if "Quote Requested (website)" in dash_df.columns else "Quote Requested"
    quote_options = dash_df[quote_col].dropna().unique().tolist()
    selected_quotes = st.multiselect("Filter by Quote Requested:", quote_options, default=quote_options)

    # Apply all filters
    filtered5 = dash_df[
        (dash_df["Gender"].isin(selected_genders)) &
        (dash_df["Income Bracket"].isin(selected_incomes)) &
        (dash_df[quote_col].isin(selected_quotes))
    ]

    # Draw grouped bar chart
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

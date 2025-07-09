import streamlit as st
import pandas as pd
from azure.cosmos import CosmosClient

# Load secrets
endpoint = st.secrets["COSMOS_ENDPOINT"]
key = st.secrets["COSMOS_KEY"]
database_name = st.secrets["DATABASE_NAME"]
container_name = st.secrets["INPUT_CONTAINER"]

# Cache the Cosmos DB query
@st.cache_data
def fetch_data():
    client = CosmosClient(endpoint, credential=key)
    db = client.get_database_client(database_name)
    container = db.get_container_client(container_name)
    items = list(container.read_all_items())
    return pd.DataFrame(items)

# Load data
st.title("📊 Sales Lead Nurture Dashboard")
df = fetch_data()

if df.empty:
    st.warning("No data found in Cosmos DB.")
else:
    # KPI Metrics
    st.subheader("🔑 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Leads", len(df))
    col2.metric("Policies Purchased", df['Policy Purchased'].sum())
    col3.metric("Conversion Rate (%)", round(100 * df['Policy Purchased'].sum() / len(df), 2))

    # Filter
    st.sidebar.header("🔍 Filter Leads")
    selected_state = st.sidebar.selectbox("Select State", ["All"] + sorted(df['State'].dropna().unique().tolist()))
    if selected_state != "All":
        df = df[df["State"] == selected_state]

    # Charts
    st.subheader("🗺️ Leads by State")
    st.bar_chart(df["State"].value_counts())

    st.subheader("📈 Application Funnel")
    funnel_cols = ['Application Started', 'Application Submitted', 'Policy Purchased']
    funnel_data = df[funnel_cols].sum()
    st.line_chart(funnel_data)

    st.subheader("📂 Full Data Table")
    st.dataframe(df)

    st.download_button("Download CSV", df.to_csv(index=False), "leads_data.csv")

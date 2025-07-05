from flask import Flask, jsonify
import pandas as pd
import joblib
from azure.cosmos import CosmosClient, PartitionKey

app = Flask(__name__)

# === Cosmos DB Settings ===
COSMOS_ENDPOINT = "https://leadnurturecosmosdb.documents.azure.com:443/"
COSMOS_KEY = "q0AmTLUTz2rJhQDEppSoElKfawnuZVbj5Yqe8lyFbv8bbCIoewm5jqgo8UutqEfnnm28g0Idmg1EACDbpTApqQ=="
DATABASE_NAME = "Lead_Nurture_DB"
INPUT_CONTAINER = "CustomerData"
OUTPUT_CONTAINER = "ScoredLeads"

# === Load Model ===
print("➡ Loading XGBoost model...")
model = joblib.load("xgboost-api/xgboost_ptb_pipeline.pkl")
print("✅ Model loaded.")

@app.route('/score', methods=['GET'])
def score_customers():
    try:
        print("➡ Connecting to Cosmos DB...")
        client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        db = client.get_database_client(DATABASE_NAME)
        input_container = db.get_container_client(INPUT_CONTAINER)

        print("➡ Reading all customer data...")
        raw_items = list(input_container.read_all_items())
        print(f"✅ Retrieved {len(raw_items)} items.")

        df = pd.DataFrame(raw_items)
        print("➡ DataFrame created with shape:", df.shape)

        kept_features = [
            'Age', 'Gender', 'Annual Income', 'Income Bracket', 'Marital Status',
            'Employment Status', 'Region', 'Urban/Rural Flag', 'State', 'ZIP Code',
            'Plan Preference Type', 'Web Form Completion Rate', 'Quote Requested',
            'Application Started', 'Behavior Score', 'Application Submitted', 'Application Applied'
        ]
        df = df[df.columns.intersection(kept_features)]
        print("➡ Filtered to kept features. Shape now:", df.shape)

        print("➡ Making predictions...")
        predictions = model.predict(df)
        df['PTB_Score'] = predictions
        print("✅ Predictions complete.")

        def tier(score):
            if score >= 0.8:
                return "Platinum"
            elif score >= 0.6:
                return "Gold"
            elif score >= 0.4:
                return "Silver"
            else:
                return "Bronze"

        df['Lead_Tier'] = df['PTB_Score'].apply(tier)
        print("➡ Lead tier assigned.")

        print("➡ Writing results to Cosmos DB...")
        output_container = db.create_container_if_not_exists(
            id=OUTPUT_CONTAINER,
            partition_key=PartitionKey(path="/id")
        )

        for idx, row in df.iterrows():
            record = row.to_dict()
            record["id"] = str(idx + 1)
            output_container.upsert_item(record)

        print(f"✅ Successfully wrote {len(df)} scored leads to Cosmos DB.")
        return jsonify({"message": f"✅ Scored {len(df)} customers and saved to '{OUTPUT_CONTAINER}'."})

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

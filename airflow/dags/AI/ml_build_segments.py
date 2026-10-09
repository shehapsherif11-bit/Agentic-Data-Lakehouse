import os
import shutil
import tempfile
from pathlib import Path

import databricks.sql
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env')
load_dotenv(env_path)

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

SEGMENTS_TABLE = "workspace.zomato_gold.ai_customer_segments"
STAGING_VOLUME = "workspace.zomato_gold.staging"
STAGING_PATH = "/Volumes/workspace/zomato_gold/staging"


def bulk_write_segments(cursor, df, local_dir):
    """Bulk-load the segments: write ONE Parquet file, PUT it into a Unity Catalog volume, then build the table
    from it with a single CREATE OR REPLACE TABLE ... AS SELECT. The table is swapped atomically (no DROP window
    where readers see nothing) and the whole load is one statement instead of ~100 row-by-row INSERTs."""
    local = Path(local_dir) / "ai_customer_segments.parquet"
    df[["user_id", "customer_segment"]].astype({"user_id": "int32"}).to_parquet(local, index=False)

    remote = f"{STAGING_PATH}/ai_customer_segments.parquet"
    cursor.execute(f"CREATE VOLUME IF NOT EXISTS {STAGING_VOLUME}")
    cursor.execute(f"PUT '{local.as_posix()}' INTO '{remote}' OVERWRITE")
    cursor.execute(f"""
        CREATE OR REPLACE TABLE {SEGMENTS_TABLE} AS
        SELECT CAST(user_id AS INT) AS user_id, CAST(customer_segment AS STRING) AS customer_segment
        FROM parquet.`{remote}`
    """)
    cursor.execute(f"REMOVE '{remote}'")


def build_segments():
    print("Connecting to Databricks...")
    staging_dir = tempfile.mkdtemp(prefix="segments_")   # only this directory may be uploaded via PUT
    connection = databricks.sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
        staging_allowed_local_path=staging_dir,
    )
    
    cursor = connection.cursor()
    
    try:
        print("Extracting customer features (frequency, AOV, recency)...")
        query = """
            SELECT 
                fo.user_id,
                COUNT(DISTINCT fo.order_id) as frequency,
                AVG(fo.sales_amount) as aov,
                MAX(dd.full_date) as last_order_date
            FROM workspace.zomato_gold.fact_orders fo
            JOIN workspace.zomato_gold.dim_date dd ON fo.date_id = dd.date_id
            GROUP BY fo.user_id
        """
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        if not data:
            print("No data found for segmentation.")
            return
            
        df = pd.DataFrame([dict(zip(columns, row)) for row in data])
        
        df['last_order_date'] = pd.to_datetime(df['last_order_date'])
        max_date = df['last_order_date'].max()
        df['recency'] = (max_date - df['last_order_date']).dt.days
        
        features = df[['frequency', 'aov', 'recency']].fillna(0)
        
        print("Scaling features and running K-Means...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(scaled_features)
        
        cluster_means = df.groupby('cluster')[['frequency', 'aov', 'recency']].mean()
        
        # Rank clusters by a composite score to ensure label stability across retrains
        # Composite score = normalized(frequency) + normalized(aov) - normalized(recency)
        # We use MinMax scaling just for scoring the cluster centers
        from sklearn.preprocessing import MinMaxScaler
        cluster_scaler = MinMaxScaler()
        
        # We need frequency, aov, and negative recency (lower recency is better)
        # Let's create a DataFrame of the cluster centers
        centers = cluster_means.copy()
        centers['inv_recency'] = -centers['recency']
        
        scaled_centers = cluster_scaler.fit_transform(centers[['frequency', 'aov', 'inv_recency']])
        
        # Calculate composite score for each cluster
        cluster_scores = []
        for i in range(len(scaled_centers)):
            # Weight: frequency (40%), AOV (40%), recency (20%)
            score = (scaled_centers[i][0] * 0.4) + (scaled_centers[i][1] * 0.4) + (scaled_centers[i][2] * 0.2)
            cluster_scores.append((i, score, centers.loc[i, 'recency']))
            
        # Sort by score descending
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        # The logic:
        # Highest score -> High Value
        # Second highest -> Loyal
        # Now between the remaining two, the one with highest recency (worst) -> At Risk
        # The other -> Occasional
        
        c0, c1, c2, c3 = cluster_scores[0][0], cluster_scores[1][0], cluster_scores[2][0], cluster_scores[3][0]
        
        remaining = [cluster_scores[2], cluster_scores[3]]
        # Sort remaining by recency descending (highest recency = At Risk)
        remaining.sort(key=lambda x: x[2], reverse=True)
        at_risk_cluster = remaining[0][0]
        occasional_cluster = remaining[1][0]
        
        cluster_mapping = {
            c0: "High Value",
            c1: "Loyal",
            occasional_cluster: "Occasional",
            at_risk_cluster: "At Risk"
        }
        
        df['customer_segment'] = df['cluster'].map(cluster_mapping)
        
        print("Saving segments back to Databricks (ai_customer_segments)...")
        bulk_write_segments(cursor, df, staging_dir)
        
        print("Successfully built and saved customer segments!")
        
    finally:
        cursor.close()
        connection.close()
        shutil.rmtree(staging_dir, ignore_errors=True)

if __name__ == "__main__":
    build_segments()

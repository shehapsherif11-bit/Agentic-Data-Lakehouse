from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

# مسارات الـ dbt جوه حاوية Docker بتاعتنا
DBT = "/opt/airflow/dbt_venv/bin/dbt"
DBT_PROJECT = "/opt/airflow/dbt/zomato"

# The agent's query / schema caches treat anything cached before this timestamp as stale (src/utils/pipeline_marker.py).
# It is written after EVERY build that changes Gold data, not only the last one: if the AI steps fail, the core build
# has already rewritten the Gold facts and cached answers must not survive until their TTL.
MARK_DATA_CHANGED = 'mkdir -p /opt/airflow/metadata && date -u +"%Y-%m-%dT%H:%M:%SZ" > /opt/airflow/metadata/last_ai_run.txt'

# Applied to every task; individual tasks override execution_timeout where they need more/less.
default_args = {
    "owner": "data-eng",
    "retries": 2,                                  # transient Databricks / Groq failures get 2 more tries
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,             # 2m, 4m, ...
    "max_retry_delay": timedelta(minutes=15),
    "execution_timeout": timedelta(minutes=30),    # a hung task is killed instead of blocking the DAG forever
}

with DAG(
    dag_id="zomato_ai_pipeline_databricks",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,                             # never let two runs write the same tables concurrently
    dagrun_timeout=timedelta(hours=2),
    default_args=default_args,
    tags=["zomato", "dbt", "databricks", "ai"],
) as dag:

    # 1. بناء طبقات البيانات (Bronze, Silver, Gold) وتجاهل جزء الـ AI مؤقتاً
    dbt_build_core = BashOperator(
        task_id="dbt_build_core",
        bash_command=f"{DBT} build --exclude tag:ai --project-dir {DBT_PROJECT} --profiles-dir {DBT_PROJECT} && {MARK_DATA_CHANGED}",
        execution_timeout=timedelta(hours=1),
    )

    # 2. سكريبت البايثون اللي بيكلم OpenAI يحلل المشاعر
    enrich_reviews = BashOperator(
        task_id="enrich_reviews",
        bash_command='python /opt/airflow/dags/AI/enrich_reviews.py',
    )
    
    # 2.5. سكريبت الـ K-Means لتقسيم العملاء
    ml_build_segments = BashOperator(
        task_id="ml_build_segments",
        bash_command='python /opt/airflow/dags/AI/ml_build_segments.py',
    )

    # 3. بناء جداول الـ AI في dbt ودمجها في الـ Gold
    dbt_build_ai = BashOperator(
        task_id="dbt_build_ai",
        bash_command=f"{DBT} build --select tag:ai --project-dir {DBT_PROJECT} --profiles-dir {DBT_PROJECT} && {MARK_DATA_CHANGED}"
    )

    # ترتيب تنفيذ المهام (المايسترو)
    dbt_build_core >> [enrich_reviews, ml_build_segments] >> dbt_build_ai
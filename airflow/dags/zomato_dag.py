from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

# مسارات الـ dbt جوه حاوية Docker بتاعتنا
DBT = "/opt/airflow/dbt_venv/bin/dbt"
DBT_PROJECT = "/opt/airflow/dbt/zomato"

with DAG(
    dag_id="zomato_ai_pipeline_databricks",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["zomato", "dbt", "databricks", "ai"],
) as dag:

    # 1. بناء طبقات البيانات (Bronze, Silver, Gold) وتجاهل جزء الـ AI مؤقتاً
    dbt_build_core = BashOperator(
        task_id="dbt_build_core",
        bash_command=f"{DBT} build --exclude tag:ai --project-dir {DBT_PROJECT} --profiles-dir {DBT_PROJECT}",
    )

    # 2. سكريبت البايثون اللي بيكلم OpenAI يحلل المشاعر
    enrich_reviews = BashOperator(
        task_id="enrich_reviews",
bash_command='python /opt/airflow/dags/ai/enrich_reviews.py',
    )

    # 3. بناء جداول الـ AI في dbt ودمجها في الـ Gold
    dbt_build_ai = BashOperator(
        task_id="dbt_build_ai",
        bash_command=f"{DBT} build --select tag:ai --project-dir {DBT_PROJECT} --profiles-dir {DBT_PROJECT}"
    )

    # ترتيب تنفيذ المهام (المايسترو)
    dbt_build_core >> enrich_reviews >> dbt_build_ai
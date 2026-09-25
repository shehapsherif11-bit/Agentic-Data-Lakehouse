# Airflow 3.x image + Databricks provider + FAB auth + dbt-venv + AI libraries.
FROM apache/airflow:3.0.3

# Providers and AI required libraries
RUN pip install --no-cache-dir \
    apache-airflow-providers-databricks \
    apache-airflow-providers-fab \
    databricks-sql-connector \
    requests

# dbt in its own venv so it never clashes with Airflow's deps.
# dbt-databricks التعديل الجوهري لتسطيب 
RUN python -m venv /opt/airflow/dbt_venv && \
    /opt/airflow/dbt_venv/bin/pip install --no-cache-dir "dbt-databricks==1.8.*"
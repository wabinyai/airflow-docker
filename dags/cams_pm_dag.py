from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from pathlib import Path

# Import functions from your existing script 
from camspipeline import (
    configure_cds_api,
    retrieve_variable,
    process_netcdf,
    save_to_postgres,
    engine
)

default_args = {
    'owner': 'airqo',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='cams_pm_forecast_pipeline',
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['cams', 'air_quality']
) as dag:

    def download_pm25():
        configure_cds_api()
        retrieve_variable('particulate_matter_2.5um', '/tmp/pm25.zip')

    def download_pm10():
        retrieve_variable('particulate_matter_10um', '/tmp/pm10.zip')

    def process_and_store_pm25():
        ds = process_netcdf('/tmp/pm25.zip', 'pm2p5')
        save_to_postgres(ds, "cams_pm25", "pm2p5", engine)

    def process_and_store_pm10():
        ds = process_netcdf('/tmp/pm10.zip', 'pm10')
        save_to_postgres(ds, "cams_pm10", "pm10", engine)

    task_download_pm25 = PythonOperator(
        task_id='download_pm25',
        python_callable=download_pm25
    )

    task_download_pm10 = PythonOperator(
        task_id='download_pm10',
        python_callable=download_pm10
    )

    task_process_store_pm25 = PythonOperator(
        task_id='process_and_store_pm25',
        python_callable=process_and_store_pm25
    )

    task_process_store_pm10 = PythonOperator(
        task_id='process_and_store_pm10',
        python_callable=process_and_store_pm10
    )

    task_download_pm25 >> task_process_store_pm25
    task_download_pm10 >> task_process_store_pm10

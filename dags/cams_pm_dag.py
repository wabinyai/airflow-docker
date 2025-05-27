from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

# Import utility functions
from utils.camspipeline import CamsDownload
from utils.process_and_store_pm import Vectortimegenerator

default_args = {
    'owner': 'airqo',
    'depends_on_past': False,
    'email': ['alerts@airqo.net'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    dag_id='cams_pm_pipeline_dag',
    default_args=default_args,
    description='DAG to download, process and tile CAMS PM2.5 and PM10 data',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['cams', 'pm25', 'pm10', 'airquality'],
) as dag:

    download_and_store_task = PythonOperator(
        task_id='download_and_store_pm_data',
        python_callable=CamsDownload.download_and_store_pm,
    )

    generate_pm25_tiles_task = PythonOperator(
        task_id='generate_pm25_vector_tiles',
        python_callable=Vectortimegenerator.generate_pm25_vector_tiles,
    )

    generate_pm10_tiles_task = PythonOperator(
        task_id='generate_pm10_vector_tiles',
        python_callable=Vectortimegenerator.generate_pm10_vector_tiles,
    )

    # Set task dependencies
    download_and_store_task >> [generate_pm25_tiles_task, generate_pm10_tiles_task]

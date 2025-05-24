from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import dag, task
from airflow.models import Variable
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import shutil
####
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import shutil

# Configure logging with Airflow context
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import functions from provided scripts
from utils.camspipeline import (
    configure_cds_api,
    retrieve_variable,
    process_netcdf,
    save_to_postgres,
    engine
)
from utils.process_and_store_pm import (
    generate_pm25_vector_tiles,
    generate_pm10_vector_tiles
)

# Default arguments for the DAG
default_args = {
    'owner': 'airqo',
    'retries': 2,  # Increased retries for robustness
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),  # Prevent runaway tasks
    'max_active_runs': 1,  # Prevent parallel DAG runs
    'email_on_failure': False,
    'email': ['airqo-team@example.com'],  # Replace with actual email
}

@dag(
    dag_id='cams_pm_forecast_pipeline',
    default_args=default_args,
    start_date=datetime(2025, 5, 24),
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    tags=['cams', 'air_quality', 'forecast'],
    description='Pipeline to download, process, store CAMS PM2.5/PM10 data, and generate vector tiles',
)
def cams_pm_forecast_pipeline():
    # Use Airflow Variables for file paths (configurable via Airflow UI)
    TEMP_DIR = Variable.get('cams_temp_dir', default_var='/tmp/cams')
    PM25_FILE = str(Path(TEMP_DIR) / 'pm25.zip')
    PM10_FILE = str(Path(TEMP_DIR) / 'pm10.zip')

    @task
    def setup_temp_dir():
        """Create temporary directory for storing downloaded files."""
        try:
            Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created temporary directory: {TEMP_DIR}")
        except Exception as e:
            logger.error(f"Failed to create temp directory {TEMP_DIR}: {str(e)}")
            raise

    @task
    def download_pm25():
        """Download PM2.5 data from CDS API."""
        try:
            # Assume .cdsapirc is pre-configured or use Airflow Connection
            configure_cds_api()
            retrieve_variable('particulate_matter_2.5um', PM25_FILE)
            logger.info(f"Downloaded PM2.5 data to {PM25_FILE}")
        except Exception as e:
            logger.error(f"Failed to download PM2.5 data: {str(e)}")
            raise

    @task
    def download_pm10():
        """Download PM10 data from CDS API."""
        try:
            configure_cds_api()
            retrieve_variable('particulate_matter_10um', PM10_FILE)
            logger.info(f"Downloaded PM10 data to {PM10_FILE}")
        except Exception as e:
            logger.error(f"Failed to download PM10 data: {str(e)}")
            raise

    @task
    def process_and_store_pm25():
        """Process PM2.5 NetCDF and store in Postgres."""
        try:
            ds = process_netcdf(PM25_FILE, 'pm2p5')
            save_to_postgres(ds, "cams_pm25", "pm2p5", engine)
            logger.info("Processed and stored PM2.5 data")
        except Exception as e:
            logger.error(f"Failed to process/store PM2.5 data: {str(e)}")
            raise

    @task
    def process_and_store_pm10():
        """Process PM10 NetCDF and store in Postgres."""
        try:
            ds = process_netcdf(PM10_FILE, 'pm10')
            save_to_postgres(ds, "cams_pm10", "pm10", engine)
            logger.info("Processed and stored PM10 data")
        except Exception as e:
            logger.error(f"Failed to process/store PM10 data: {str(e)}")
            raise

    @task
    def generate_pm25_tiles():
        """Generate PM2.5 vector tiles."""
        try:
            generate_pm25_vector_tiles()
            logger.info("Generated PM2.5 vector tiles")
        except Exception as e:
            logger.error(f"Failed to generate PM2.5 tiles: {str(e)}")
            raise

    @task
    def generate_pm10_tiles():
        """Generate PM10 vector tiles."""
        try:
            generate_pm10_vector_tiles()
            logger.info("Generated PM10 vector tiles")
        except Exception as e:
            logger.error(f"Failed to generate PM10 tiles: {str(e)}")
            raise

    @task
    def cleanup_files():
        """Clean up temporary files and directories."""
        try:
            if Path(TEMP_DIR).exists():
                shutil.rmtree(TEMP_DIR)
                logger.info(f"Removed temporary directory: {TEMP_DIR}")
            Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)  # Recreate for next run
        except Exception as e:
            logger.warning(f"Failed to clean up {TEMP_DIR}: {str(e)}")

    # Define tasks
    t0 = setup_temp_dir()
    t1 = download_pm25()
    t2 = download_pm10()
    t3 = process_and_store_pm25()
    t4 = process_and_store_pm10()
    t5 = generate_pm25_tiles()
    t6 = generate_pm10_tiles()
    t7 = cleanup_files()

    # Set dependencies
    t0 >> [t1, t2]  # Setup temp dir before downloads
    t1 >> t3 >> t5  # PM2.5: download -> process -> tiles
    t2 >> t4 >> t6  # PM10: download -> process -> tiles
    [t5, t6] >> t7  # Cleanup after both tile generations

# Instantiate the DAG
dag = cams_pm_forecast_pipeline()
import datetime
import cdsapi
import netCDF4 as nc
import numpy as np
import xarray as xr
import zipfile
import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Float, DateTime, Index, Numeric
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from pathlib import Path
import cftime
import logging
import tempfile
import shutil

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CamsDownload:
    """A class to handle downloading, processing, and storing CAMS data."""
    
    def __init__(self):
        """Initialize the CamsDownload class with database and CDS API configuration."""
        # Load environment variables
        load_dotenv()
        self.DB_USER = os.getenv('DB_USER', 'airqo')
        self.DB_PASS = os.getenv('DB_PASS')
        self.DB_HOST = os.getenv('DB_HOST', 'localhost')
        self.DB_PORT = os.getenv('DB_PORT', '5432')
        self.DB_NAME = os.getenv('DB_NAME')
        self.CDS_API_KEY = os.getenv('CDS_API_KEY')

        # Validate environment variables
        if not all([self.DB_USER, self.DB_PASS, self.DB_HOST, self.DB_PORT, self.DB_NAME, self.CDS_API_KEY]):
            logger.error("Missing required environment variables. Please check .env file.")
            raise ValueError("Missing required environment variables.")

        # Initialize database engine
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )
        
        self.cdsapirc_path = Path.home() / ".cdsapirc"

    def configure_cds_api(self):
        """Configure the CDS API client by writing the .cdsapirc file."""
        if not self.CDS_API_KEY:
            logger.error("CDS API key is not set.")
            raise ValueError("CDS API key is not set.")
        cdsapirc_content = f"url: https://ads.atmosphere.copernicus.eu/api\nkey: {self.CDS_API_KEY}"
        if not self.cdsapirc_path.exists() or self.cdsapirc_path.read_text() != cdsapirc_content:
            self.cdsapirc_path.write_text(cdsapirc_content)
            logger.info(f"Created/updated {self.cdsapirc_path}")

    def retrieve_variable(self, variable_name: str, output_zip_path: str) -> None:
        """Download a variable from the CAMS dataset."""
        try:
            c = cdsapi.Client()
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)
            date_range = f"{yesterday:%Y-%m-%d}/{today:%Y-%m-%d}"

            logger.info(f"Retrieving {variable_name} data for {date_range}...")
            c.retrieve(
                'cams-global-atmospheric-composition-forecasts',
                {
                    'date': date_range,
                    'type': 'forecast',
                    'format': 'netcdf_zip',
                    'leadtime_hour': '12',
                    'time': ['00:00', '12:00'],
                    'variable': variable_name,
                },
                output_zip_path
            )
            logger.info(f"Downloaded {variable_name} to {output_zip_path}")
        except Exception as e:
            logger.error(f"Failed to download {variable_name}: {e}")
            raise

    def process_netcdf(self, zip_file_path: str, variable_short_name: str) -> tuple[xr.Dataset, str]:
        """Process a NetCDF file and return an xarray Dataset and the temporary directory path."""
        temp_dir = tempfile.mkdtemp()
        extract_dir = Path(temp_dir)
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            nc_files = list(extract_dir.glob("*.nc"))
            if not nc_files:
                logger.error(f"No .nc files found in {extract_dir}")
                raise FileNotFoundError(f"No .nc files found in {extract_dir}")

            file_path = nc_files[0]
            logger.info(f"Processing NetCDF file: {file_path}")

            with nc.Dataset(file_path) as dataset:
                if variable_short_name not in dataset.variables:
                    logger.error(f"Variable {variable_short_name} not found in NetCDF file")
                    raise KeyError(f"Variable {variable_short_name} not found")

                longitude = dataset.variables['longitude'][:]
                latitude = dataset.variables['latitude'][:]
                valid_time = dataset.variables['valid_time'][:]
                variable_data = dataset.variables[variable_short_name][:]

                longitude = np.where(longitude > 180, longitude - 360, longitude)

                if len(valid_time.shape) == 2:
                    valid_time = valid_time.reshape(-1)
                time_units = dataset.variables['valid_time'].units
                valid_time = nc.num2date(valid_time, units=time_units)

                if len(variable_data.shape) == 4:
                    variable_data = variable_data.reshape(-1, variable_data.shape[2], variable_data.shape[3])
                else:
                    logger.error(f"{variable_short_name} has unexpected shape: {variable_data.shape}")
                    raise ValueError(f"{variable_short_name} has unexpected shape")

                ds = xr.Dataset(
                    {variable_short_name: (["time", "latitude", "longitude"], variable_data)},
                    coords={"longitude": longitude, "latitude": latitude, "time": valid_time}
                )

                ds = ds.resample(time="1D").max()
                ds = ds.squeeze(drop=True)
                ds[variable_short_name] *= 1e9

                return ds, temp_dir

        except Exception as e:
            logger.error(f"Failed to process NetCDF file {zip_file_path}: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def create_table(self, table_name: str, variable_name: str, metadata: MetaData) -> Table:
        """Define a table schema for the database."""
        return Table(
            table_name,
            metadata,
            Column('time', DateTime, primary_key=True, nullable=False),
            Column('latitude', Float, primary_key=True, nullable=False),
            Column('longitude', Float, primary_key=True, nullable=False),
            Column(variable_name.lower(), Float, nullable=False),
            Index(f'idx_{table_name}_time', 'time'),
        )

    def save_to_postgres(self, ds: xr.Dataset, table_name: str, variable_name: str, zip_file_path: str, temp_dir: str, extract_dir_name: str) -> None:
        """Save xarray Dataset to PostgreSQL database, drop and recreate table, insert in chunks, and delete files."""
        try:
            # Convert xarray Dataset to DataFrame
            df = ds.to_dataframe().reset_index()
            df.columns = [col.lower() for col in df.columns]

            # Convert cftime to datetime
            if df['time'].dtype == 'object' and any(isinstance(t, cftime.DatetimeGregorian) for t in df['time']):
                df['time'] = pd.to_datetime([t.isoformat() for t in df['time']])

            # Verify required columns
            expected_columns = {'time', 'latitude', 'longitude', variable_name.lower()}
            if not expected_columns.issubset(df.columns):
                logger.error(f"DataFrame missing required columns: {expected_columns - set(df.columns)}")
                raise ValueError(f"DataFrame missing required columns")

            # Round and aggregate
            df['latitude'] = df['latitude'].round(6)
            df['longitude'] = df['longitude'].round(6)
            df[variable_name.lower()] = df[variable_name.lower()].round(4)

            # Group by unique keys and take mean
            df = df.groupby(['time', 'latitude', 'longitude']).mean().reset_index()

            # Drop and recreate table
            metadata = MetaData()
            table = self.create_table(table_name, variable_name, metadata)
            try:
                with self.engine.begin() as conn:
                    table.drop(self.engine, checkfirst=True)
                    logger.info(f"Dropped table {table_name}")
                    table.create(self.engine)
                    logger.info(f"Created table {table_name}")
            except SQLAlchemyError as e:
                logger.error(f"Failed to drop/create table {table_name}: {e}")
                raise

            # Insert data into the table in chunks of 1000
            if not df.empty:
                with self.engine.begin() as conn:
                    df.to_sql(table_name, conn, if_exists='append', index=False, chunksize=1000)
                logger.info(f"Saved {len(df)} records to {table_name} in chunks of 1000")
            else:
                logger.info(f"No records to save to {table_name}")

            # Delete files after successful database storage
            try:
                # Delete zip file
                if os.path.exists(zip_file_path):
                    os.unlink(zip_file_path)
                    logger.info(f"Deleted zip file: {zip_file_path}")
                # Delete temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"Deleted temporary directory: {temp_dir}")
                # Delete extracted directory
                if os.path.exists(extract_dir_name):
                    shutil.rmtree(extract_dir_name, ignore_errors=True)
                    logger.info(f"Deleted extracted directory: {extract_dir_name}")
            except Exception as e:
                logger.warning(f"Failed to delete files for {table_name}: {e}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to save data to {table_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving data to {table_name}: {e}")
            raise

    def run(self):
        """Execute the CAMS data download, processing, and storage pipeline."""
        try:
            self.configure_cds_api()

            pm25_zip = 'pm25_download.zip'
            pm10_zip = 'pm10_download.zip'
            pm25_extract_dir = 'pm25_extracted'
            pm10_extract_dir = 'pm10_extracted'

            # Download data
            logger.info("Downloading PM10 data...")
            self.retrieve_variable('particulate_matter_10um', pm10_zip)
            logger.info("Downloading PM2.5 data...")
            self.retrieve_variable('particulate_matter_2.5um', pm25_zip)

            # Process data
            logger.info("Processing NetCDF PM10 data...")
            pm10_ds, pm10_temp_dir = self.process_netcdf(pm10_zip, 'pm10')
            logger.info("Processing NetCDF PM2.5 data...")
            pm25_ds, pm25_temp_dir = self.process_netcdf(pm25_zip, 'pm2p5')

            # Save to database and delete files
            logger.info("Saving PM10 data to PostgreSQL...")
            self.save_to_postgres(pm10_ds, "cams_pm10", "pm10", pm10_zip, pm10_temp_dir, pm10_extract_dir)
            logger.info("Saving PM2.5 data to PostgreSQL...")
            self.save_to_postgres(pm25_ds, "cams_pm25", "pm2p5", pm25_zip, pm25_temp_dir, pm25_extract_dir)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.engine.dispose()
            logger.info("Database engine disposed")

if __name__ == "__main__":
    cams = CamsDownload()
    cams.run()
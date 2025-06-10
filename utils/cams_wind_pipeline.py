import datetime
import cdsapi
import netCDF4 as nc
import numpy as np
import xarray as xr
import zipfile
import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Float, DateTime, Index
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
    """A class to handle downloading, processing, and storing CAMS wind data."""
    
    def __init__(self):
        """Initialize the CamsDownload class with database and CDS API configuration."""
        # Load environment variables
        load_dotenv()
        self.DB_USER = os.getenv('DB_USER', 'airqo')
        self.DB_PASS = os.getenv('DB_PASS')
        self.DB_HOST = os.getenv('DB_HOST', 'localhost')
        self.DB_PORT = os.getenv('DB_PORT','5432')
        self.DB_NAME = os.getenv('DB_NAME')
        self.CDS_API_KEY = os.getenv('CDS_API_KEY')

        if not all([self.DB_USER, self.DB_PASS, self.DB_HOST, self.DB_PORT, self.DB_NAME, self.CDS_API_KEY]):
            logger.error("Missing required environment variables. Please check .env file.")
            raise ValueError("Missing required environment variables.")

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

    def get_latest_forecast_hour(self):
        """Get the latest forecast time based on current UTC time rounded down to nearest hour slot."""
        now = datetime.datetime.now(datetime.timezone.utc)
        hour = now.hour % 12
        return "0" if hour == 0 else str(hour)

    def retrieve_variable(self, variable_name: str, output_zip_path: str) -> None:
        """Download a variable from the CAMS dataset."""
        try:
            c = cdsapi.Client()
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=2)
            date_range = f"{yesterday:%Y-%m-%d}/{today:%Y-%m-%d}"
            latest_time = self.get_latest_forecast_hour()

            logger.info(f"Retrieving {variable_name} data for {date_range} Lead Time {latest_time}.")
            c.retrieve(
                'cams-global-atmospheric-composition-forecasts',
                {
                    'date': date_range,
                    'type': ['forecast'],
                    'format': 'netcdf_zip',
                    'leadtime_hour': [latest_time],
                    'time': ['00:00', '12:00'],
                    'variable': variable_name,
                },
                output_zip_path
            )
            logger.info(f"Downloaded {variable_name} to {output_zip_path}")
        except Exception as e:
            logger.error(f"Failed to download {variable_name}: {e}")
            raise

    def process_netcdf(self, u_zip_path: str, v_zip_path: str) -> tuple[xr.Dataset, str]:
        """Process NetCDF files for u and v wind components and calculate wind speed and direction."""
        temp_dir = tempfile.mkdtemp()
        extract_dir = Path(temp_dir)
        try:
            # Extract u-component
            with zipfile.ZipFile(u_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir / "u_component")
            # Extract v-component
            with zipfile.ZipFile(v_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir / "v_component")

            u_files = list((extract_dir / "u_component").glob("*.nc"))
            v_files = list((extract_dir / "v_component").glob("*.nc"))
            if not u_files or not v_files:
                logger.error(f"No .nc files found in {extract_dir}")
                raise FileNotFoundError(f"No .nc files found in {extract_dir}")

            u_file, v_file = u_files[0], v_files[0]
            logger.info(f"Processing NetCDF files: {u_file}, {v_file}")

            with nc.Dataset(u_file) as u_dataset, nc.Dataset(v_file) as v_dataset:
                if 'u10' not in u_dataset.variables or 'v10' not in v_dataset.variables:
                    logger.error("Wind components u10 or v10 not found in NetCDF files")
                    raise KeyError("Wind components u10 or v10 not found")

                longitude = u_dataset.variables['longitude'][:]
                latitude = u_dataset.variables['latitude'][:]
                valid_time = u_dataset.variables['valid_time'][:]
                u_data = u_dataset.variables['u10'][:]
                v_data = v_dataset.variables['v10'][:]

                longitude = np.where(longitude > 180, longitude - 360, longitude)

                if len(valid_time.shape) == 2:
                    valid_time = valid_time.reshape(-1)
                time_units = u_dataset.variables['valid_time'].units
                valid_time = nc.num2date(valid_time, units=time_units)

                if len(u_data.shape) == 4:
                    u_data = u_data.reshape(-1, u_data.shape[2], u_data.shape[3])
                    v_data = v_data.reshape(-1, v_data.shape[2], v_data.shape[3])
                else:
                    logger.error(f"Wind data has unexpected shape: u_data {u_data.shape}, v_data {v_data.shape}")
                    raise ValueError("Wind data has unexpected shape")

                # Calculate wind speed and direction
                wind_speed = np.sqrt(u_data**2 + v_data**2)  # m/s
                wind_direction = (np.arctan2(v_data, u_data) * 180 / np.pi) % 360  # degrees

                ds = xr.Dataset(
                    {
                        "wind_speed": (["time", "latitude", "longitude"], wind_speed),
                        "wind_direction": (["time", "latitude", "longitude"], wind_direction)
                    },
                    coords={"longitude": longitude, "latitude": latitude, "time": valid_time}
                )

                ds = ds.resample(time="1D").mean()
                ds = ds.squeeze(drop=True)

                return ds, temp_dir

        except Exception as e:
            logger.error(f"Failed to process NetCDF files {u_zip_path}, {v_zip_path}: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def create_table(self, table_name: str, metadata: MetaData) -> Table:
        """Define a table schema for the database."""
        return Table(
            table_name,
            metadata,
            Column('time', DateTime, primary_key=True, nullable=False),
            Column('latitude', Float, primary_key=True, nullable=False),
            Column('longitude', Float, primary_key=True, nullable=False),
            Column('wind_speed', Float, nullable=False),
            Column('wind_direction', Float, nullable=False),
            Index(f'idx_{table_name}_time', 'time'),
        )

    def save_to_postgres(self, ds: xr.Dataset, table_name: str, u_zip_path: str, v_zip_path: str, temp_dir: str) -> None:
        """Save xarray Dataset to PostgreSQL database, drop and recreate table, insert in chunks, and delete files."""
        try:
            df = ds.to_dataframe().reset_index()
            df.columns = [col.lower() for col in df.columns]

            if df['time'].dtype == 'object' and any(isinstance(t, cftime.DatetimeGregorian) for t in df['time']):
                df['time'] = pd.to_datetime([t.isoformat() for t in df['time']])

            expected_columns = {'time', 'latitude', 'longitude', 'wind_speed', 'wind_direction'}
            if not expected_columns.issubset(df.columns):
                logger.error(f"DataFrame missing required columns: {expected_columns - set(df.columns)}")
                raise ValueError(f"DataFrame missing required columns")

            df['latitude'] = df['latitude'].round(6)
            df['longitude'] = df['longitude'].round(6)
            df['wind_speed'] = df['wind_speed'].round(4)
            df['wind_direction'] = df['wind_direction'].round(4)

            df = df.groupby(['time', 'latitude', 'longitude']).mean().reset_index()

            metadata = MetaData()
            table = self.create_table(table_name, metadata)
            try:
                with self.engine.begin() as conn:
                    table.drop(self.engine, checkfirst=True)
                    logger.info(f"Dropped table {table_name}")
                    table.create(self.engine)
                    logger.info(f"Created table {table_name}")
            except SQLAlchemyError as e:
                logger.error(f"Failed to drop/create table {table_name}: {e}")
                raise

            if not df.empty:
                with self.engine.begin() as conn:
                    df.to_sql(table_name, conn, if_exists='append', index=False, chunksize=1000)
                logger.info(f"Saved {len(df)} records to {table_name} in chunks of 1000")
            else:
                logger.info(f"No records to save to {table_name}")

            try:
                for zip_path in [u_zip_path, v_zip_path]:
                    if os.path.exists(zip_path):
                        os.unlink(zip_path)
                        logger.info(f"Deleted zip file: {zip_path}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"Deleted temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete files: {e}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to save data to {table_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving data to {table_name}: {e}")
            raise

    def run(self):
        """Execute the CAMS wind data download, processing, and storage pipeline."""
        try:
            self.configure_cds_api()

            u_zip = 'u_component_download.zip'
            v_zip = 'v_component_download.zip'

            # Download data
            logger.info("Downloading u-component data...")
            self.retrieve_variable('10m_u_component_of_wind', u_zip)
            logger.info("Downloading v-component data...")
            self.retrieve_variable('10m_v_component_of_wind', v_zip)

            # Process data
            logger.info("Processing NetCDF wind data...")
            wind_ds, wind_temp_dir = self.process_netcdf(u_zip, v_zip)

            # Save to database
            logger.info("Saving wind data to PostgreSQL...")
            self.save_to_postgres(wind_ds, "cams_wind", u_zip, v_zip, wind_temp_dir)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.engine.dispose()
            logger.info("Database engine disposed")

if __name__ == "__main__":
    cams = CamsDownload()
    cams.run()
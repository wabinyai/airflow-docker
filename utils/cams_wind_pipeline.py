import datetime
import os
import requests
import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import create_engine, MetaData, Table, Column, Float, DateTime, Index
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import logging
import glob
from typing import Optional

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NomadsWindDownload:
    def __init__(self, grib_filename: str = "gdas.t00z.pgrb2.0p25.f000"):
        """Initialize the NomadsWindDownload pipeline.

        Args:
            grib_filename (str): Name of the GRIB2 file to download/process.
        """
        load_dotenv()
        self.DB_USER = os.getenv('DB_USER', 'airqo')
        self.DB_PASS = os.getenv('DB_PASS')
        self.DB_HOST = os.getenv('DB_HOST', 'localhost')
        self.DB_PORT = os.getenv('DB_PORT', '5432')
        self.DB_NAME = os.getenv('DB_NAME')
        if not all([self.DB_USER, self.DB_PASS, self.DB_HOST, self.DB_PORT, self.DB_NAME]):
            raise ValueError("Missing required environment variables for database connection.")
        
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}",
            pool_size=5, max_overflow=10, pool_timeout=30
        )
        self.grib_filename = grib_filename
        self.idx_filename = f"{grib_filename}.idx"

    def download_grib2_file(self) -> None:
        """Download the GRIB2 file from NOAA NOMADS."""
        today = datetime.datetime.today().strftime("%Y%m%d")
        url = (
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gdas_0p25.pl"
            f"?dir=%2Fgdas.{today}%2F00%2Fatmos"
            f"&file={self.grib_filename}"
            f"&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on"
        )
        logger.info(f"Downloading GRIB2 data from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes
            with open(self.grib_filename, "wb") as f:
                f.write(response.content)
            logger.info(f"GRIB2 file downloaded successfully: {self.grib_filename}")
        except requests.RequestException as e:
            logger.error(f"Failed to download GRIB2 file: {e}")
            raise

    def process_grib2(self) -> pd.DataFrame:
        """Process the GRIB2 file and adjust longitudes to -180:180 range.

        Returns:
            pd.DataFrame: Processed wind speed and direction data.
        """
        logger.info(f"Processing GRIB2 file: {self.grib_filename}")
        try:
            # Open the GRIB2 file with xarray
            ds = xr.open_dataset(self.grib_filename, engine="cfgrib")
            
            # Shift longitudes from 0:360 to -180:180
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            ds = ds.sortby("longitude")  # Ensure ascending order

            # Compute wind speed and direction
            wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
            wind_direction = (270 - np.rad2deg(np.arctan2(ds['v10'], ds['u10']))) % 360
            
            # Add computed variables to dataset
            ds['wind_speed'] = wind_speed
            ds['wind_direction'] = wind_direction
            
            # Convert to DataFrame and reset index
            df = ds[['wind_speed', 'wind_direction']].to_dataframe().reset_index()
            
            # Keep only relevant columns
            df = df[['time', 'latitude', 'longitude', 'wind_speed', 'wind_direction']]
            
            # Round numerical values for consistency
            df['latitude'] = df['latitude'].round(6)
            df['longitude'] = df['longitude'].round(6)
            df['wind_speed'] = df['wind_speed'].round(4)
            df['wind_direction'] = df['wind_direction'].round(4)
            
            logger.info(f"Processed {len(df)} records from GRIB2 file. Preview:\n{df.head()}")
            return df
        except Exception as e:
            logger.error(f"Error processing GRIB2 file: {e}")
            raise

    def create_table(self, table_name: str, metadata: MetaData) -> Table:
        """Define the database table schema.

        Args:
            table_name (str): Name of the table.
            metadata (MetaData): SQLAlchemy metadata object.

        Returns:
            Table: SQLAlchemy Table object.
        """
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

    def save_to_postgres(self, df: pd.DataFrame, table_name: str) -> None:
        """Save the processed DataFrame to PostgreSQL.

        Args:
            df (pd.DataFrame): DataFrame containing wind data.
            table_name (str): Name of the target table.
        """
        try:
            # Group by to handle duplicates
            df = df.groupby(['time', 'latitude', 'longitude']).mean().reset_index()
            
            metadata = MetaData()
            table = self.create_table(table_name, metadata)
            
            with self.engine.begin() as conn:
                table.drop(self.engine, checkfirst=True)
                table.create(self.engine)
                df.to_sql(table_name, conn, if_exists='append', index=False, chunksize=1000, method='multi')
                logger.info(f"Saved {len(df)} records to {table_name}")
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during saving: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for filename in [self.grib_filename, self.idx_filename]:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    logger.info(f"Deleted file: {filename}")
                except OSError as e:
                    logger.warning(f"Failed to delete {filename}: {e}")

        # Delete any .*.idx files
        idx_pattern = "*.idx"
        idx_files = glob.glob(idx_pattern)
        for idx_file in idx_files:
            try:
                os.remove(idx_file)
                logger.info(f"Deleted index file: {idx_file}")
            except OSError as e:
                logger.warning(f"Failed to delete {idx_file}: {e}")

    def run(self) -> None:
        """Execute the full pipeline."""
        try:
            self.download_grib2_file()
            df = self.process_grib2()
            self.save_to_postgres(df, "nomads_wind")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            self.cleanup()
            self.engine.dispose()
            logger.info("Pipeline complete and database engine disposed.")

if __name__ == "__main__":
    pipeline = NomadsWindDownload()
    pipeline.run()
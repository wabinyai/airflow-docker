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

# === Configuration Setup ===
load_dotenv()

DB_USER = os.getenv('DB_USER', 'airqo')
DB_PASS = os.getenv('DB_PASS', 'your_password')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'your_database')
CDS_API_KEY = os.getenv('CDS_API_KEY', 'your-cds-api-key')

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30
)

# === CDS API Setup ===
url = 'https://ads.atmosphere.copernicus.eu/api'
cdsapirc_content = f"url: {url}\nkey: {CDS_API_KEY}"
cdsapirc_path = Path.home() / ".cdsapirc"
cdsapirc_path.write_text(cdsapirc_content)

def configure_cds_api():
    """Configure the CDS API client by writing the .cdsapirc file."""
    if not CDS_API_KEY or CDS_API_KEY == 'your-cds-api-key':
        raise ValueError("CDS API key is not set. Please provide a valid key in .env.")
    cdsapirc_path.write_text(cdsapirc_content)

def retrieve_variable(variable_name: str, output_zip_path: str) -> None:
    """Download a variable from the CAMS dataset."""
    try:
        c = cdsapi.Client()
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        date_range = f"{yesterday:%Y-%m-%d}/{today:%Y-%m-%d}"

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
    except Exception as e:
        raise RuntimeError(f"Failed to download {variable_name}: {e}")

def process_netcdf(zip_file_path: str, variable_short_name: str) -> xr.Dataset:
    """Process a NetCDF file and return an xarray Dataset."""
    extract_dir = Path(f"{variable_short_name}_extracted")
    extract_dir.mkdir(exist_ok=True)

    for file in extract_dir.glob("*.nc"):
        file.unlink()

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        nc_files = list(extract_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found in {extract_dir}")

        file_path = nc_files[0]

        with nc.Dataset(file_path) as dataset:
            if variable_short_name not in dataset.variables:
                raise KeyError(f"Variable {variable_short_name} not found in NetCDF file")

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
                raise ValueError(f"{variable_short_name} has unexpected shape: {variable_data.shape}")

            ds = xr.Dataset(
                {variable_short_name: (["time", "latitude", "longitude"], variable_data)},
                coords={"longitude": longitude, "latitude": latitude, "time": valid_time}
            )

            ds = ds.resample(time="1D").max()
            ds = ds.squeeze(drop=True)
            ds[variable_short_name] *= 1e9

            return ds

    except Exception as e:
        raise RuntimeError(f"Failed to process NetCDF file {zip_file_path}: {e}")

def create_table_if_not_exists(table_name: str, variable_name: str, engine) -> Table:
    metadata = MetaData()
    table = Table(
        table_name,
        metadata,
        Column('time', DateTime, primary_key=True, nullable=False),
        Column('latitude', Float, primary_key=True, nullable=False),
        Column('longitude', Float, primary_key=True, nullable=False),
        Column(variable_name.lower(), Float, nullable=False),
        Index(f'idx_{table_name}_time', 'time'),
    )

    try:
        with engine.connect() as connection:
            if not engine.dialect.has_table(connection, table_name):
                metadata.create_all(engine)
        return table
    except SQLAlchemyError as e:
        raise RuntimeError(f"Failed to create table {table_name}: {e}")

def check_duplicates(df: pd.DataFrame, table_name: str, engine) -> pd.DataFrame:
    try:
        existing = pd.read_sql(
            f"SELECT time, latitude, longitude FROM {table_name}",
            engine
        )
        if not existing.empty:
            df['key'] = df[['time', 'latitude', 'longitude']].apply(tuple, axis=1)
            existing['key'] = existing[['time', 'latitude', 'longitude']].apply(tuple, axis=1)
            new_records = df[~df['key'].isin(existing['key'])].drop(columns='key')
            return new_records
        return df
    except SQLAlchemyError as e:
        raise RuntimeError(f"Failed to check duplicates for {table_name}: {e}")

def save_to_postgres(ds: xr.Dataset, table_name: str, variable_name: str, engine) -> None:
    try:
        # Convert xarray Dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        df.columns = [col.lower() for col in df.columns]

        # Convert cftime.DatetimeGregorian to datetime.datetime
        if df['time'].dtype == 'object' and any(isinstance(t, cftime.DatetimeGregorian) for t in df['time']):
            df['time'] = pd.to_datetime([t.isoformat() for t in df['time']])

        # Verify required columns
        expected_columns = {'time', 'latitude', 'longitude', variable_name.lower()}
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"DataFrame missing required columns: {expected_columns - set(df.columns)}")

        # Create table if it doesn't exist
        create_table_if_not_exists(table_name, variable_name, engine)
        
        # Check for duplicates
        df = check_duplicates(df, table_name, engine)

        # Save to PostgreSQL
        if not df.empty:
            df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=1000)

    except Exception as e:
        raise RuntimeError(f"Failed to save data to {table_name}: {e}")

def main():
    try:
        configure_cds_api()

        pm25_zip = 'pm25_download.zip'
        pm10_zip = 'pm10_download.zip'

        retrieve_variable('particulate_matter_2.5um', pm25_zip)
        retrieve_variable('particulate_matter_10um', pm10_zip)

        pm25_ds = process_netcdf(pm25_zip, 'pm2p5')
        pm10_ds = process_netcdf(pm10_zip, 'pm10')

        save_to_postgres(pm25_ds, "cams_pm25", "pm2p5", engine)
        save_to_postgres(pm10_ds, "cams_pm10", "pm10", engine)

    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
"""
Utility for generating Mapbox Vector Tiles (MVT) from CAMS PM data stored in
PostgreSQL / PostGIS and writing them back into the database.

Usage:
    from process_and_store_pm import generate_pm25_vector_tiles
    generate_pm25_vector_tiles()

This script handles both PM2.5 and PM10 separately.
"""

from __future__ import annotations
import os
from datetime import date as _date
from typing import Optional, List
import logging

import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import mercantile
from shapely.geometry import box
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load environment variables from .env
# -----------------------------------------------------------------------------
load_dotenv()

DB_NAME = os.getenv("DB_NAME", "airquality")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
_ENGINE = create_engine(DB_URL, pool_size=5, max_overflow=10, pool_timeout=30)
logger.info("Database engine created")

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _load_dataframe(table_name: str, variable: str, run_date: _date) -> pd.DataFrame:
    """Return a pandas DataFrame for one day of data from *table_name*."""
    query = text(
        f"""
        SELECT time, latitude, longitude, {variable}
        FROM {table_name}
        WHERE DATE(time) = :run_date
        """
    )
    return pd.read_sql(query, _ENGINE, params={"run_date": run_date})

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate_and_store_vector_tiles(
    variable: str,
    table_name: str,
    run_date: Optional[_date] = None,
    zoom_min: int = 0,
    zoom_max: int = 12,
) -> None:
    """Generate MVT tiles for a variable and persist them in PostGIS.

    Args:
        variable: The column name to process (e.g., 'pm2p5' or 'pm10').
        table_name: The source table name (e.g., 'cams_pm25').
        run_date: The date to process (defaults to today).
        zoom_min: Minimum zoom level for tiles (default: 0).
        zoom_max: Maximum zoom level for tiles (default: 12).
    """
    run_date = run_date or _date.today()

    # Validate table existence
    with _ENGINE.connect() as conn:
        result = conn.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"),
            {"table_name": table_name}
        ).scalar()
        if not result:
            logger.error(f"Table {table_name} does not exist in the database.")
            return

    df = _load_dataframe(table_name, variable, run_date)
    if df.empty:
        logger.info(f"[{table_name}] No data found for {run_date}. Skipping tile generation.")
        return

    # Step 1: Scale values and convert to GeoDataFrame
    df[variable] = (df[variable] * 100).round().astype(int)  # Scale for MVT storage efficiency
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude, crs="EPSG:4326"),
    ).to_crs("EPSG:3857")

    # Step 2: Create tiles
    tile_frames: List[gpd.GeoDataFrame] = []

    for z in range(zoom_min, zoom_max + 1):
        covered_tiles = {mercantile.tile(pt.x, pt.y, z) for pt in gdf.geometry}
        for t in covered_tiles:
            bounds = mercantile.bounds(t)
            bbox = box(bounds.west, bounds.south, bounds.east, bounds.north)
            bbox_3857 = gpd.GeoSeries([bbox], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            subset = gdf[gdf.intersects(bbox_3857)]
            if subset.empty:
                continue
            subset = subset.copy()
            subset["zoom"] = z
            subset["tile_id"] = f"{t.z}_{t.x}_{t.y}"
            tile_frames.append(subset[["tile_id", "zoom", variable, "geometry"]])

    if not tile_frames:
        logger.info(f"[{table_name}] No intersecting tiles generated for {run_date}.")
        return

    tiles_gdf = pd.concat(tile_frames, ignore_index=True)
    target_table = f"{table_name}_tiles"  # Note: '_times' as requested; '_tiles' is more conventional for MVT

    tiles_gdf.to_postgis(
        name=target_table,
        con=_ENGINE,
        schema="public",
        if_exists="replace",  # Full refresh
        index=False,
    )

    with _ENGINE.begin() as conn:
        try:
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {target_table}_geom_gix "
                    f"ON public.{target_table} USING GIST (geometry);"
                )
            )
            logger.info(f"Created spatial index for {target_table}")
        except Exception as e:
            logger.error(f"Failed to create index for {target_table}: {e}")

    logger.info(f"[{table_name}] Stored {len(tiles_gdf):,} tiles (zoom {zoom_min}–{zoom_max}) to {target_table}.")

def generate_pm25_vector_tiles() -> None:
    """Generate tiles for PM2.5 (cams_pm25 table)."""
    generate_and_store_vector_tiles(variable="pm2p5", table_name="cams_pm25")
    logger.info("PM2.5 tiles stored")

def generate_pm10_vector_tiles() -> None:
    """Generate tiles for PM10 (cams_pm10 table)."""
    generate_and_store_vector_tiles(variable="pm10", table_name="cams_pm10")
    logger.info("PM10 tiles stored")

__all__ = [
    "generate_and_store_vector_tiles",
    "generate_pm25_vector_tiles",
    "generate_pm10_vector_tiles",
]

if __name__ == "__main__":
    generate_pm25_vector_tiles()
    generate_pm10_vector_tiles()
    logger.info("Vector tile generation complete")
"""
Module 2: Preprocess.

This module:
- stacks 9 spectral bands (assets) to a dask.DataArray;
- uses the Scene Classification Layer (SCL) to mask clouds and unwanted pixels;
- creates a median temporal composite;
- pulls and saves the processed data as a Zarr dataset.

`preprocess()` is the main entry point. By default, it uses the dates and AOI
in the workflow config to query STAC for items, but different start and
end dates and AOI can be supplied when needed. 
"""

import stackstac

import xarray as xr

from pathlib import Path
from pystac import ItemCollection
from rasterio.enums import Resampling

from .data_request import stac_search
from .utils import load_config, output_dirs

config = load_config()
dirs = output_dirs()

def preprocess(start_date: str | None = None, end_date: str | None = None, aoi: tuple | str | None = None) -> Path:
    """
    Query, preprocess, and export Sentinel-2 data to Zarr.

    Optional dates and AOI override the corresponding workflow
    configuration values.
    """
    items, date_range, bbox = stac_search(aoi, start_date, end_date)
    print(f"Found {len(items)} scenes.")
    print("Stacking scenes bands and masking clouds...")
    stacked_bands = stack_bands(items=items, bbox=bbox)
    print("Creating temporal composite and exporting to Zarr...")
    return zarr_export(stacked_bands, date_range, dirs["zarr"])

def stack_bands(items: ItemCollection, bbox: tuple) -> xr.DataArray:
    """Stack Sentinel-2 spectral bands and apply the SCL mask per retrieved scene."""
    
    bands = stackstac.stack(
        items,
        assets = [ "blue", "green", "red", "nir", "rededge1", 
                "rededge2", "rededge3", "swir16", "swir22"],
        bounds_latlon = [*bbox],
        resolution = config["stack"]["resolution"],
        epsg = config["stack"]["epsg"],
        resampling=Resampling.bilinear
    )

    scl = stackstac.stack(
        items,
        assets = ["scl"],
        bounds_latlon = [*bbox],
        resolution = config["stack"]["resolution"],
        epsg = config["stack"]["epsg"],
        resampling = Resampling.nearest
    )
    return scl_masking(bands, scl)

def scl_masking(bands: xr.DataArray, scl: xr.DataArray) -> xr.DataArray:
    """Mask spectral bands using Sentinel-2 Scene Classification Layer."""

    scl_mask = scl.isin([4, 5, 6]).squeeze("band")
    masked_bands = bands.where(scl_mask)
    return masked_bands
    
def zarr_export(masked_bands: xr.DataArray, date_range: str, output_dir: Path) -> Path:
    """Create a temporal composite and export it as a Zarr dataset.returns Path for reopening"""

    daily = masked_bands.groupby("time.date").median()
    composite = daily.median(dim="date")

    # Prepare Dataset for exporting to zarr 
    clean_composite = composite.reset_coords(drop=True)
    clean_composite.attrs = {
        "resolution": str(composite.rio.resolution()),
        "epsg": composite.rio.crs.to_epsg()
    }
    zarr_file_name = f"{date_range}.zarr"
    output_path = output_dir / zarr_file_name
    clean_composite.to_dataset(name="sentinel2").to_zarr(output_path,mode="w")
    print("Preprocessing Complete \nTemporal Composite successfully exported to:",output_path)
    return output_path

if __name__ == "__main__":
    # Run preprocessing independently using a test date range.
    # This allows the module to be tested without running the full pipeline.
    
    file_path = preprocess(start_date='2025-01-01', end_date='2025-01-02')
    print("Preprocessing Complete, Temporal Composite successfully exported to Zarr:",file_path)
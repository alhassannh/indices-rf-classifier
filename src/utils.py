"""Utility functions used across the workflow."""

from pathlib import Path
from pyproj import Transformer

import yaml

import geopandas as gpd
import xarray as xr



def load_config(path: str = "config.yaml") -> dict:
    """Load workflow configuration from the YAML file and."""
    
    with open(path, "r") as f:
        return yaml.safe_load(f)

def output_dirs() -> dict:
    """Create and return the workflow output directories."""

    output_dir = Path("outputs")
    directories = {
        "zarr": output_dir / "zarr",
        "models": output_dir / "models",
        "maps": output_dir / "maps",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories

def get_bbox(aoi: str | None = None) -> tuple:
    """Derive a WGS84 bbox from the supplied or configured AOI polygon."""

    config = load_config()
    aoi_config = config["aoi"]
    target_crs = "EPSG:4326"
    polygon = aoi or aoi_config["polygon"]["name"]
    polygon_crs = aoi_config["polygon"]["crs"]

    if polygon is None:
        if aoi_config["bbox"]:
            return aoi_config["bbox"]
        raise ValueError("No AOI supplied. Provide either a polygon or bbox.")
    gdf = gpd.read_file(polygon)
    if gdf.crs is None:
        if polygon_crs is None:
            raise ValueError("AOI polygon has no CRS. Specify its CRS in config.")
        gdf = gdf.set_crs(polygon_crs)
    gdf = gdf.to_crs(target_crs)
    return gdf.total_bounds

def open_zarr(path: str)-> xr.DataArray:
    """Open and return the Sentinel-2 dataset from a Zarr file."""

    return xr.open_zarr(path)['sentinel2'] 

def bbox_transform(bbox: tuple, raster_crs: str) -> tuple:
    """Convert bbox from EPSG:4326 to raster specified crs"""
    transformer = Transformer.from_crs(
        "EPSG:4326", raster_crs, always_xy=True
    )
    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])
    return minx, miny, maxx, maxy
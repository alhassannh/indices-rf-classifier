"""
Module 3: Features.

This module contains functions to:
- prepare data for classification and training samples selection;
- calculate and combine spectral indices with the original bands;
- select training samples using threshold-based class conditions;
- visualise the spatial distribution of selected samples.

Training samples are identified using multiple spectral indices and
hierarchical class conditions:

    All pixels
       │
       ├── Vegetation? → YES → Vegetation
       │                  NO ↓
       ├── Built-up?   → YES → Built-up
       │                  NO ↓
       ├── Water?      → YES → Water
       │                  NO ↓
       └── Bare soil?  → YES → Bare soil

Pixels already assigned to an earlier class are excluded from
subsequent classes.
"""

import geopandas as gpd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import rioxarray

from .utils import load_config, bbox_transform  

config = load_config()

def clip_dataset(zarr_dataset: xr.DataArray, aoi: str | tuple | None = None ) -> xr.DataArray:
    """Clip the dataset to the supplied or configured AOI polygon."""

    if aoi is not None:
        if isinstance(aoi, str):
            polygon = aoi
        else:
            raster_crs = zarr_dataset.attrs["epsg"]
            bbox = bbox_transform(aoi, raster_crs)
            return zarr_dataset.rio.write_crs(raster_crs).rio.clip_box(*bbox)
    else:
        polygon = config["aoi"]["polygon"]["name"]
    if not polygon:
        return zarr_dataset
    try:
        gdf = gpd.read_file(polygon)
    except Exception as e:
        print(e ,'Error occur while trying to read polygon')
    raster_crs = zarr_dataset.attrs["epsg"]
    if gdf.crs is None:
        raise ValueError("Polygon CRS is undefined. Provide an AOI polygon with a CRS.")
    if gdf.crs.to_epsg() != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    # Ensure the raster has CRS information for rioxarray clipping.
    zarr_dataset = zarr_dataset.rio.write_crs(raster_crs)
    return zarr_dataset.rio.clip(gdf.geometry, gdf.crs)

def calculate_indices(zarr_dataset: xr.DataArray) -> list[xr.DataArray]:
    """Calculate spectral indices from the dataset bands."""

    red = zarr_dataset.sel(band="red")
    blue = zarr_dataset.sel(band="blue")
    green = zarr_dataset.sel(band="green")
    nir = zarr_dataset.sel(band="nir")
    swir = zarr_dataset.sel(band="swir16")
    swir22 = zarr_dataset.sel(band="swir22")
    rededge1 = zarr_dataset.sel(band="rededge1")

    ndvi = ((nir - red) / (nir + red)).rename("NDVI")
    ndbi = ((swir - nir) / (swir + nir)).rename("NDBI")
    ndwi = ((green - nir) / (green + nir)).rename("NDWI")
    bsi = (((swir + red) - (nir + blue)) /((swir + red) + (nir + blue))).rename("BSI")
    mndwi = ((green - swir) / (green + swir)).rename("MNDWI")
    ndvi_re = ((rededge1 - red) / (rededge1 + red)).rename("NDVIre")
    ndti = ((swir - swir22) / (swir + swir22)).rename("NDTI")
    dbsi = (((swir - green) / (swir + green)) - ndvi).rename("DBSI")
    bui = (ndbi - ndvi).rename("BUI")

    return [ndbi, ndvi, ndwi, bsi, mndwi,ndvi_re, ndti, dbsi, bui]

def join_bands(zarr_dataset: xr.DataArray,indices: list[xr.DataArray]) -> xr.DataArray:
    """Combine dataset bands with calculated indices."""

    #Alignd the indices dims with the zarr_dataset dims
    aligned_indices = [index.expand_dims({"band": [index.name]}) for index in indices]

    return xr.concat([zarr_dataset, *aligned_indices],dim="band")

def indices_mask(index: xr.DataArray) -> xr.DataArray:
    """Mask index values outside the valid [-1, 1] range."""

    mask = (index >= -1) & (index <= 1)
    return index.where(mask)

def prepare_dataset(zarr_dataset: xr.DataArray) -> xr.DataArray:
    """Combine spectral bands and indices, then retain only pixels with valid features for model prediction."""

    indices = calculate_indices(zarr_dataset)
    dataset = join_bands(zarr_dataset, indices)
    dataset = dataset.stack(pixel_yx=("y", "x")).T
    mask = np.isfinite(dataset).all(dim="band")
    dataset = dataset.where(mask).dropna(dim="pixel_yx")

    return dataset

def prepare_samples(zarr_dataset: xr.DataArray ) -> xr.DataArray:
    """Prepare valid pixels with bands and spectral indices for sample selection."""

    dataset = clip_dataset(zarr_dataset )
    indices = calculate_indices(dataset)
    masked_indices = [indices_mask(index)for index in indices]
    joined_dataset = join_bands(dataset,masked_indices)
    data_samples = joined_dataset.stack(pixel_yx=("y", "x")).T
    mask = (data_samples.notnull().all(dim="band") & np.isfinite(data_samples).all(dim="band"))

    return (data_samples.where(mask).dropna(dim="pixel_yx"))

def select_samples(zarr_dataset: xr.DataArray ) -> tuple[xr.DataArray, xr.DataArray]:
    """Select and randomly sample pixels using configured thresholds, return selected pixels and their coords."""

    clean_data = prepare_samples(zarr_dataset )
    trsh = config["thresholds"]

    #Thresholds conditions 
    veg = (
        (clean_data.sel(band="NDVI") > trsh["vegetation"]["ndvi"]) &
        (clean_data.sel(band="NDVIre") > trsh["vegetation"]["ndvi_re"])
    )
    bup = (
        (~veg) &
        (clean_data.sel(band="NDVI") < trsh["builtup"]["ndvi"]) &
        (clean_data.sel(band="BUI") > trsh["builtup"]["bui"]) &
        (clean_data.sel(band="NDBI") > trsh["builtup"]["ndbi"])
    )
    wter = (
        (~veg) & (~bup) &
        (clean_data.sel(band="NDWI") > trsh["water"]["ndwi"]) &
        (clean_data.sel(band="MNDWI") > trsh["water"]["mndwi"])
    )
    bsoil = (
        (~veg) & (~bup) & (~wter) &
        (clean_data.sel(band="NDVI") < trsh["baresoil"]["ndvi"]) &
        (
            (clean_data.sel(band="DBSI") >= trsh["baresoil"]["dbsi"]) |
            (clean_data.sel(band="NDTI") > trsh["baresoil"]["ndti"])
        )
    )
    class_masks = {"Vegetation": veg, "Builtup": bup, "Water": wter, "Baresoil": bsoil }
    class_ids = { "Vegetation": 0, "Builtup": 1, "Water": 2, "Baresoil": 3 }

    samples = [
        clean_data.where(mask).rename(name).dropna(dim="pixel_yx", how="all")
        for name, mask in class_masks.items()
    ]
    if any(sample.sizes["pixel_yx"] == 0 for sample in samples):
        raise ValueError("One or more classes are not present in the scene. "
        "Configure a different scene for extracting training samples.")

    #Take the min between provided sample number and min valid sample per class
    sample_number = min(
        config["training"]["samples_per_class"],
        min(sample.sizes["pixel_yx"] for sample in samples)
    )
    print("Extracting Samples from the dataset")
    rng = np.random.default_rng(config['training']['random_state']) #Defining a random seed
    sampled = [
        (sample.isel(pixel_yx=rng.choice(sample.sizes["pixel_yx"],size=sample_number, replace=False)))
        for sample in samples
    ]
    training_samples_yx = xr.concat([xr.full_like(sample["pixel_yx"],class_ids[sample.name],dtype=int)
            .unstack("pixel_yx")
            for sample in sampled
        ], dim="class", join="outer"
    ).max("class").rename("Class")

    training_samples = [sample.reset_index("pixel_yx",drop = True)
        .assign_coords(Class=class_ids[sample.name])
        for sample in sampled
    ]

    training_samples = (xr.concat(training_samples,dim="samples")
        .stack(training_samples=("samples", "pixel_yx")).T
        .reset_index("training_samples", drop=True)
    )
    print(f"{training_samples.sizes['training_samples']} Training samples successfully extracted: ")

    return training_samples, training_samples_yx

def clusterd_window(samples_yx: xr.DataArray,clean_data: xr.DataArray,class_key: int) -> tuple[float, float, float, float]:
    """Return the AOI grid cell containing the most class samples."""

    x_min = clean_data.x.min().item()
    x_max = clean_data.x.max().item()
    y_min = clean_data.y.min().item()
    y_max = clean_data.y.max().item()

    # Divide the AOI into 20 equal grids
    x_bins = np.linspace(x_min, x_max, 21)
    y_bins = np.linspace(y_min, y_max, 21)
    #Construct the grids
    grids = [(
            x_bins[i], y_bins[j],
            x_bins[i + 1], y_bins[j + 1]
        )
        for i in range(20) for j in range(20)
    ]
    class_mask = samples_yx == class_key
    xy_counts = {
        grid: (
            class_mask &
            (samples_yx.x >= grid[0]) &
            (samples_yx.x <= grid[2]) &
            (samples_yx.y >= grid[1]) &
            (samples_yx.y <= grid[3])
        ).sum().item()
        for grid in grids
    }
    return max(xy_counts, key=xy_counts.get)

def visualise_samples(samples_yx: xr.DataArray,clean_data: xr.DataArray) -> None:
    """Visualise clustered sampled pixels over an RGB image."""

    class_ids = {
        0: "Vegetation",
        1: "Builtup",
        2: "Water",
        3: "Baresoil"
    }
    colors = ["lime", "red", "cyan", "yellow"]
    print("Visualising clustered training samples...")
    fig, axes = plt.subplots(2, 2,figsize=(8, 6))
    for ax, class_key in zip(axes.flat, class_ids):
        bbox = clusterd_window(samples_yx,clean_data,class_key)
        print (f"Plotting {class_ids[class_key]}..." )
        rgb = (
            clean_data.sel(band=["red", "green", "blue"])
            .sel(x=slice(bbox[0], bbox[2]),y=slice(bbox[3], bbox[1]))
            .transpose("y", "x", "band")
            .compute()
        )
        sample_window = (samples_yx.where(
                (samples_yx == class_key) &
                (samples_yx.x >= bbox[0]) &
                (samples_yx.x <= bbox[2]) &
                (samples_yx.y >= bbox[1]) &
                (samples_yx.y <= bbox[3]),
                drop=True
            ).compute()
        )
        ax.imshow(rgb.clip(0, 1),
            extent=[
                rgb.x.min().item(),
                rgb.x.max().item(),
                rgb.y.min().item(),
                rgb.y.max().item()
            ]
        )
        points = (sample_window == class_key)
        y_idx, x_idx = np.where(points)
        ax.scatter(
            sample_window.x.values[x_idx],
            sample_window.y.values[y_idx],
            c=colors[class_key],
            s=1
        )
        ax.set_axis_off()
        ax.set_title(class_ids[class_key])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from .utils import open_zarr, output_dirs
    
    zarr_dir = output_dirs()["zarr"]
    # Reconstruct dataset name for reopening
    dataset_name = (f"{config['stac']['start_date']}_"f"{config['stac']['end_date']}")  
    zarr_name = f"{dataset_name}.zarr" 

    try:
        zarr_dataset = open_zarr(zarr_dir / zarr_name)      
    except FileNotFoundError:
        from .preprocess import preprocess

        zarr_file_path = preprocess()
        zarr_dataset = open_zarr(zarr_file_path)

    _, samples_yx = select_samples(zarr_dataset)    
    clipped_dataset = clip_dataset(zarr_dataset)    
    visualise_samples(samples_yx, clipped_dataset)    
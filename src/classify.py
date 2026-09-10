import joblib
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier

from .preprocess import preprocess
from .features import clip_dataset, prepare_dataset
from .utils import output_dirs, load_config, open_zarr


dirs = output_dirs()
config = load_config()

def load_model() -> RandomForestClassifier:
    """Load the saved land-cover Random Forest model."""

    model_path = dirs["models"] / f"{config["model"]["name"]}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run the training stage before classification."
        )
    return joblib.load(model_path)
def classify(zarr_dataset: xr.DataArray, model: RandomForestClassifier) -> xr.DataArray:
    """Classify prepared pixels using the trained model."""

    prepared_dataset = prepare_dataset(zarr_dataset)
    X_full = prepared_dataset.compute().values
    classified = model.predict(X_full)
    return xr.DataArray(
        classified,
        dims="pixel_yx",
        coords={"pixel_yx": prepared_dataset.pixel_yx}
    ).unstack("pixel_yx")

def visualise_classification(classification: xr.DataArray, output_path: Path |None=None) -> None:
    """Display and save the classified land-cover map."""

    cmap = ListedColormap(["green", "red", "blue", "saddlebrown"])
    fig, ax = plt.subplots(figsize=(8, 6))
    image = classification.plot.imshow(
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=3,
        add_colorbar=True,
        cbar_kwargs={
        "ticks": [0, 1, 2, 3],
        "shrink": 0.5,
        "aspect": 25
    }
    )
    image.colorbar.ax.set_yticklabels(["Vegetation", "Built-up", "Water", "Bare soil"])
    ax.set_title("Land Cover Classification Map", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_axis_off()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print("Map successfully saved to", output_path)
    plt.show()

def classify_scene(
    start_date: str | None = None,
    end_date: str |None = None, 
    aoi: tuple | str | None = None, 
    zarr_dataset: xr.DataArray | None =None,
    dataset_name : str | None = None
    ) -> None:
    """Retrieve, preprocess, and classify a date range or a dataset using an already saved model."""

    model = load_model()
    output_path = None
    if zarr_dataset is None:
        zarr_path = preprocess(start_date, end_date, aoi)
        zarr_dataset = open_zarr(zarr_path)
        date_range = Path(zarr_path).stem
        output_path = dirs["maps"] / f"{date_range}.png" 
    elif dataset_name is not None:
        output_path = dirs["maps"] / f"{Path(dataset_name).stem}.png"   
    dataset = clip_dataset(zarr_dataset, aoi)
    print("Classifying pixels...")
    classification = classify(dataset, model)
    print("Creating classification map...")
    visualise_classification(classification, output_path)

if __name__ == "__main__":
    
    model = load_model()
    dataset_name = (f"{config['stac']['start_date']}_"f"{config['stac']['end_date']}")
    zarr_name = f"{dataset_name}.zarr"
    try:
        zarr_dataset = open_zarr(dirs["zarr"] / zarr_name)
    except FileNotFoundError:
        zarr_file_path = preprocess()
        zarr_dataset = open_zarr(zarr_file_path)
    dataset = clip_dataset(zarr_dataset)
    visualise_classification(classify(dataset,model))
    
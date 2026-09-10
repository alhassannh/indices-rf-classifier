"""
Module 4: Train.

This module:
- trains and evaluates a Random Forest classifier using the selected samples;
- saves the trained model and its metadata;
- uses the configured training parameters and class thresholds for everything.
"""

import json
import joblib

import xarray as xr
import datetime as dt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from .utils import output_dirs, load_config, get_bbox 

CLASS_NAMES = ["Vegetation", "Builtup", "Water", "Baresoil"]

config = load_config()
dirs = output_dirs()

def train_model(training_samples: xr.DataArray) -> tuple[RandomForestClassifier, float, dict]:
    """Train and evaluate a Random Forest classifier."""

    X = training_samples.values
    y = training_samples.coords["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size = config["training"]["train_size"],
        random_state = config["training"]["random_state"],
        stratify=y
    )
    model = RandomForestClassifier(
        n_estimators = config["training"]["n_estimators"],
        random_state = config["training"]["random_state"]
    )
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        target_names=CLASS_NAMES
    )
    return model, accuracy, report

def save_model(model: RandomForestClassifier,accuracy: float,report: dict) -> None:
    """Save the trained model and metadata."""
    
    bbox = get_bbox()
    model_name = config["model"]["name"]
    model_path = dirs["models"] / f"{model_name}.joblib"
    metadata_path = dirs["models"] / f"{model_name}.json"

    joblib.dump(model, model_path)
    metadata = {
        "model_name": model_name,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "aoi_bbox": [*bbox],
        "training_parameters": config["training"],
        "thresholds": config["thresholds"],
        "accuracy": accuracy,
        "classification_report": report
    }
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)
    print(f"Models successfully saved to {model_path}")
    
if __name__ == "__main__":
    from .features import select_samples
    from .utils import open_zarr

    zarr_dir = dirs["zarr"] 
    dataset_name = (f"{config['stac']['start_date']}_"f"{config['stac']['end_date']}")
    zarr_name = f"{dataset_name}.zarr"
    try:
        zarr_dataset = open_zarr(zarr_dir / zarr_name)
    except FileNotFoundError:
        from .preprocess import preprocess
        
        zarr_file_path = preprocess()
        zarr_dataset = open_zarr(zarr_file_path)       
    training_samples, _ = select_samples(zarr_dataset)
    model, accuracy, report = train_model(training_samples)
    save_model(model,accuracy,report)
    print (f"Model and metadata successfully saved to:{dirs['models']}")

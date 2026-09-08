"""
Single entry point for the Indices Random Forest classification workflow.

    python main.py --stage search --start DATE --end DATE --aoi AOI
        Search the STAC catalog using the configured dates/AOI or optional
        command-line overrides.

    python main.py --stage preprocess --start DATE --end DATE --aoi AOI
        Search for imagery, preprocess it, and save the composite as Zarr,
        using either the configured values or supplied overrides.

    python main.py --stage samples
        Open the configured dataset, select training samples, and visualise them.

    python main.py --stage train
        Open the configured dataset, select samples, train the Random Forest,
        and save the model.

    python main.py --stage classify --start DATE --end DATE [--aoi AOI]
        Classify an existing matching dataset or preprocess the requested
        date range before classification.

    python main.py --stage all
        Run sample selection, training, and classification on the configured dataset.
"""

import argparse
import warnings

from src.data_request import stac_search
from src.preprocess import preprocess
from src.features import clip_dataset, select_samples, visualise_samples
from src.train import train_model, save_model
from src.classify import classify_scene
from src.utils import load_config, output_dirs, open_zarr

warnings.filterwarnings("ignore")

def get_dataset(config = None, dataset_name= None):
    """Open the configured Zarr dataset, preprocessing it if necessary."""
    dirs = output_dirs()
    if dataset_name is not None:
        try:
            return open_zarr(dirs["zarr"]/dataset_name)
        except FileNotFoundError:
            return None
    if config is None:
        raise ValueError("Either config or dataset name must be provided.")
    dataset_name = f"{config['stac']['start_date']}_{config['stac']['end_date']}.zarr"
    zarr_path = dirs["zarr"] /dataset_name
    try:
        return open_zarr(zarr_path)
    except FileNotFoundError:
        zarr_path = preprocess()
        return open_zarr(zarr_path) 
def run_search(start = None, end = None, aoi = None):
    """Search and display available STAC scenes."""
    items, _, _ = stac_search(start=start, end=end, aoi=aoi)
    for item in items:
        print(
            f"Scene ID: {item.id}, "
            f"Acquisition Date: {item.datetime.date()}, "
            f"Cloud Cover: {item.properties.get('eo:cloud_cover')}"
        )
def run_preprocess(start = None, end = None, aoi = None):
    """Create the configured Zarr dataset."""
    path = preprocess(start_date=start, end_date= end, aoi=aoi)
    print("Composite Image successfully exported to Zarr:", path)
def run_samples(config, dataset=None):
    """Select and visualise training samples."""
    if dataset is None:
        dataset = get_dataset(config)
    _, samples_yx = select_samples(dataset)
    clipped_dataset = clip_dataset(dataset)
    visualise_samples(samples_yx, clipped_dataset)
def run_training(config, dataset=None):
    """Select training samples and train the Random Forest model."""
    if dataset is None:
        dataset = get_dataset(config)
    training_samples, _ = select_samples(dataset)
    model, accuracy, report = train_model(training_samples)
    save_model(model, accuracy, report)
    print(f"Model successfully saved to {output_dirs()['models']}")
def run_classify(start=None, end=None, aoi=None, dataset=None):
    """Classify a requested date range using the saved model."""
    if dataset is not None:
        classify_scene(zarr_dataset=dataset, aoi=aoi)
    else:
        dataset_name = f"{start}_{end}.zarr"
        dataset = get_dataset(dataset_name = dataset_name)
        if dataset is not None:
            classify_scene(zarr_dataset= dataset, aoi=aoi, dataset_name = dataset_name)
        else:
            classify_scene(start_date=start, end_date=end, aoi=aoi)

def main():
    parser = argparse.ArgumentParser(description="Indices RF Classifier")

    parser.add_argument(
        "--stage",choices=["search","preprocess","samples","train","classify","all"],
        default="all"
    )
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--aoi", nargs="+")
    args = parser.parse_args()
    config = load_config()

    if args.aoi:
        if len(args.aoi) == 4:
            args.aoi = tuple(map(float, args.aoi))
        elif len(args.aoi) == 1:
            args.aoi = args.aoi[0]
        else:
            parser.error("--aoi must be a polygon path or four bbox coordinates")

    if args.stage == "search":
        run_search(args.start, args.end, args.aoi)
    elif args.stage == "preprocess":
        run_preprocess(args.start, args.end, args.aoi)
    elif args.stage == "samples":
        run_samples(config)
    elif args.stage == "train":
        run_training(config)
    elif args.stage == "classify":
        if not args.start or not args.end:
            dataset = get_dataset(config=config)
            run_classify(dataset=dataset, aoi= args.aoi)
        else: 
            run_classify(start=args.start, end=args.end, aoi=args.aoi)
    elif args.stage == "all":
        dataset = get_dataset(config=config)
        run_samples(config,  dataset)
        run_training(config, dataset)
        run_classify(dataset=dataset)

if __name__ == "__main__":
    main()
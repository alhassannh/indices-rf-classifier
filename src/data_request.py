"""
Module 1: Data Request.

This module:
- connects to the STAC catalog;
- searches for Sentinel-2 L2A items using the configured AOI and date range;
- applies the configured cloud-cover filter;
- returns the matching STAC items as an ItemCollection.
"""

from pystac import ItemCollection
from pystac_client import Client

from .utils import get_bbox, load_config

def stac_search(aoi: str | tuple | None = None ,start: str |None = None, end : str| None = None) -> tuple[ItemCollection,str,tuple]:
    """
    Search the STAC catalog for Sentinel-2 L2A items.

    Parameters:
        aoi: Optional polygon path or bbox to override the configured AOI.
        start: Optional start date to override the configured start date.
        end: Optional end date to override the configured end date.

    Returns:
        The matching STAC items, the resolved date range, and the
        bbox used for the search.
    """

    stac_config = load_config()["stac"]
    start_date = start or stac_config['start_date']
    end_date = end or stac_config['end_date']
    date_range = f"{start_date}_{end_date}"
    if aoi:
        if isinstance(aoi, str):
            bbox = get_bbox(aoi)
        else:
            bbox = aoi
    else:
        bbox = get_bbox() 
    catalog = Client.open(stac_config["url"])

    try:
        search = catalog.search(
            collections=[stac_config["collection"]],
            datetime=(f"{start_date}/{end_date}"),
            query={"eo:cloud_cover": {"lt": stac_config["max_cloud_cover"]}},
            bbox=bbox
        )
        items = search.item_collection()
    except Exception as e:
        raise RuntimeError(f"STAC search failed: {e}") from e
    if len(items) == 0:
        raise ValueError("No scenes found for the specified AOI, date range, and cloud-cover threshold.")

    print(f"Search returned {len(items)} scenes covering AOI.")
    return items, date_range, bbox


if __name__ == "__main__":

    collection , _, _ = stac_search()
    for item in collection:
        print(
            f"Scene ID: {item.id}, "
            f"Acquisition Date: {item.datetime.date()}, "
            f"Cloud Cover: {item.properties.get('eo:cloud_cover')}"
        )
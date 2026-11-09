"""
Collect monthly Sentinel-1 VV/VH composites for multiple German cities.

Input:
    - Earth Engine project ID
    - City coordinates
    - Start year and month
    - Number of months to collect

Output:
    data/raw/sentinel1/<city>_<year>_<month>.tif

Each GeoTIFF contains two bands:
    Band 1: VV backscatter in dB
    Band 2: VH backscatter in dB

Requirements:
    earthengine-api
    geemap
    rasterio
    numpy
"""

from pathlib import Path
import calendar
import re
import time

import ee
import geemap
import numpy as np
import rasterio
import yaml


# ---------------------------------------------------------
# DEFAULT PROJECT SETTINGS
# ---------------------------------------------------------

EARTH_ENGINE_PROJECT = "lulc-sentinel1-project-512811"

# Output folder relative to the repository root.
DEFAULT_OUTPUT_DIR = Path("data") / "raw" / "sentinel1"

# Use a 2.5 km buffer around each city centre.
DEFAULT_BUFFER_METRES = 2500

# Export resolution in metres.
DEFAULT_SCALE_METRES = 10

# Required Sentinel-1 polarization bands.
SENTINEL_BANDS = ["VV", "VH"]


# ---------------------------------------------------------
# CITY CONFIGURATION
# ---------------------------------------------------------

CITY_CONFIG_PATH = Path("configs/cities.yaml")


def load_cities():
    """Load city coordinates from the project YAML configuration."""
    with CITY_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return {
        city["name"]: (float(city["lon"]), float(city["lat"]))
        for city in config["cities"]
    }


def get_utm_crs(longitude, latitude):
    """Return the appropriate WGS84 UTM CRS for a location."""
    zone = int((longitude + 180) // 6) + 1
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


GERMAN_CITIES = load_cities()




# ---------------------------------------------------------
# EARTH ENGINE INITIALIZATION
# ---------------------------------------------------------

def initialize_earth_engine(project_id=EARTH_ENGINE_PROJECT):
    """
    Initialize Google Earth Engine.

    Authenticate interactively first if this environment
    has not been authenticated before.
    """
    try:
        ee.Initialize(project=project_id)
        print("Earth Engine initialized successfully.")

    except Exception as exc:
        raise RuntimeError(
            "Could not initialize Earth Engine. "
            "Check authentication and the project ID. "
            "If needed, run ee.Authenticate() in a notebook "
            "cell and then rerun this function."
        ) from exc


# ---------------------------------------------------------
# DATE HELPERS
# ---------------------------------------------------------

def get_month_dates(year, month):
    """
    Return the inclusive start and exclusive end dates
    for a calendar month.
    """
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    start_date = f"{year:04d}-{month:02d}-01"

    if month == 12:
        end_date = f"{year + 1:04d}-01-01"
    else:
        end_date = f"{year:04d}-{month + 1:02d}-01"

    return start_date, end_date


def generate_month_sequence(start_year, start_month, number_of_months):
    """
    Generate year/month pairs, including the starting month.
    """
    if number_of_months < 1:
        raise ValueError("number_of_months must be at least 1")

    if not 1 <= start_month <= 12:
        raise ValueError("start_month must be between 1 and 12")

    months = []

    year = start_year
    month = start_month

    for _ in range(number_of_months):
        months.append((year, month))

        month += 1

        if month > 12:
            month = 1
            year += 1

    return months


# ---------------------------------------------------------
# CITY GEOMETRY
# ---------------------------------------------------------

def create_study_area(
    longitude,
    latitude,
    buffer_metres=DEFAULT_BUFFER_METRES,
):
    """
    Create a rectangular study area around a city centre.

    Earth Engine Point coordinates must be:
        [longitude, latitude]
    """
    point = ee.Geometry.Point([longitude, latitude])

    study_area = point.buffer(buffer_metres).bounds()

    return study_area


# ---------------------------------------------------------
# SENTINEL-1 MONTHLY COMPOSITE
# ---------------------------------------------------------

def create_monthly_composite(
    study_area,
    year,
    month,
    scale_metres=DEFAULT_SCALE_METRES,
):
    """
    Create a monthly median Sentinel-1 VV/VH composite.

    Returns:
        composite: Earth Engine image or None if no imagery
                   is available for the selected month.
        image_count: Number of matching Sentinel-1 scenes.
    """
    start_date, end_date = get_month_dates(year, month)

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(study_area)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.eq("instrumentMode", "IW")
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .select(SENTINEL_BANDS)
    )

    image_count = collection.size().getInfo()

    if image_count == 0:
        print(
            f"No Sentinel-1 scenes for "
            f"{year}-{month:02d}."
        )
        return None, 0

    # COPERNICUS/S1_GRD backscatter is already represented
    # in dB. Do not apply log10 conversion again.
    composite = (
        collection
        .median()
        .select(SENTINEL_BANDS)
        .clip(study_area)
        .toFloat()
    )

    return composite, image_count


# ---------------------------------------------------------
# FILE-NAME HELPER
# ---------------------------------------------------------

def safe_filename(name):
    """
    Convert a city name into a filesystem-friendly name.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned.strip("_").lower()


# ---------------------------------------------------------
# EXPORT ONE MONTH
# ---------------------------------------------------------

def export_monthly_composite(
    composite,
    output_path,
    study_area,
    scale_metres=DEFAULT_SCALE_METRES,
    crs=None,
):
    """
    Export a two-band Earth Engine image to a local GeoTIFF.

    Returns:
        Path to the downloaded file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"File already exists; skipping: {output_path}")
        return output_path

    # geemap downloads the Earth Engine image to this
    # environment. Large exports may need a larger timeout
    # or a Google Drive export workflow.
    geemap.ee_export_image(
        composite,
        filename=str(output_path),
        scale=scale_metres,
        crs=crs,
        region=study_area,
        file_per_band=False,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Export did not create the expected file: {output_path}"
        )

    if output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Export produced an empty file: {output_path}"
        )

    # Verify that the output is a readable raster.
    with rasterio.open(output_path) as dataset:
        if dataset.count != 2:
            raise ValueError(
                f"Expected two bands (VV and VH), "
                f"but found {dataset.count} in {output_path}"
            )

        if dataset.width == 0 or dataset.height == 0:
            raise ValueError(
                f"Raster has invalid dimensions: {output_path}"
            )

    print(f"Saved: {output_path}")

    return output_path


# ---------------------------------------------------------
# COLLECT ONE CITY
# ---------------------------------------------------------

def collect_city_monthly_data(
    city_name,
    longitude,
    latitude,
    start_year,
    start_month,
    number_of_months=12,
    output_dir=DEFAULT_OUTPUT_DIR,
    buffer_metres=DEFAULT_BUFFER_METRES,
    scale_metres=DEFAULT_SCALE_METRES,
    delay_seconds=1,
):
    """
    Collect monthly Sentinel-1 composites for one city.

    Returns a list of dictionaries describing the results.
    Months without available imagery are recorded as skipped.
    """
    output_dir = Path(output_dir)

    study_area = create_study_area(
        longitude=longitude,
        latitude=latitude,
        buffer_metres=buffer_metres,
    )

    month_sequence = generate_month_sequence(
        start_year=start_year,
        start_month=start_month,
        number_of_months=number_of_months,
    )

    results = []

    print(f"\nCity: {city_name}")
    print(f"Coordinates: {longitude}, {latitude}")
    print(f"Months to process: {len(month_sequence)}")

    for year, month in month_sequence:
        print(f"\nProcessing {city_name}: {year}-{month:02d}")

        try:
            composite, image_count = create_monthly_composite(
                study_area=study_area,
                year=year,
                month=month,
                scale_metres=scale_metres,
            )

            if composite is None:
                results.append(
                    {
                        "city": city_name,
                        "year": year,
                        "month": month,
                        "image_count": 0,
                        "status": "skipped_no_imagery",
                        "path": None,
                    }
                )
                continue

            filename = (
                f"{safe_filename(city_name)}_"
                f"{year}_{month:02d}.tif"
            )

            output_path = output_dir / filename

            exported_path = export_monthly_composite(
                composite=composite,
                output_path=output_path,
                study_area=study_area,
                scale_metres=scale_metres,
                crs=get_utm_crs(longitude, latitude),
            )

            results.append(
                {
                    "city": city_name,
                    "year": year,
                    "month": month,
                    "image_count": image_count,
                    "status": "success",
                    "path": str(exported_path),
                }
            )

        except Exception as exc:
            print(
                f"ERROR processing {city_name} "
                f"{year}-{month:02d}: {exc}"
            )

            results.append(
                {
                    "city": city_name,
                    "year": year,
                    "month": month,
                    "image_count": None,
                    "status": "error",
                    "path": None,
                    "error": str(exc),
                }
            )

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    successful = sum(
        result["status"] == "success"
        for result in results
    )

    skipped = sum(
        result["status"] == "skipped_no_imagery"
        for result in results
    )

    failed = sum(
        result["status"] == "error"
        for result in results
    )

    print(f"\nCompleted: {city_name}")
    print(f"Successful exports: {successful}")
    print(f"Months without imagery: {skipped}")
    print(f"Errors: {failed}")

    return results


# ---------------------------------------------------------
# MAIN PUBLIC FUNCTION
# ---------------------------------------------------------

def collect_monthly_data(
    city_name="Heidelberg",
    start_year=2025,
    start_month=1,
    number_of_months=12,
    output_dir=None,
    project_id=EARTH_ENGINE_PROJECT,
    buffer_metres=DEFAULT_BUFFER_METRES,
    scale_metres=DEFAULT_SCALE_METRES,
):
    """
    Main function imported by the project notebook.

    Example:
        results = collect_monthly_data(
            city_name="Heidelberg",
            start_year=2025,
            start_month=1,
            number_of_months=12,
        )

    The default city list is defined in GERMAN_CITIES.
    """
    initialize_earth_engine(project_id=project_id)

    if city_name not in GERMAN_CITIES:
        available = ", ".join(GERMAN_CITIES.keys())
        raise ValueError(
            f"Unknown city: {city_name}. "
            f"Available cities: {available}"
        )

    longitude, latitude = GERMAN_CITIES[city_name]

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    results = collect_city_monthly_data(
        city_name=city_name,
        longitude=longitude,
        latitude=latitude,
        start_year=start_year,
        start_month=start_month,
        number_of_months=number_of_months,
        output_dir=output_dir,
        buffer_metres=buffer_metres,
        scale_metres=scale_metres,
    )

    return results


# ---------------------------------------------------------
# OPTIONAL: COLLECT ALL 15 CITIES
# ---------------------------------------------------------

def collect_all_cities(
    start_year=2025,
    start_month=1,
    number_of_months=12,
    output_dir=None,
    project_id=EARTH_ENGINE_PROJECT,
):
    """
    Collect monthly imagery for all configured German cities.

    Warning:
        15 cities x 12 months can require many downloads.
        Test one city first before running the full collection.
    """
    initialize_earth_engine(project_id=project_id)

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    all_results = []

    for city_name, (longitude, latitude) in GERMAN_CITIES.items():
        city_results = collect_city_monthly_data(
            city_name=city_name,
            longitude=longitude,
            latitude=latitude,
            start_year=start_year,
            start_month=start_month,
            number_of_months=number_of_months,
            output_dir=output_dir,
        )

        all_results.extend(city_results)

    return all_results


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    # Start with Heidelberg for January-December 2025.
    results = collect_monthly_data(
        city_name="Heidelberg",
        start_year=2025,
        start_month=1,
        number_of_months=12,
    )

    print("\nCollection summary:")

    for item in results:
        print(item)
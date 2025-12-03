# pip install -r requirements.txt

# File imports
import config
import src.data.ais_downloader as ais_downloader
import src.data.ais_filtering as ais_filtering
import src.data.ais_reader as ais_reader
import src.data.ais_to_parquet as ais_to_parquet

# Setup data

AIS_DATA_FOLDER = config.AIS_DATA_FOLDER
VERBOSE_MODE = config.VERBOSE_MODE                          # Whether to print verbose output

START_DATE = config.START_DATE                              # Start date for data downloading
END_DATE   = config.END_DATE                                # End date for data downloading

AIS_DATA_NAME = config.AIS_DATA_FOLDER                      # Name of the folder to store AIS data
DELETE_DOWNLOADED_CSV = config.DELETE_DOWNLOADED_CSV        # Whether to delete raw downloaded CSV files after processing

VESSEL_AIS_CLASS = config.VESSEL_AIS_CLASS                  # AIS classes of vessels to include

REMOVE_ZERO_SOG_VESSELS = config.REMOVE_ZERO_SOG_VESSELS    # Whether to remove vessels with zero Speed Over Ground
SOG_IN_MS = config.SOG_IN_MS                                # If True, SOG is in meters/second; if False, SOG is in knots
SOG_MIN_KNOTS = config.SOG_MIN_KNOTS                        # Minimum SOG in knots
SOG_MAX_KNOTS = config.SOG_MAX_KNOTS                        # Maximum SOG in knots

BBOX = config.BBOX                                          # Bounding Box to prefilter AIS data
POLYGON_COORDINATES = config.POLYGON_COORDINATES            # Polygon coordinates for filter Area of Interest AOI in (lon,lat) 
COLUMNS_TO_DROP = config.COLUMNS_TO_DROP                    # Columns to drop from AIS data

# Import for data processing
from tqdm import tqdm
import gc
from pathlib import Path
import pandas as pd
from datetime import date, timedelta

def main_data():
    # --- Create paths ---
    folder_path = Path(AIS_DATA_FOLDER)
    folder_path.mkdir(parents=True, exist_ok=True)
    csv_folder_path = folder_path / "csv"
    csv_folder_path.mkdir(parents=True, exist_ok=True)
    parquet_folder_path = folder_path / "parquet"
    parquet_folder_path.mkdir(parents=True, exist_ok=True)

    file_port_locations = folder_path / "port_locodes.csv"


    # --- If you want to download all csv files before, uncomment the line below ---
    # ais_downloader.download_multiple_ais_data(START_DATE, END_DATE, folder_path)

    # --- Build the schedule of download string dates ---
    dates = ais_downloader.get_work_dates(START_DATE, END_DATE, csv_folder_path, filter=False)

    # --- Iterate with tqdm and download, unzip and delete ---
    for day in tqdm(dates, desc=f"Processing data", unit="file" ):
        tag = f"{day:%Y-%m}" if day < date.fromisoformat("2024-03-01") else f"{day:%Y-%m-%d}"
        print(f"\nProcessing date: {tag}")

        # --- Download one day ---
        csv_path = ais_downloader.download_one_ais_data(day, csv_folder_path)
        
        # --- Load CSV into DataFrame ---
        df_raw = ais_reader.read_single_ais_df(csv_path, BBOX, columns_to_drop=COLUMNS_TO_DROP, verbose=VERBOSE_MODE)
        # --- Optionally delete the downloaded CSV file ---
        if DELETE_DOWNLOADED_CSV: csv_path.unlink(missing_ok=True)
        
        # --- Filter and split ---
        # Filter AIS data, keeping Class A and Class B by default,
        
        # --- Filter and split ---
        # Filter AIS data, keeping Class A and Class B by default,
        df_filtered = ais_filtering.filter_ais_df(
            df_raw,                                               # raw AIS DataFrame
            polygon_coords=POLYGON_COORDINATES,                   # polygon coordinates for precise AOI filtering
            allowed_mobile_types=VESSEL_AIS_CLASS,                # vessel AIS class filter
            apply_polygon_filter=True,                            # keep polygon filtering enabled boolean
            remove_zero_sog_vessels=REMOVE_ZERO_SOG_VESSELS,      # use True/False to enable/disable 90% zero-SOG removal
            output_sog_in_ms=SOG_IN_MS,                           # convert SOG from knots in m/s (default) boolean
            sog_min_knots=SOG_MIN_KNOTS,                          # min SOG in knots to keep (None to disable)
            sog_max_knots=SOG_MAX_KNOTS,                          # max SOG in knots to keep (None to disable) 
            port_locodes_path=file_port_locations,                # path to port locodes CSV
            exclude_ports=True,                                   # exclude port areas boolean 
            verbose=VERBOSE_MODE,                                 # verbose mode boolean
        )
        
        # Free df_raw memory
        del df_raw
        gc.collect()

        # --- Parquet conversion ---
        # Save to Parquet by MMSI
        ais_to_parquet.save_by_mmsi(
            df_filtered,                                             # filtered AIS DataFrame 
            verbose=VERBOSE_MODE,                                    # verbose mode boolean
            output_folder=parquet_folder_path                        # output folder path
        )

        # Free df_filtered memory
        del df_filtered
        gc.collect()

        
if __name__ == "__main__":
    main_data()
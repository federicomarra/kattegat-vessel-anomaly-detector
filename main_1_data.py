# Main data script

# File imports
import config
import src.data.ais_downloader as ais_downloader
import src.data.ais_filtering as ais_filtering
import src.data.ais_reader as ais_reader
import src.data.ais_to_parquet as ais_to_parquet

# Library imports
from pathlib import Path
from datetime import date
from tqdm import tqdm
import gc

def main_data():
    """
    Main function to orchestrate the downloading, filtering, and conversion of AIS data to Parquet format.
    This function performs the following high-level steps:
    1.  **Configuration Loading**: Reads settings from `config.py`, including date ranges, bounding boxes, polygon coordinates, and file paths.
    2.  **Directory Setup**: Ensures necessary directories for raw CSVs and processed Parquet files exist.
    3.  **Data Pipeline Execution**: Iterates through the specified date range to:
        *   Download daily AIS data CSV files via `ais_downloader`.
        *   Load the raw CSV data into a DataFrame, applying an initial bounding box filter.
        *   Optionally delete the raw CSV to save space.
        *   Apply rigorous filtering via `ais_filtering`, including:
            *   Polygon-based Area of Interest (AOI) filtering.
            *   Vessel class filtering (Class A/B).
            *   Speed Over Ground (SOG) filtering (min/max thresholds and zero-SOG removal).
            *   Port area exclusion using a database of port locations.
        *   Convert the filtered data into Parquet format, partitioned by MMSI, using `ais_to_parquet`.
    4.  **Memory Management**: Explicitly deletes DataFrames and triggers garbage collection to manage memory usage during the loop.
    """
    
    # --- 1. CONFIGURATION ---
    # Read configuration from config.py
    VERBOSE_MODE = config.VERBOSE_MODE                          # Whether to print verbose output

    START_DATE = config.START_DATE                              # Start date for data downloading
    END_DATE   = config.END_DATE                                # End date for data downloading

    DELETE_DOWNLOADED_CSV = config.DELETE_DOWNLOADED_CSV        # Whether to delete raw downloaded CSV files after parquet conversion

    BBOX = config.BBOX                                          # Bounding Box to prefilter AIS data
    POLYGON_COORDINATES = config.POLYGON_COORDINATES           # Polygon coordinates for filter Area of Interest AOI in (lon,lat)


    # --- 2. PATHS ---
    folder_path = Path(config.AIS_DATA_FOLDER)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    csv_folder_path = folder_path / config.AIS_DATA_FOLDER_CSV_SUBFOLDER
    csv_folder_path.mkdir(parents=True, exist_ok=True)
    
    parquet_folder_path = folder_path / config.AIS_DATA_FOLDER_PARQUET_SUBFOLDER
    parquet_folder_path.mkdir(parents=True, exist_ok=True)

    file_port_locations = folder_path / config.FILE_PORT_LOCATIONS


    # --- 3. DOWNLOAD, FILTER AND SAVE INTO PARQUET ---
    # --- Build the schedule of download string dates ---
    dates = ais_downloader.get_work_dates(START_DATE, END_DATE, csv_folder_path, filter=False)

    # --- Iterate with tqdm and download, unzip and delete ---
    for day in tqdm(dates, desc=f"Processing data", unit="file" ):
        tag = f"{day:%Y-%m}" if day < date.fromisoformat("2024-03-01") else f"{day:%Y-%m-%d}"
        print(f"\nProcessing date: {tag}")

        # --- Download one day ---
        csv_path = ais_downloader.download_one_ais_data(day, csv_folder_path)
        
        # --- Load CSV into DataFrame ---
        df_raw = ais_reader.read_single_ais_df(csv_path, BBOX, columns_to_drop=config.COLUMNS_TO_DROP, verbose=VERBOSE_MODE)
        # --- Optionally delete the downloaded CSV file ---
        if DELETE_DOWNLOADED_CSV: csv_path.unlink(missing_ok=True)
        
        # --- Filter and split ---
        # Filter AIS data, keeping Class A and Class B by default,
        
        # --- Filter and split ---
        # Filter AIS data, keeping Class A and Class B by default,
        df_filtered = ais_filtering.filter_ais_df(
            df_raw,                                               # raw AIS DataFrame
            polygon_coords=POLYGON_COORDINATES,                   # polygon coordinates for precise AOI filtering
            allowed_mobile_types=config.VESSEL_AIS_CLASS,                # vessel AIS class filter
            apply_polygon_filter=True,                            # keep polygon filtering enabled boolean
            remove_zero_sog_vessels=config.REMOVE_ZERO_SOG_VESSELS,      # use True/False to enable/disable 90% zero-SOG removal
            output_sog_in_ms=config.SOG_IN_MS,                           # convert SOG from knots in m/s (default) boolean
            sog_min_knots=config.SOG_MIN_KNOTS,                          # min SOG in knots to keep (None to disable)
            sog_max_knots=config.SOG_MAX_KNOTS,                          # max SOG in knots to keep (None to disable) 
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
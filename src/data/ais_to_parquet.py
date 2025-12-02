from pathlib import Path
from typing import Optional, Union
import shutil

import pandas as pd
import pyarrow
import pyarrow.parquet

def segment_ais_tracks(
    df: pd.DataFrame,
    min_track_len: Optional[int] = 300,              # datapoints
    min_track_duration_sec: Optional[int] = 60 * 60, # 1 hour
    max_time_gap_sec: Optional[int] = 30,            # 30 seconds
    max_segment_duration_sec: Optional[int] = 12 * 60 * 60,  # 12 hours
    verbose: bool = False,
) -> pd.DataFrame:
    
    """
    Segment AIS vessel tracks using optional filters based on track length,
    duration, time gaps, and maximum segment duration.

    The function can enforce:
      - minimum number of AIS points per track
      - minimum track duration (seconds)
      - maximum allowed time gap between consecutive messages
      - maximum allowed duration of any segment (e.g., split long trajectories)

    Any parameter set to ``None`` disables that rule. If all are ``None``,
    the full dataset is returned with a single segment per MMSI.

    Steps
    -----
    1) Filter tracks per MMSI using length/duration (if enabled)
    2) Sort by (MMSI, Timestamp)
    3) Create segments using:
         - gaps ≥ max_time_gap_sec
         - segment duration ≥ max_segment_duration_sec
    4) Apply the same filters at (MMSI, Segment) level
    5) Add a YYYY-MM-DD ``Date`` column

    Example
    -------
    >>> df = pd.DataFrame({
    ...   "MMSI": ["111","111","111"],
    ...   "Timestamp": pd.to_datetime([
    ...       "2023-01-01 00:00:00",
    ...       "2023-01-01 00:00:20",
    ...       "2023-01-01 12:30:00",  # gap → new segment
    ...   ]),
    ...   "SOG": [10, 11, 12],
    ... })

    >>> out = segment_ais_tracks(df, max_time_gap_sec=60)
    >>> out[["MMSI", "Timestamp", "Segment"]]
       MMSI             Timestamp           Segment
    0  111   2023-01-01 00:00:00          0
    1  111   2023-01-01 00:00:20          0
    2  111   2023-01-01 12:30:00          1
    """

    df = df.copy()

    # ---------------- Basic checks ----------------
    required_cols = ["MMSI", "Timestamp", "SOG"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"segment_ais_tracks: required columns missing: {missing}")

    df["MMSI"] = df["MMSI"].astype(str)

    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        raise TypeError("segment_ais_tracks: 'Timestamp' must be a datetime dtype")

    if verbose:
        print(
            f"[segment_ais_tracks] Starting with {len(df):,} rows, "
            f"{df['MMSI'].nunique():,} unique vessels"
        )

    # ---------- helper: track filter ----------
    def track_filter(g: pd.DataFrame) -> bool:
        """
        Returns True if group passes length and duration constraints.
        Any constraint set to None is ignored.
        """
        # length criterion
        len_ok = True
        if min_track_len is not None:
            len_ok = len(g) > min_track_len

        # duration criterion
        time_ok = True
        if min_track_duration_sec is not None:
            dt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds()
            time_ok = dt >= min_track_duration_sec

        return len_ok and time_ok

    # ---------- 1) Filter per MMSI ----------
    if (min_track_len is not None) or (min_track_duration_sec is not None):
        df = df.groupby("MMSI", group_keys=False).filter(track_filter)

        if verbose:
            print(
                f"[segment_ais_tracks] After MMSI-level filter: {len(df):,} rows, "
                f"{df['MMSI'].nunique():,} vessels"
            )

    # ---------- 2) Sort ----------
    df = df.sort_values(["MMSI", "Timestamp"])

    # ---------- 3) Compute Segment IDs ----------
    def compute_segments(ts: pd.Series) -> pd.Series:
        """
        Compute segment IDs for a single MMSI based on:
        - time gaps >= max_time_gap_sec
        - cumulative duration >= max_segment_duration_sec

        If both max_time_gap_sec and max_segment_duration_sec are None,
        a single segment (0) is returned for all rows.
        """
        ts = ts.sort_values()

        # No segmentation: single segment per MMSI
        if max_time_gap_sec is None and max_segment_duration_sec is None:
            return pd.Series(0, index=ts.index, dtype="int64")

        seg_ids = pd.Series(0, index=ts.index, dtype="int64")

        # Initialize segment bookkeeping
        current_seg = 0
        seg_start_time = ts.iloc[0]
        prev_time = ts.iloc[0]

        # First index is already in segment 0
        for idx, t in ts.iloc[1:].items():
            new_segment = False

            # Condition 1: gap-based segmentation
            if max_time_gap_sec is not None:
                if (t - prev_time).total_seconds() >= max_time_gap_sec:
                    new_segment = True

            # Condition 2: duration-based segmentation (e.g. > 12 hours)
            if (not new_segment) and (max_segment_duration_sec is not None):
                if (t - seg_start_time).total_seconds() >= max_segment_duration_sec:
                    new_segment = True

            if new_segment:
                current_seg += 1
                seg_start_time = t  # reset segment start

            seg_ids.at[idx] = current_seg
            prev_time = t

        return seg_ids

    df["Segment"] = df.groupby("MMSI")["Timestamp"].transform(compute_segments)

    # ---------- 4) Filter per (MMSI, Segment) ----------
    if (min_track_len is not None) or (min_track_duration_sec is not None):
        df = df.groupby(["MMSI", "Segment"], group_keys=False).filter(track_filter)

    df = df.reset_index(drop=True)

    if verbose:
        n_segments = df[["MMSI", "Segment"]].drop_duplicates().shape[0]
        print(
            f"[segment_ais_tracks] After segment-level filter: {len(df):,} rows, "
            f"{n_segments:,} segments"
        )

    # ---------- 5) Add Date ----------
    df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

    return df


def save_by_mmsi(
    df: pd.DataFrame,
    verbose: bool = False,
    output_folder: Union[Path, str] = "ais_data_parquet",
) -> Path:
    """
    Write AIS data to a partitioned Parquet dataset.

    The output directory is **always** "ais_data_parquet".
    If it does not exist, it will be created.

    IMPORTANT
    ---------
    This function is *overwrite-safe* for the partitions present in `df`.
    For each unique (MMSI, Date, Segment) combination in `df`, the existing
    partition directory is removed before writing new data. This avoids
    accumulating multiple parquet files for the same segment when rerunning
    the pipeline for the same date/file.

    Expected columns
    ----------------
    df must contain:
    - "MMSI"    (string-like)
    - "Date"    (string, e.g. "2025-11-05")
    - "Segment" (int)

    Partition layout
    ----------------
    ais_data_parquet/
        MMSI=123456789/
            Date=2025-11-05/
                Segment=0/part-*.parquet
                Segment=1/part-*.parquet
        MMSI=987654321/
            Date=2025-11-06/
                Segment=0/part-*.parquet

    Parameters
    ----------
    df : pd.DataFrame
        Segmented AIS DataFrame containing "MMSI", "Date", "Segment".
    verbose : bool, optional
        If True, print the output path and some info.

    Returns
    -------
    Path
        Path to the root parquet dataset folder ("ais_data_parquet").

    Examples
    --------
    >>> df_seg = segment_ais_tracks(df_filt)
    >>> out_root = save_by_mmsi(df_seg, verbose=True)
    """
    df = df.copy()

    required_cols = ["MMSI", "Date", "Segment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f" save_by_mmsi: required columns missing: {missing}")

    df["MMSI"] = df["MMSI"].astype(str)
    
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Remove existing segment folders for (MMSI, Date, Segment) in df
    # ------------------------------------------------------------------
    partitions = df[["MMSI", "Date", "Segment"]].drop_duplicates()

    for _, row in partitions.iterrows():
        mmsi_val = row["MMSI"]
        date_val = row["Date"]
        seg_val = row["Segment"]

        seg_dir = (
            out_path
            / f"MMSI={mmsi_val}"
            / f"Date={date_val}"
            / f"Segment={seg_val}"
        )

        if seg_dir.exists():
            if verbose:
                print(f" [save_by_mmsi] Removing existing partition: {seg_dir}")
            shutil.rmtree(seg_dir)

    # ------------------------------------------------------------------
    # Write new dataset (append is fine now that partitions are cleaned)
    # ------------------------------------------------------------------
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=str(out_path),
        partition_cols=["MMSI", "Date", "Segment"],
    )

    if verbose:
        print(f" [save_by_mmsi] Parquet dataset written/appended at: {out_path.resolve()}")

    return out_path

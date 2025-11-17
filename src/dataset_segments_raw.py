import torch
from torch.utils.data import Dataset
# pip install pyarrow
import pyarrow.dataset as ds
import numpy as np
from pathlib import Path


def to_seq(pdf):
    # Build base features [lat, lon, sog, cog, rot] as a tensor
    if "Timestamp" in pdf.columns:
        pdf = pdf.sort_values("Timestamp")

    # COG 
    cog = pdf["COG"].astype(float)
    if cog.isna().any():
        cog = cog.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    cog_rad = np.deg2rad(cog.to_numpy(np.float32))

    lat = pdf["Latitude"].astype("float32").to_numpy()
    lon = pdf["Longitude"].astype("float32").to_numpy()
    sog = pdf["SOG"].astype("float32").to_numpy() # in m/s, already converted in ais_to_parquet.py

    x = np.stack([lat, lon, sog, np.cos(cog_rad), np.sin(cog_rad)], axis=-1)  # (T, 5)
    return torch.from_numpy(x)  # (T, 5)

# Dataset of raw AIS segments stored in Parquet format
class AISSegmentRaw(Dataset):
    def __init__(self, root_dir: str, min_length: int=1):
        self.min_length = int(min_length)
        root = Path(root_dir)
        
        if not root.exists():
            raise FileNotFoundError(f"Dataset root directory {root} not found.")

        # find all the couples of (MMSI, Segment) in the dataset
        # and save the path of the Parquet files (Segment)
        paths = []
        for mdir in sorted(root.glob("MMSI=*")):
            mmsi = mdir.name.split("=", 1)[1]
            for sdir in sorted(mdir.glob("Segment=*")):
                seg_str = sdir.name.split("=", 1)[1]
                try:
                    seg = int(seg_str)
                except ValueError:
                    # skip non-integer segment directories
                    continue
                has_parquet = any(sdir.glob("*.parquet"))
                if has_parquet:
                    paths.append((mmsi, seg, sdir))

        if not paths:
            raise RunTimeError(f"No valid MMSI/Segment Parquet files found in {root_dir}.")

        # order for MMSI, Segment
        self.paths = sorted(paths, key=lambda x: (x[0], x[1]))

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index: int):
        mmsi, seg, seg_dir = self.paths[index]

        # read just the needed segment Parquet file
        dset = ds.dataset(str(seg_dir), format="parquet")
        pdf = dset.to_table().to_pandas()

        # filter eventual empty segments
        required_cols = ["Latitude", "Longitude", "SOG", "COG"]
        if not required_cols.issubset(set(pdf.columns)):
            raise KeyError(f"Missing columns {required_cols - set(pdf.columns)} in segment MMSI={mmsi}, Segment={seg}.")

        x = to_seq(pdf)  # (T, 5)
        if x.size(0) < self.min_length:
            raise IndexError(f"Segment MMSI={mmsi}, Segment={seg} length {x.size(0)} is less than min_length {self.min_length}.")
        
        return {"x": x}         # variable length sequence of features
        
"""
Utilities for test-set evaluation and visual inspection.

This module assumes you already have:
- a trained model (LSTMAutoencoderWithShipType or similar),
- a preprocessed test dataframe df_seq_test with at least:
    ["Sequence", "ShipTypeID", "Segment_nr", "MMSI"],
- a loss function sequence_loss_fn(recon_batch, x_batch) -> (B,),
- a df_test_errors produced by your evaluation pipeline, with columns:
    ["Segment_nr", "ShipTypeID", "MMSI", "reconstruction_error"].

It provides:
  1) build_test_predictions_df(...)  -> DF with real & predicted sequences
  2) plot_error_histograms(...)      -> error distribution plots (global + by category)
  3) plot_best_and_worst_segments(...) -> top/bottom segments with per-timestep error
  4) denormalize_and_save_predictions(...) -> uses inspection_utils.denormalize_predictions
  5) create_interactive_map(...)     -> uses inspection_utils.save_interactive_html
"""

import os
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import config
from src.utils.inspection_utils import (
    denormalize_predictions,
    save_interactive_html,
)

# Base output directory for plots
PLOT_PATH = config.PLOT_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def plot_histogram(
    scores: np.ndarray,
    title: str,
    filename: str,
    bins: int = 50,
) -> None:
    """Plot a simple histogram of reconstruction errors."""
    ensure_dir(os.path.dirname(filename) or ".")
    plt.figure()
    plt.hist(scores, bins=bins)
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------------------------------------------------------------------
# 1) Build post-evaluation predictions DataFrame for test
# ---------------------------------------------------------------------------

def build_test_predictions_df(
    model: torch.nn.Module,
    df_seq_test: pd.DataFrame,
    device: torch.device,
    seq_loss_fn,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a predictions dataframe for the test set.

    The resulting dataframe has columns:
      - Segment_nr
      - MMSI
      - ShipTypeID
      - Sequence_real: list of lists (T, F)
      - Sequence_pred: list of lists (T, F)
      - recon_error:   scalar reconstruction error (mean over T and F)

    Parameters
    ----------
    model : torch.nn.Module
        Trained autoencoder model.
    df_seq_test : pd.DataFrame
        Test dataframe with at least:
        ["Sequence", "ShipTypeID", "Segment_nr", "MMSI"].
        Sequences are assumed normalized, shape (T, F).
    device : torch.device
        Device (cpu or cuda).
    seq_loss_fn : callable
        Function taking (recon_batch, x_batch) and returning a tensor (B,)
        with per-sequence reconstruction error.
    save_path : str, optional
        If provided, the resulting dataframe is saved to this path
        (format inferred from file extension: .parquet, .csv, ...).

    Returns
    -------
    df_pred : pd.DataFrame
        Predictions dataframe as described above.
    """
    model.eval()
    records: List[Dict[str, Any]] = []

    for idx, row in df_seq_test.iterrows():
        seq = np.array(row["Sequence"], dtype=np.float32)  # (T, F)
        x = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, T, F)

        ship_type_id = int(row["ShipTypeID"])
        st = torch.tensor([ship_type_id], dtype=torch.long, device=device)

        with torch.no_grad():
            recon_batch, _ = model(x, st)                  # (1, T, F)
            seq_error_batch = seq_loss_fn(recon_batch, x)  # (1,)
            seq_error = seq_error_batch.mean().item()

        recon = recon_batch.squeeze(0).cpu().numpy()       # (T, F)
        x_np = seq                                        # already numpy

        seg_nr = int(row["Segment_nr"])
        mmsi = row.get("MMSI", None)

        records.append(
            {
                "Segment_nr": seg_nr,
                "MMSI": mmsi,
                "ShipTypeID": ship_type_id,
                "Sequence_real": x_np.tolist(),
                "Sequence_pred": recon.tolist(),
                "recon_error": seq_error,
            }
        )

    df_pred = pd.DataFrame(records)

    # Optional saving for future inspection
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path) or ".")
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".parquet":
            df_pred.to_parquet(save_path, index=False)
        elif ext in (".csv", ".txt"):
            df_pred.to_csv(save_path, index=False)
        else:
            # default to parquet if extension is unknown
            df_pred.to_parquet(save_path, index=False)

    return df_pred


# ---------------------------------------------------------------------------
# 2) Error distributions on the test set
# ---------------------------------------------------------------------------

def plot_error_histograms(
    df_test_errors: pd.DataFrame,
    out_dir: Optional[str] = None,
    by_shiptype: bool = True,
    by_shiptype_name: bool = False,
) -> None:
    """
    Plot error distributions for the test set.

    Parameters
    ----------
    df_test_errors : pd.DataFrame
        Dataframe with at least:
        ["reconstruction_error", "ShipTypeID"].
        Optionally may contain "ShipType" (string category).
    out_dir : str, optional
        Output directory; if None, uses config.PLOT_PATH.
    by_shiptype : bool
        If True, plots separate histograms per ShipTypeID.
    by_shiptype_name : bool
        If True and column "ShipType" exists, plots per ship type name.
    """
    if out_dir is None:
        out_dir = PLOT_PATH
    ensure_dir(out_dir)

    # Global histogram
    scores = df_test_errors["reconstruction_error"].values
    plot_histogram(
        scores,
        title="Test reconstruction error (all)",
        filename=os.path.join(out_dir, "test_error_hist_all.png"),
    )

    # By ShipTypeID
    if by_shiptype and "ShipTypeID" in df_test_errors.columns:
        for st_id, group in df_test_errors.groupby("ShipTypeID"):
            scores_st = group["reconstruction_error"].values
            plot_histogram(
                scores_st,
                title=f"Test reconstruction error - ShipTypeID={st_id}",
                filename=os.path.join(out_dir, f"test_error_hist_shiptype_{st_id}.png"),
            )

    # By ShipType (string label)
    if by_shiptype_name and "ShipType" in df_test_errors.columns:
        for st_name, group in df_test_errors.groupby("ShipType"):
            safe_name = str(st_name).replace("/", "_").replace(" ", "_")
            scores_st = group["reconstruction_error"].values
            plot_histogram(
                scores_st,
                title=f"Test reconstruction error - ShipType={st_name}",
                filename=os.path.join(out_dir, f"test_error_hist_shiptype_{safe_name}.png"),
            )


# ---------------------------------------------------------------------------
# 2-bis) Feature-wise error distributions (e.g. Latitude, Longitude, SOG, COG)
# ---------------------------------------------------------------------------

def compute_feature_errors_per_segment(
    df_pred: pd.DataFrame,
    feature_indices: Sequence[int],
) -> pd.DataFrame:
    """
    Compute per-feature reconstruction error per segment.

    For each segment:
      - real_seq:  (T, F)
      - pred_seq:  (T, F)
      - feature_error[f] = mean_t (real[t,f] - pred[t,f])^2

    Returns a long-form dataframe with columns:
      - Segment_nr
      - feature_idx
      - feature_error
    """
    records: List[Dict[str, Any]] = []

    for row in df_pred.itertuples():
        real_seq = np.array(row.Sequence_real, dtype=np.float32)  # (T, F)
        pred_seq = np.array(row.Sequence_pred, dtype=np.float32)  # (T, F)

        sq_err = (real_seq - pred_seq) ** 2                      # (T, F)
        per_feat = sq_err.mean(axis=0)                           # (F,)

        seg_nr = int(row.Segment_nr)

        for fidx in feature_indices:
            records.append(
                {
                    "Segment_nr": seg_nr,
                    "feature_idx": int(fidx),
                    "feature_error": float(per_feat[fidx]),
                }
            )

    df_feat_err = pd.DataFrame(records)
    return df_feat_err


def plot_feature_error_histograms_from_pred(
    df_pred: pd.DataFrame,
    feature_indices: Sequence[int],
    feature_names: Dict[int, str],
    out_dir: str,
    bins: int = 50,
) -> None:
    """
    Plot feature-wise error distributions across segments.

    For each feature in feature_indices, computes a per-segment MSE and
    plots a histogram of these values.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Dataframe produced by build_test_predictions_df.
    feature_indices : list[int]
        Indices of features to analyze (e.g. [0, 1, 2, 3]).
    feature_names : dict[int, str]
        Map from feature index to readable name.
    out_dir : str
        Directory where the plots will be saved.
    bins : int
        Number of bins for histograms.
    """
    ensure_dir(out_dir)
    df_feat_err = compute_feature_errors_per_segment(df_pred, feature_indices)

    for fidx in feature_indices:
        fname = feature_names.get(fidx, f"feat_{fidx}")
        mask = df_feat_err["feature_idx"] == fidx
        scores = df_feat_err.loc[mask, "feature_error"].values

        filename = os.path.join(out_dir, f"feature_error_hist_{fname}.png")
        plot_histogram(
            scores,
            title=f"Feature-wise reconstruction error - {fname}",
            filename=filename,
            bins=bins,
        )



# ---------------------------------------------------------------------------
# 3) Plot best/worst segments with per-timestep error
# ---------------------------------------------------------------------------

def plot_segment_with_error(
    real_seq: np.ndarray,
    pred_seq: np.ndarray,
    feature_indices: Sequence[int],
    feature_names: Dict[int, str],
    title: str,
    filename: str,
) -> None:
    """
    Plot real vs predicted features for a single segment, plus per-timestep error.

    Parameters
    ----------
    real_seq : np.ndarray
        Real sequence, shape (T, F).
    pred_seq : np.ndarray
        Reconstructed sequence, shape (T, F).
    feature_indices : list[int]
        Indices of features to plot (e.g. [0, 1, 2, 3]).
    feature_names : dict[int, str]
        Map from feature index to readable name.
    title : str
        Plot title.
    filename : str
        Path where the figure will be saved.
    """
    ensure_dir(os.path.dirname(filename) or ".")

    T = real_seq.shape[0]
    t = np.arange(T)
    n_feats = len(feature_indices)

    # Per-timestep error (mean squared error over features)
    per_timestep_mse = ((real_seq - pred_seq) ** 2).mean(axis=1)

    plt.figure(figsize=(9, 2.5 * (n_feats + 1)))

    # Feature plots
    for i, fidx in enumerate(feature_indices):
        plt.subplot(n_feats + 1, 1, i + 1)
        plt.plot(t, real_seq[:, fidx], label="real")
        plt.plot(t, pred_seq[:, fidx], label="recon", linestyle="--")
        plt.ylabel(feature_names.get(fidx, f"feat_{fidx}"))
        if i == 0:
            plt.title(title)
        if i == n_feats - 1:
            plt.xlabel("timestep")
        plt.legend()

    # Error per timestep
    plt.subplot(n_feats + 1, 1, n_feats + 1)
    plt.plot(t, per_timestep_mse, label="MSE per timestep", color="red")
    plt.ylabel("MSE")
    plt.xlabel("timestep")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_best_and_worst_segments(
    df_pred: pd.DataFrame,
    feature_indices: Sequence[int],
    feature_names: Dict[int, str],
    out_dir: Optional[str] = None,
    num_examples: int = 3,
) -> None:
    """
    Plot best and worst segments from df_pred (built by build_test_predictions_df).

    For each of the top `num_examples` best (lowest recon_error) and worst
    (highest recon_error) segments:
      - plot selected features over time (real vs recon),
      - plot per-timestep MSE.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Dataframe with:
          ["Segment_nr", "Sequence_real", "Sequence_pred", "recon_error", ...].
    feature_indices : list[int]
        Indices of features to plot.
    feature_names : dict[int, str]
        Map from feature index to readable name.
    out_dir : str, optional
        Base output directory; if None, uses PLOT_PATH + "/segments".
    num_examples : int
        Number of best and worst segments to plot.
    """
    if out_dir is None:
        out_dir = os.path.join(PLOT_PATH, "segments")
    ensure_dir(out_dir)

    # Sort by error
    df_sorted = df_pred.sort_values("recon_error").reset_index(drop=True)
    best_rows = df_sorted.head(num_examples)
    worst_rows = df_sorted.tail(num_examples)

    # Best segments
    for _, row in best_rows.iterrows():
        seg = int(row["Segment_nr"])
        real_seq = np.array(row["Sequence_real"], dtype=np.float32)
        pred_seq = np.array(row["Sequence_pred"], dtype=np.float32)

        title = f"Segment {seg} (low error)"
        filename = os.path.join(out_dir, f"segment_{seg}_low.png")

        plot_segment_with_error(
            real_seq=real_seq,
            pred_seq=pred_seq,
            feature_indices=feature_indices,
            feature_names=feature_names,
            title=title,
            filename=filename,
        )

    # Worst segments
    for _, row in worst_rows.iterrows():
        seg = int(row["Segment_nr"])
        real_seq = np.array(row["Sequence_real"], dtype=np.float32)
        pred_seq = np.array(row["Sequence_pred"], dtype=np.float32)

        title = f"Segment {seg} (high error)"
        filename = os.path.join(out_dir, f"segment_{seg}_high.png")

        plot_segment_with_error(
            real_seq=real_seq,
            pred_seq=pred_seq,
            feature_indices=feature_indices,
            feature_names=feature_names,
            title=title,
            filename=filename,
        )


# ---------------------------------------------------------------------------
# 4) Denormalization & saving for future inspections
# ---------------------------------------------------------------------------

def denormalize_and_save_predictions(
    df_pred: pd.DataFrame,
    metadata_path: str,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Denormalize predictions dataframe (real_* and pred_* columns) using
    the same metadata file you used in inspection_utils.denormalize_predictions.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Dataframe produced by build_test_predictions_df.
    metadata_path : str
        Path to the JSON file with normalization statistics.
    save_path : str, optional
        If provided, the denormalized dataframe is saved to this path.

    Returns
    -------
    df_denorm : pd.DataFrame
        Denormalized dataframe, ready to be used with save_interactive_html.
    """
    df_denorm = denormalize_predictions(df_pred, metadata_path=metadata_path)

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path) or ".")
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".parquet":
            df_denorm.to_parquet(save_path, index=False)
        elif ext in (".csv", ".txt"):
            df_denorm.to_csv(save_path, index=False)
        else:
            df_denorm.to_parquet(save_path, index=False)

    return df_denorm


# ---------------------------------------------------------------------------
# 5) Interactive map (HTML) using denormalized predictions
# ---------------------------------------------------------------------------

def create_interactive_map(
    df_denorm: pd.DataFrame,
    out_html: str = "ais_maps.html",
    mmsi: Optional[Sequence[int]] = None,
    segment: Optional[Sequence[int]] = None,
    n_random: int = 8,
    head_n: Optional[int] = None,
    tiles: str = "OpenStreetMap",
    zoom_start: int = 20,
    real_color: str = "blue",
    pred_color: str = "red",
    circle_radius: float = 4.0,
) -> str:
    """
    Create an interactive HTML map using the denormalized dataframe.

    This is a thin wrapper around inspection_utils.save_interactive_html.

    Parameters
    ----------
    df_denorm : pd.DataFrame
        Denormalized dataframe with columns like:
          - real_Latitude, real_Longitude, real_SOG, real_COG,
          - pred_Latitude, pred_Longitude, pred_SOG, pred_COG,
          - MMSI, Segment_nr, recon_error, ShipType, ...
        Typically produced by denormalize_and_save_predictions().
    out_html : str
        Output HTML file path.
    mmsi : sequence[int] or int, optional
        Filter by MMSI(s).
    segment : sequence[int] or int, optional
        Filter by Segment_nr(s).
    n_random : int
        Number of random rows if no filters and no head_n.
    head_n : int, optional
        If provided (>0), take the first head_n rows (overall or after filtering).
    tiles : str
        Tile provider for Folium.
    zoom_start : int
        Initial zoom level.
    real_color : str
        Color for real trajectories.
    pred_color : str
        Color for predicted trajectories.
    circle_radius : float
        Radius for circle markers.

    Returns
    -------
    out_html : str
        Path to the generated HTML file.
    """
    ensure_dir(os.path.dirname(out_html) or ".")
    out_html_path = save_interactive_html(
        df=df_denorm,
        out_html=out_html,
        mmsi=mmsi,
        segment=segment,
        n_random=n_random,
        tiles=tiles,
        zoom_start=zoom_start,
        real_color=real_color,
        pred_color=pred_color,
        circle_radius=circle_radius,
        head_n=head_n,
    )
    return out_html_path



def create_map_for_mmsi(
    df_denorm: pd.DataFrame,
    mmsi: int,
    out_html: str,
    zoom_start: int = 20,
) -> str:
    """
    Convenience wrapper to create an interactive map for a single MMSI.
    """
    ensure_dir(os.path.dirname(out_html) or ".")
    return create_interactive_map(
        df_denorm=df_denorm,
        out_html=out_html,
        mmsi=mmsi,
        segment=None,
        n_random=0,
        head_n=None,
        zoom_start=zoom_start,
    )


def create_map_for_segments(
    df_denorm: pd.DataFrame,
    segments: Sequence[int],
    out_html: str,
    zoom_start: int = 20,
) -> str:
    """
    Convenience wrapper to create an interactive map for specific segments.
    """
    ensure_dir(os.path.dirname(out_html) or ".")
    return create_interactive_map(
        df_denorm=df_denorm,
        out_html=out_html,
        mmsi=None,
        segment=list(segments),
        n_random=0,
        head_n=None,
        zoom_start=zoom_start,
    )



def create_maps_for_best_and_worst_segments(
    df_denorm: pd.DataFrame,
    top_k: int,
    base_output_dir: str,
    zoom_start: int = 20,
) -> Dict[str, str]:
    """
    Create two interactive maps:
      - one for the best K segments (lowest recon_error),
      - one for the worst K segments (highest recon_error).

    Returns a dict with paths to the generated HTML files.
    """
    ensure_dir(base_output_dir)

    df_sorted = df_denorm.sort_values("recon_error").reset_index(drop=True)

    best_segments = df_sorted.head(top_k)["Segment_nr"].tolist()
    worst_segments = df_sorted.tail(top_k)["Segment_nr"].tolist()

    best_html = os.path.join(base_output_dir, f"ais_best_top{top_k}.html")
    worst_html = os.path.join(base_output_dir, f"ais_worst_top{top_k}.html")

    best_html = create_map_for_segments(
        df_denorm=df_denorm,
        segments=best_segments,
        out_html=best_html,
        zoom_start=zoom_start,
    )
    worst_html = create_map_for_segments(
        df_denorm=df_denorm,
        segments=worst_segments,
        out_html=worst_html,
        zoom_start=zoom_start,
    )

    return {
        "best_map": best_html,
        "worst_map": worst_html,
    }





def run_test_inspection_pipeline(
    model: torch.nn.Module,
    df_seq_test: pd.DataFrame,
    df_test_errors: pd.DataFrame,
    device: torch.device,
    seq_loss_fn,
    metadata_path: str,
    base_output_dir: str,
    feature_indices: Sequence[int],
    feature_names: Dict[int, str],
    num_examples: int = 3,
    top_k_maps: int = 3,
) -> Dict[str, Any]:
    """
    High-level convenience function to run the full test inspection pipeline.

    It performs:
      1) Build and save predictions dataframe (normalized)
      2) Plot error histograms (global + by category)
      3) Plot best and worst segments (real vs recon + per-timestep error)
      4) Plot feature-wise error histograms (per feature)
      5) Denormalize predictions and save them
      6) Create an interactive HTML map (random or full)
      7) Create interactive maps for best and worst segments (top_k_maps)

    Parameters
    ----------
    model : torch.nn.Module
        Trained autoencoder model.
    df_seq_test : pd.DataFrame
        Test dataframe with at least:
        ["Sequence", "ShipTypeID", "Segment_nr", "MMSI"].
    df_test_errors : pd.DataFrame
        Dataframe with reconstruction errors:
        ["Segment_nr", "ShipTypeID", "MMSI", "reconstruction_error", ...].
    device : torch.device
        Device used for inference.
    seq_loss_fn : callable
        Function taking (recon_batch, x_batch) and returning a tensor (B,)
        with per-sequence reconstruction error.
    metadata_path : str
        Path to JSON file with normalization statistics.
    base_output_dir : str
        Base directory where all outputs (plots, parquet, html) will be stored.
    feature_indices : sequence[int]
        Indices of features to plot/analyze.
    feature_names : dict[int, str]
        Mapping from feature index to readable feature name.
    num_examples : int
        Number of best/worst segments to plot (time-series).
    top_k_maps : int
        Number of best/worst segments to include in dedicated maps.

    Returns
    -------
    results : dict
        Dictionary with:
          - "df_pred": normalized predictions dataframe
          - "df_denorm": denormalized predictions dataframe
          - "paths": dict with paths of saved files
    """
    ensure_dir(base_output_dir)

    # 1) Build predictions DF (normalized) and save it
    pred_df_path = os.path.join(base_output_dir, "test_predictions.parquet")
    df_pred = build_test_predictions_df(
        model=model,
        df_seq_test=df_seq_test,
        device=device,
        seq_loss_fn=seq_loss_fn,
        save_path=pred_df_path,
    )

    # 2) Error histograms (global + by ShipTypeID)
    plots_dir = os.path.join(base_output_dir, "plots")
    plot_error_histograms(
        df_test_errors=df_test_errors,
        out_dir=plots_dir,
        by_shiptype=True,
        by_shiptype_name=False,
    )

    # 3) Best and worst segments (time-series + per-timestep error)
    segments_dir = os.path.join(base_output_dir, "segments")
    plot_best_and_worst_segments(
        df_pred=df_pred,
        feature_indices=feature_indices,
        feature_names=feature_names,
        out_dir=segments_dir,
        num_examples=num_examples,
    )

    # 4) Feature-wise error histograms
    feature_err_dir = os.path.join(base_output_dir, "feature_errors")
    plot_feature_error_histograms_from_pred(
        df_pred=df_pred,
        feature_indices=feature_indices,
        feature_names=feature_names,
        out_dir=feature_err_dir,
        bins=50,
    )

    # 5) Denormalize and save predictions
    denorm_df_path = os.path.join(base_output_dir, "test_predictions_denorm.parquet")
    df_denorm = denormalize_and_save_predictions(
        df_pred=df_pred,
        metadata_path=metadata_path,
        save_path=denorm_df_path,
    )

    # 6) Interactive HTML map (random or general overview)
    html_path = os.path.join(base_output_dir, "ais_test_map.html")
    html_path = create_interactive_map(
        df_denorm=df_denorm,
        out_html=html_path,
        n_random=8,
        head_n=None,
    )

    paths = {
        "pred_df_path": pred_df_path,
        "denorm_df_path": denorm_df_path,
        "plots_dir": plots_dir,
        "segments_dir": segments_dir,
        "feature_error_dir": feature_err_dir,
        "html_map": html_path,
    }

    # 7) Interactive maps for best / worst segments
    if top_k_maps is not None and top_k_maps > 0:
        maps_bw = create_maps_for_best_and_worst_segments(
            df_denorm=df_denorm,
            top_k=top_k_maps,
            base_output_dir=base_output_dir,
            zoom_start=20,
        )
        paths.update(maps_bw)

    return {
        "df_pred": df_pred,
        "df_denorm": df_denorm,
        "paths": paths,
    }


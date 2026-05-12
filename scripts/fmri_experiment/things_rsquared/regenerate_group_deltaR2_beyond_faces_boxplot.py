#!/usr/bin/env python
"""Regenerate group_deltaR2_beyond_faces_boxplot with custom layout/colors."""

from __future__ import annotations

import argparse
import os
from os.path import join as pjoin
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

def build_ffa_mask(voxdata: pd.DataFrame) -> np.ndarray:
    # Keep FFA voxels from both hemispheres.
    return ((voxdata["lFFA"] == 1) | (voxdata["rFFA"] == 1)).to_numpy()


def resolve_nc_column(results_root: str, nc_source: str) -> str:
    if nc_source == "auto":
        if "eval-sessioncv-trainpool" in results_root:
            return "nc_singletrial"
        return "nc_testset"
    if nc_source == "singletrial":
        return "nc_singletrial"
    if nc_source == "testset":
        return "nc_testset"
    raise ValueError(f"Unknown nc_source: {nc_source}")


def load_nc_ffa(thingsmri_dir: str, sub: str, nc_col: str) -> np.ndarray:
    vox_csv = pjoin(thingsmri_dir, "betas_csv", f"sub-{sub}_VoxelMetadata.csv")
    voxdata = pd.read_csv(vox_csv)
    ffa_mask = build_ffa_mask(voxdata)
    # Normalized by 100 to get [0,1].
    nc_all = voxdata[nc_col].to_numpy().astype(float) / 100.0
    return nc_all[ffa_mask]


def load_labels(labels_path: str, ndims: int) -> List[str]:
    if not labels_path:
        return [f"dim{d+1}" for d in range(ndims)]
    try:
        with open(labels_path, "r") as f:
            txt = f.read().strip()
        # Strip whitespace and potential surrounding quotes from each token.
        labels = [s.strip().strip('"').strip("'") for s in txt.split(",") if s.strip()]
        if len(labels) != ndims:
            return [f"dim{d+1}" for d in range(ndims)]
        return labels
    except FileNotFoundError:
        return [f"dim{d+1}" for d in range(ndims)]


def parse_subjects(subs_arg: str, results_root: str) -> List[str]:
    if subs_arg != "all":
        return [s.strip().replace("sub-", "") for s in subs_arg.split(",") if s.strip()]

    root = Path(results_root)
    out = []
    for d in sorted(root.glob("sub-*")):
        if d.is_dir():
            sid = d.name.replace("sub-", "")
            # Only keep subjects that appear to have the required array for this plot.
            f = d / "faces_plus_dimension" / f"sub-{sid}_deltaR2_beyond_faces_ffa.npy"
            if f.exists():
                out.append(sid)
    return out


def gather_beyond_faces_metric(
    subjects: List[str],
    thingsmri_dir: str,
    results_root: str,
    nc_threshold: float,
    labels_path: str,
    nc_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    all_rows: List[pd.DataFrame] = []
    labels_ref: List[str] | None = None
    ndims_ref: int | None = None

    for sub in subjects:
        res_sub = pjoin(results_root, f"sub-{sub}")
        path = pjoin(res_sub, "faces_plus_dimension", f"sub-{sub}_deltaR2_beyond_faces_ffa.npy")
        if not os.path.exists(path):
            continue

        arr = np.load(path)  # (ndims, n_ffa_vox)
        ndims, _ = arr.shape
        if ndims_ref is None:
            ndims_ref = ndims
        elif ndims != ndims_ref:
            raise ValueError("Mismatch in ndims across subjects.")

        nc_ffa = load_nc_ffa(thingsmri_dir, sub, nc_col)
        if nc_ffa.shape[0] != arr.shape[1]:
            raise ValueError("nc_ffa and metric array have incompatible shapes.")

        good_vox = nc_ffa > nc_threshold
        rows = []
        for d in range(ndims):
            vals = arr[d, good_vox]
            for v in vals:
                rows.append({"dim": d + 1, "value": float(v), "sub": f"sub-{sub}"})
        all_rows.append(pd.DataFrame(rows))

    if not all_rows:
        return pd.DataFrame(), []

    df_all = pd.concat(all_rows, ignore_index=True)
    labels_ref = load_labels(labels_path, ndims_ref or 0)
    df_all["dim_label"] = df_all["dim"].apply(lambda d: labels_ref[d - 1])
    return df_all, labels_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate group deltaR2 beyond faces boxplot.")
    parser.add_argument("--subs", type=str, default="all", help="Comma-separated subs or 'all'.")
    parser.add_argument(
        "--results-root",
        default=pjoin("results", "fmri_experiment", "things_rsquared"),
        help="Root directory where things_rsquared results are stored.",
    )
    parser.add_argument(
        "--thingsmri-dir",
        default=pjoin("data", "things-fmri"),
        help="Root directory of THINGS-fmri dataset.",
    )
    parser.add_argument("--nc-threshold", type=float, default=0.2, help="Noise ceiling threshold (0-1).")
    parser.add_argument(
        "--nc-source",
        type=str,
        default="auto",
        choices=["auto", "testset", "singletrial"],
        help="Noise-ceiling source for voxel thresholding (auto uses results-root).",
    )
    parser.add_argument(
        "--labels-path",
        default=pjoin("plots", "validation", "subj-all_relthresh-0.60_k-17", "manual_labels.txt"),
        help="Path to manual dimension labels file.",
    )
    parser.add_argument(
        "--outdir",
        default=pjoin("results", "fmri_experiment", "things_rsquared", "group_plots"),
        help="Base output directory for group plots.",
    )
    parser.add_argument(
        "--panel-width-in",
        type=float,
        default=7.5,
        help="Figure width in inches (full-width panel). Height scales to match the old aspect.",
    )
    parser.add_argument(
        "--start-tab10-idx",
        type=int,
        default=6,
        help="Start index in matplotlib tab10 colors to avoid the first colors.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (PDF is vector).")
    args = parser.parse_args()

    subjects = parse_subjects(args.subs, args.results_root)
    if not subjects:
        raise RuntimeError("No subjects found for this plot.")

    # Match original aspect ratio figsize=(12,5): height = width * 5/12.
    fig_w = float(args.panel_width_in)
    fig_h = fig_w * (5.0 / 12.0)

    outdir = Path(args.outdir) / f"nc-{args.nc_threshold:.2f}"
    outdir.mkdir(parents=True, exist_ok=True)

    sns.set(style="whitegrid")
    # Publication-ready tiny typography for full-width panels.
    mpl.rcParams.update(
        {
            "font.size": 5,
            "axes.labelsize": 5,
            "axes.titlesize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
        }
    )

    nc_col = resolve_nc_column(args.results_root, args.nc_source)
    df_beyond, _ = gather_beyond_faces_metric(
        subjects=subjects,
        thingsmri_dir=args.thingsmri_dir,
        results_root=args.results_root,
        nc_threshold=args.nc_threshold,
        labels_path=args.labels_path,
        nc_col=nc_col,
    )
    if df_beyond.empty:
        raise RuntimeError("No data collected for the beyond-faces metric.")

    means_bf = df_beyond.groupby("dim_label")["value"].mean().sort_values(ascending=False)
    order_bf = list(means_bf.index)

    # Subject color mapping: tab10 but offset so the first 6 tab10 colors are avoided.
    tab10 = plt.get_cmap("tab10").colors
    # Stable order so the mapping doesn't jump between runs.
    subjects_sorted = sorted(df_beyond["sub"].unique())
    palette_list = [tab10[(args.start_tab10_idx + i) % len(tab10)] for i in range(len(subjects_sorted))]
    palette = {sub: palette_list[i] for i, sub in enumerate(subjects_sorted)}

    plt.figure(figsize=(fig_w, fig_h))
    sns.boxplot(
        data=df_beyond,
        x="dim_label",
        y="value",
        hue="sub",
        order=order_bf,
        palette=palette,
        showfliers=False,
        whis=(5, 95),
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Embedding dimension")
    plt.ylabel("ΔR² (face-label+dim – face-label) / noise ceiling")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    png_path = outdir / "group_deltaR2_beyond_faces_boxplot.png"
    pdf_path = outdir / "group_deltaR2_beyond_faces_boxplot.pdf"
    plt.savefig(str(png_path), dpi=args.dpi)
    plt.savefig(str(pdf_path))
    plt.close()

    print(str(pdf_path))


if __name__ == "__main__":
    main()


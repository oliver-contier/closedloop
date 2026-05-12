#!/usr/bin/env python
"""Create a combined group boxplot: manual face-label model vs full model."""

from __future__ import annotations

import argparse
import os
from os.path import join as pjoin
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from closedloop.data import ThingsmriLoader


def build_ffa_mask(voxdata: pd.DataFrame) -> np.ndarray:
    # Keep FFA voxels from both hemispheres.
    return ((voxdata["lFFA"] == 1) | (voxdata["rFFA"] == 1)).to_numpy()


def load_nc_ffa(thingsmri_dir: str, sub: str) -> np.ndarray:
    loader = ThingsmriLoader(thingsmri_dir=thingsmri_dir)
    _, _, voxdata = loader.load_responses(sub)
    ffa_mask = build_ffa_mask(voxdata)
    # Normalized by 100 to get [0, 1].
    nc_all = voxdata["nc_testset"].to_numpy().astype(float) / 100.0
    return nc_all[ffa_mask]


def parse_subjects_all(results_root: str) -> List[str]:
    root = Path(results_root)
    out: List[str] = []
    for d in sorted(root.glob("sub-*")):
        if not d.is_dir():
            continue
        sid = d.name.replace("sub-", "")
        p_full = d / f"sub-{sid}_r2_full_ffa.npy"
        p_face = d / f"sub-{sid}_r2_face_label_ffa.npy"
        if p_full.exists() and p_face.exists():
            out.append(sid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate combined face-vs-full group R² boxplot.")
    parser.add_argument("--subs", type=str, default="all", help="Comma-separated subs (e.g., 01,02) or 'all'.")
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
        "--outdir",
        default=pjoin("results", "fmri_experiment", "things_rsquared", "group_plots"),
        help="Base output directory for group plots.",
    )
    parser.add_argument(
        "--panel-width-in",
        type=float,
        default=7.5,
        help="Figure width in inches (full-width panel).",
    )
    parser.add_argument(
        "--start-tab10-idx",
        type=int,
        default=6,
        help="Start index in matplotlib tab10 colors (offset to avoid the first colors).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (PDF is vector).")
    args = parser.parse_args()

    # rcParams-based font control (match your deltaR2 plot).
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
    sns.set(style="whitegrid")

    if args.subs == "all":
        subjects = parse_subjects_all(args.results_root)
    else:
        subjects = [s.strip().replace("sub-", "") for s in args.subs.split(",") if s.strip()]

    if not subjects:
        raise RuntimeError("No subjects found for this plot.")

    # Match old aspect ratio (12 x 5 in -> 864 x 360 pts). height = width * 5/12.
    fig_w = float(args.panel_width_in)
    fig_h = fig_w * (5.0 / 12.0)

    outdir = Path(args.outdir) / f"nc-{args.nc_threshold:.2f}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect values per subject + model.
    rows: List[dict] = []
    for sub in subjects:
        res_sub = Path(args.results_root) / f"sub-{sub}"
        p_full = res_sub / f"sub-{sub}_r2_full_ffa.npy"
        p_face = res_sub / f"sub-{sub}_r2_face_label_ffa.npy"
        if not p_full.exists() or not p_face.exists():
            continue

        r2_full = np.load(str(p_full))  # (n_ffa_vox,)
        r2_face = np.load(str(p_face))  # (n_ffa_vox,)

        nc_ffa = load_nc_ffa(args.thingsmri_dir, sub)  # (n_ffa_vox,)
        if nc_ffa.shape[0] != r2_full.shape[0]:
            raise ValueError(f"nc_ffa shape mismatch for sub-{sub}.")

        good_vox = nc_ffa > args.nc_threshold
        rows.append(
            pd.DataFrame(
                {
                    "model": "Full model",
                    "value": r2_full[good_vox].astype(float),
                    "sub": f"sub-{sub}",
                }
            )
        )
        rows.append(
            pd.DataFrame(
                {
                    "model": "Manual face-label model",
                    "value": r2_face[good_vox].astype(float),
                    "sub": f"sub-{sub}",
                }
            )
        )

    df = pd.concat(rows, ignore_index=True)

    # Stable subject→color mapping, using tab10 offset (same as your deltaR2 regenerate).
    tab10 = plt.get_cmap("tab10").colors
    subjects_sorted = sorted(df["sub"].unique())
    palette_list = [tab10[(args.start_tab10_idx + i) % len(tab10)] for i in range(len(subjects_sorted))]
    palette = {sub: palette_list[i] for i, sub in enumerate(subjects_sorted)}

    model_order = ["Manual face-label model", "Full model"]
    plt.figure(figsize=(fig_w, fig_h))
    sns.boxplot(
        data=df,
        x="model",
        y="value",
        hue="sub",
        order=model_order,
        palette=palette,
        fliersize=1,
    )
    # Overlay individual subject samples (swarm-like points).
    sns.stripplot(
        data=df,
        x="model",
        y="value",
        hue="sub",
        order=model_order,
        palette=palette,
        dodge=True,
        jitter=0.25,
        alpha=0.2,
        size=2.2,
        linewidth=0,
    )
    plt.xlabel("Model")
    plt.ylabel("R² / noise ceiling")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()

    png_path = outdir / "group_r2_face_vs_full_boxplot.png"
    pdf_path = outdir / "group_r2_face_vs_full_boxplot.pdf"
    plt.savefig(str(png_path), dpi=args.dpi)
    plt.savefig(str(pdf_path))
    plt.close()

    print(str(pdf_path))


if __name__ == "__main__":
    main()


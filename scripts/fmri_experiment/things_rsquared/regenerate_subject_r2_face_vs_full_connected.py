#!/usr/bin/env python
"""Per-subject connected boxplot: manual face-label model vs full model.

For each subject, for every voxel that passes the noise-ceiling threshold:
draw a line connecting the (face-label R²) value (left box) to the
(full-model R²) value (right box).
"""

from __future__ import annotations

import argparse
from os.path import join as pjoin
from pathlib import Path
from typing import Dict, List, Tuple

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
    nc_all = voxdata["nc_testset"].to_numpy().astype(float) / 100.0
    return nc_all[ffa_mask]


def parse_subjects_all(results_root: Path) -> List[str]:
    out: List[str] = []
    for d in sorted(results_root.glob("sub-*")):
        if not d.is_dir():
            continue
        out.append(d.name.replace("sub-", ""))
    return out


def load_subject_arrays(
    results_root: Path,
    sub: str,
) -> Tuple[np.ndarray, np.ndarray]:
    res_sub = results_root / f"sub-{sub}"
    p_full = res_sub / f"sub-{sub}_r2_full_ffa.npy"
    p_face = res_sub / f"sub-{sub}_r2_face_label_ffa.npy"
    if not p_full.exists() or not p_face.exists():
        raise FileNotFoundError(f"Missing arrays for sub-{sub}: {p_full.name} / {p_face.name}")
    r2_full = np.load(str(p_full))  # (n_ffa_vox,)
    r2_face = np.load(str(p_face))  # (n_ffa_vox,)
    return r2_face, r2_full


def plot_one_subject(
    *,
    subject: str,
    r2_face: np.ndarray,
    r2_full: np.ndarray,
    nc_ffa: np.ndarray,
    nc_threshold: float,
    out_pdf: Path,
    out_png: Path,
    panel_width_in: float,
    linewidth: float,
    point_alpha: float,
    dpi: int,
) -> None:
    if nc_ffa.shape[0] != r2_face.shape[0] or r2_face.shape[0] != r2_full.shape[0]:
        raise ValueError(f"Shape mismatch for sub-{subject}: nc_ffa={nc_ffa.shape}, face={r2_face.shape}, full={r2_full.shape}")

    good_vox = nc_ffa > nc_threshold
    face_vals = r2_face[good_vox].astype(float)
    full_vals = r2_full[good_vox].astype(float)

    # Match old aspect ratio figsize=(12,5): height = width * 5/12.
    fig_w = float(panel_width_in)
    fig_h = fig_w * (5.0 / 12.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.set(style="whitegrid")

    model_order = ["Manual face-label model", "Full model"]
    x_face = 0
    x_full = 1

    # Side-by-side boxplots (neutral styling).
    bp = ax.boxplot(
        [face_vals, full_vals],
        positions=[x_face, x_full],
        widths=0.5,
        patch_artist=True,
        showfliers=True,
    )
    # Style boxes (left = gray, right = white).
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("#e6e6e6" if i == 0 else "#ffffff")
        box.set_edgecolor("black")
        box.set_linewidth(1.0)
    # Style whiskers/caps/medians for consistent contrast.
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.0)

    # Keep connected lines behind axes/ticks so labels remain readable.
    ax.set_axisbelow(True)

    # Connected per-voxel lines (each voxel is one line pair across models).
    # Use sorted x so lines look "ascending" when full > face.
    for fv, rv in zip(face_vals, full_vals):
        ax.plot(
            [x_face, x_full],
            [fv, rv],
            color="black",
            alpha=point_alpha,
            linewidth=linewidth,
            zorder=1,
        )

    # Axis styling.
    ax.set_ylim(-0.1, 0.5)
    ax.set_xlabel("")
    ax.set_ylabel("R² / noise ceiling")
    ax.set_xticks([x_face, x_full])
    ax.set_xticklabels(model_order, rotation=0, ha="center")

    fig.tight_layout()
    # Narrow figures still need enough margin for x tick labels.
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.12, right=0.98)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=dpi)
    fig.savefig(str(out_pdf))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-subject connected boxplot (face-label vs full).")
    parser.add_argument(
        "--subs",
        type=str,
        default="all",
        help="Comma-separated subs (e.g., 02,03) or 'all'.",
    )
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
        help="Base output directory for per-subject plots.",
    )
    parser.add_argument(
        "--panel-width-in",
        type=float,
        default=3.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.18,
        help="Alpha for the connected voxel lines.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=0.4,
        help="Line width for the connected voxel lines.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    args = parser.parse_args()

    # Typography: match fontsize 5 requirement.
    mpl.rcParams.update(
        {
            "font.size": 5,
            "axes.labelsize": 5,
            "axes.titlesize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
        }
    )

    results_root = Path(args.results_root).resolve()

    if args.subs == "all":
        subjects = parse_subjects_all(results_root)
    else:
        subjects = [s.strip().replace("sub-", "") for s in args.subs.split(",") if s.strip()]

    for sub in subjects:
        r2_face, r2_full = load_subject_arrays(results_root=results_root, sub=sub)
        nc_ffa = load_nc_ffa(args.thingsmri_dir, sub)
        sub_outdir = results_root / f"sub-{sub}" / "plots"
        sub_outdir.mkdir(parents=True, exist_ok=True)
        out_pdf = sub_outdir / f"sub-{sub}_r2_face-vs-full_boxplot_connected_nc{args.nc_threshold:.2f}.pdf"
        out_png = sub_outdir / f"sub-{sub}_r2_face-vs-full_boxplot_connected_nc{args.nc_threshold:.2f}.png"
        plot_one_subject(
            subject=sub,
            r2_face=r2_face,
            r2_full=r2_full,
            nc_ffa=nc_ffa,
            nc_threshold=args.nc_threshold,
            out_pdf=out_pdf,
            out_png=out_png,
            panel_width_in=args.panel_width_in,
            linewidth=args.line_width,
            point_alpha=args.line_alpha,
            dpi=args.dpi,
        )

    print("Done")


if __name__ == "__main__":
    main()


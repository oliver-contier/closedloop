#!/usr/bin/env python
"""Stack per-subject connected face-vs-full comparisons along x axis."""

from __future__ import annotations

import argparse
import os
from os.path import join as pjoin
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_ffa_mask(voxdata: pd.DataFrame) -> np.ndarray:
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
    nc_all = voxdata[nc_col].to_numpy().astype(float) / 100.0
    return nc_all[ffa_mask]


def parse_subjects_all(results_root: Path) -> List[str]:
    out: List[str] = []
    for d in sorted(results_root.glob("sub-*")):
        if not d.is_dir():
            continue
        sid = d.name.replace("sub-", "")
        p_full = d / f"sub-{sid}_r2_full_ffa.npy"
        p_face = d / f"sub-{sid}_r2_face_label_ffa.npy"
        if p_full.exists() and p_face.exists():
            out.append(sid)
    return out


def load_subject_arrays(results_root: Path, sub: str) -> Tuple[np.ndarray, np.ndarray]:
    res_sub = results_root / f"sub-{sub}"
    p_full = res_sub / f"sub-{sub}_r2_full_ffa.npy"
    p_face = res_sub / f"sub-{sub}_r2_face_label_ffa.npy"
    if not p_full.exists() or not p_face.exists():
        raise FileNotFoundError(f"Missing arrays for sub-{sub}.")
    r2_full = np.load(str(p_full))  # (n_ffa_vox,)
    r2_face = np.load(str(p_face))  # (n_ffa_vox,)
    return r2_face, r2_full


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack connected subjectwise face-vs-full plots.")
    parser.add_argument("--subs", type=str, default="all", help="Comma-separated subs (01,02) or 'all'.")
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
        default=7.2,
        help="Figure width in inches for the stacked group plot.",
    )
    parser.add_argument(
        "--pair-gap",
        type=float,
        default=0.85,
        help="Distance between face and full box centers for the same subject.",
    )
    parser.add_argument(
        "--subject-step",
        type=float,
        default=2.0,
        help="Center-to-center distance between subject pairs along x axis.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (PDF is vector).")
    parser.add_argument("--line-alpha", type=float, default=0.18, help="Alpha for connected voxel lines.")
    parser.add_argument("--line-width", type=float, default=0.4, help="Line width for connected voxel lines.")
    parser.add_argument(
        "--nc-source",
        type=str,
        default="auto",
        choices=["auto", "testset", "singletrial"],
        help="Noise-ceiling source for voxel thresholding (auto uses results-root).",
    )
    args = parser.parse_args()

    # Typography: match fontsize 5.
    mpl.rcParams.update(
        {
            "font.size": 5,
            "axes.labelsize": 5,
            "axes.titlesize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
        }
    )
    sns.set(style="whitegrid")

    results_root = Path(args.results_root).resolve()
    nc_col = resolve_nc_column(str(results_root), args.nc_source)
    if args.subs == "all":
        subjects = parse_subjects_all(results_root)
    else:
        subjects = [s.strip().replace("sub-", "") for s in args.subs.split(",") if s.strip()]
    if not subjects:
        raise RuntimeError("No subjects found for this stacked plot.")

    outdir = Path(args.outdir) / f"nc-{args.nc_threshold:.2f}"
    outdir.mkdir(parents=True, exist_ok=True)

    fig_w = float(args.panel_width_in)
    fig_h = fig_w * (5.0 / 12.0)  # keep aspect consistent with other panels
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Ensure connected lines don't cover ticks/labels.
    ax.set_axisbelow(True)

    face_width = 0.5
    full_width = 0.5

    # Build boxes + lines for each subject.
    # x layout: subject pair i at base_x=i*subject_step with centers base_x and base_x+pair_gap.
    for i, sub in enumerate(subjects):
        base_x = i * float(args.subject_step)
        x_face = base_x
        x_full = base_x + float(args.pair_gap)

        r2_face, r2_full = load_subject_arrays(results_root=results_root, sub=sub)
        nc_ffa = load_nc_ffa(args.thingsmri_dir, sub, nc_col)
        if nc_ffa.shape[0] != r2_face.shape[0] or r2_face.shape[0] != r2_full.shape[0]:
            raise ValueError(f"Shape mismatch for sub-{sub}.")

        good_vox = nc_ffa > args.nc_threshold
        face_vals = r2_face[good_vox].astype(float)
        full_vals = r2_full[good_vox].astype(float)
        keep = (
            np.isfinite(face_vals)
            & np.isfinite(full_vals)
            & (face_vals <= 1.0)
            & (full_vals <= 1.0)
        )
        face_vals = face_vals[keep]
        full_vals = full_vals[keep]
        if face_vals.size:
            lo_face = np.percentile(face_vals, 1.0)
            face_vals = np.maximum(face_vals, lo_face)
        if full_vals.size:
            lo_full = np.percentile(full_vals, 1.0)
            full_vals = np.maximum(full_vals, lo_full)

        # Side-by-side boxplots for this subject (neutral styling; no subject colors).
        bp = ax.boxplot(
            [face_vals, full_vals],
            positions=[x_face, x_full],
            widths=[face_width, full_width],
            patch_artist=True,
            showfliers=False,
            whis=[5, 95],
            zorder=2,
        )

        # Style: left = gray, right = white.
        for j, box in enumerate(bp["boxes"]):
            box.set_facecolor("#e6e6e6" if j == 0 else "#ffffff")
            box.set_edgecolor("black")
            box.set_linewidth(1.0)
        for whisker in bp["whiskers"]:
            whisker.set_color("black")
            whisker.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_color("black")
            cap.set_linewidth(1.0)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.0)

        # Show voxel-level distributions as light-alpha swarm-like points.
        rng = np.random.default_rng(0)
        jf = rng.uniform(-0.07, 0.07, size=face_vals.shape[0])
        jr = rng.uniform(-0.07, 0.07, size=full_vals.shape[0])
        ax.scatter(
            np.full(face_vals.shape[0], x_face) + jf,
            face_vals,
            s=10,
            alpha=0.18,
            color="black",
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            np.full(full_vals.shape[0], x_full) + jr,
            full_vals,
            s=10,
            alpha=0.18,
            color="black",
            linewidths=0,
            zorder=3,
        )

    # No axis labels/ticks; user can add manually later.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.tick_params(axis="y", which="both", labelleft=False, length=0)
    ax.set_ylim(-0.2, 1.0)

    fig.tight_layout()

    png_path = outdir / "group_r2_face-vs-full_subjectstack_connected.png"
    pdf_path = outdir / "group_r2_face-vs-full_subjectstack_connected.pdf"
    plt.savefig(str(png_path), dpi=args.dpi)
    plt.savefig(str(pdf_path))
    plt.close(fig)

    print(str(pdf_path))


if __name__ == "__main__":
    main()


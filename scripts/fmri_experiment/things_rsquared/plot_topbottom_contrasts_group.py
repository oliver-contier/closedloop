#!/usr/bin/env python
"""Group plots for top-k minus bottom-k FFA response contrasts."""

import argparse
import os
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_labels(path, ndims):
    try:
        labels = [s.strip().strip('"').strip("'") for s in open(path).read().split(",") if s.strip()]
    except FileNotFoundError:
        return [f"dim{d + 1}" for d in range(ndims)]
    return labels if len(labels) == ndims else [f"dim{d + 1}" for d in range(ndims)]


def ffa_mask(voxdata):
    return ((voxdata["lFFA"] == 1) | (voxdata["rFFA"] == 1)).to_numpy()


def load_nc(thingsmri_dir, sub):
    vox = pd.read_csv(pjoin(thingsmri_dir, "betas_csv", f"sub-{sub}_VoxelMetadata.csv"))
    return vox.loc[ffa_mask(vox), "nc_singletrial"].to_numpy(float) / 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subs", nargs="+", required=True)
    parser.add_argument("--results-root", default=pjoin("results", "fmri_experiment", "things_rsquared", "eval-trainpool-leave3"))
    parser.add_argument("--thingsmri-dir", default=pjoin("data", "things-fmri"))
    parser.add_argument("--labels-path", default=pjoin("plots", "validation", "subj-all_relthresh-0.60_k-17", "manual_labels.txt"))
    parser.add_argument("--nc-threshold", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=75)
    parser.add_argument("--outdir", default=pjoin("results", "fmri_experiment", "things_rsquared", "eval-trainpool-leave3", "group_plots"))
    args = parser.parse_args()

    rows = []
    meta_rows = []
    ndims_ref = None
    for sub in args.subs:
        arr_path = pjoin(
            args.results_root,
            f"sub-{sub}",
            "top_bottom_k",
            f"sub-{sub}_topbottom-k{args.top_k}_contrast_ffa.npy",
        )
        if not os.path.exists(arr_path):
            continue
        arr = np.load(arr_path)
        ndims, nvox = arr.shape
        ndims_ref = ndims if ndims_ref is None else ndims_ref
        labels = load_labels(args.labels_path, ndims)
        nc = load_nc(args.thingsmri_dir, sub)
        good = np.isfinite(nc) & (nc > args.nc_threshold)
        for dim in range(ndims):
            vals = arr[dim, good]
            vals = vals[np.isfinite(vals)]
            for val in vals:
                rows.append({"sub": f"sub-{sub}", "dim": dim + 1, "dim_label": labels[dim], "value": float(val)})
        face_arr_path = pjoin(
            args.results_root,
            f"sub-{sub}",
            "top_bottom_k",
            f"sub-{sub}_topbottom-k{args.top_k}_face_label_contrast_ffa.npy",
        )
        if os.path.exists(face_arr_path):
            vals = np.load(face_arr_path)[good]
            vals = vals[np.isfinite(vals)]
            for val in vals:
                rows.append(
                    {
                        "sub": f"sub-{sub}",
                        "dim": 0,
                        "dim_label": "Binary face labels",
                        "value": float(val),
                    }
                )
        meta_path = pjoin(
            args.results_root,
            f"sub-{sub}",
            "top_bottom_k",
            f"sub-{sub}_topbottom-k{args.top_k}_metadata.csv",
        )
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path)
            meta["sub"] = f"sub-{sub}"
            meta["dim_label"] = meta["dim"].apply(
                lambda d: "Binary face labels" if d == 0 else labels[d - 1]
            )
            meta_rows.append(meta)

    outdir = pjoin(args.outdir, f"nc-{args.nc_threshold:.2f}")
    os.makedirs(outdir, exist_ok=True)
    if not rows:
        print("No contrast rows found.")
        return

    df = pd.DataFrame(rows)
    order = (
        df.groupby("dim_label")["value"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if "Binary face labels" in order:
        order = ["Binary face labels"] + [lab for lab in order if lab != "Binary face labels"]
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    plt.figure(figsize=(12, 5))
    sns.boxplot(
        data=df,
        x="dim_label",
        y="value",
        hue="sub",
        order=order,
        palette="Set2",
        showfliers=False,
        whis=(5, 95),
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Embedding dimension")
    plt.ylabel(f"Top-{args.top_k} minus bottom-{args.top_k} response (voxel SD)")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"FFA top-bottom dimension contrasts (nc_singletrial > {args.nc_threshold})")
    plt.tight_layout()
    plt.savefig(pjoin(outdir, f"group_topbottom-k{args.top_k}_contrast_boxplot.png"), dpi=300)
    plt.savefig(pjoin(outdir, f"group_topbottom-k{args.top_k}_contrast_boxplot.pdf"))
    plt.close()

    summary = df.groupby(["dim_label", "sub"])["value"].agg(["mean", "count", "std"]).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["sem"]
    x_spacing = 1.25
    xmap = {lab: i for i, lab in enumerate(order)}
    summary["x"] = summary["dim_label"].map(xmap) * x_spacing
    plt.figure(figsize=(6.85, 3.1))
    subjects = sorted(summary["sub"].unique())
    offsets = np.linspace(-0.18, 0.18, len(subjects)) if len(subjects) > 1 else [0.0]
    for sub, offset in zip(subjects, offsets):
        dsub = summary[summary["sub"] == sub].sort_values("x")
        plt.errorbar(
            dsub["x"] + offset,
            dsub["mean"],
            yerr=dsub["ci95"],
            fmt="o",
            capsize=2.4,
            capthick=1.08,
            elinewidth=1.08,
            linestyle="none",
            markersize=5.0,
            markeredgewidth=0,
            label=sub,
        )
    plt.axhline(0, color="0.3", linestyle="--", linewidth=0.7)
    plt.xticks(np.arange(len(order)) * x_spacing, order, rotation=45, ha="right")
    plt.xlabel("Embedding dimension")
    plt.ylabel(f"Top-{args.top_k} minus bottom-{args.top_k} response (mean +/- 95% CI)")
    plt.title(f"Mean FFA top-bottom contrasts (nc_singletrial > {args.nc_threshold})")
    plt.ylim(-0.5, 0.5)
    plt.tick_params(axis="x", length=0)
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", linewidth=0.4)
    sns.despine(trim=True, bottom=True)
    plt.legend(title="Subject", frameon=False, handlelength=1.0, borderaxespad=0.2)
    plt.tight_layout()
    plt.savefig(pjoin(outdir, f"group_topbottom-k{args.top_k}_contrast_mean_ci.png"), dpi=300)
    plt.savefig(pjoin(outdir, f"group_topbottom-k{args.top_k}_contrast_mean_ci.pdf"))
    plt.close()

    df.to_csv(pjoin(outdir, f"group_topbottom-k{args.top_k}_contrast_values.csv"), index=False)
    if meta_rows:
        pd.concat(meta_rows, ignore_index=True).to_csv(
            pjoin(outdir, f"group_topbottom-k{args.top_k}_metadata.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()

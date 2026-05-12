#!/usr/bin/env python
"""Generate full group plot suite for all eval modes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from os.path import join as pjoin


def mode_results_root(base_results_root: str, mode: str) -> str:
    if mode == "all":
        return base_results_root
    return pjoin(base_results_root, f"eval-{mode}")


def mode_outdir(base_outdir: str, mode: str) -> str:
    return pjoin(base_outdir, f"eval-{mode}")


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate group-level THINGS-R^2 plots for all evaluation modes "
            "(all, balanced, face, nonface)."
        )
    )
    parser.add_argument("--subs", nargs="+", required=True, help="Subject IDs, e.g. 01 02 03")
    parser.add_argument(
        "--results-root",
        default=pjoin("results", "fmri_experiment", "things_rsquared"),
        help="Base results directory containing sub-XX and eval-* subdirectories.",
    )
    parser.add_argument(
        "--thingsmri-dir",
        default=pjoin("data", "things-fmri"),
        help="Root directory of THINGS-fmri dataset.",
    )
    parser.add_argument(
        "--labels-path",
        default=pjoin("plots", "validation", "subj-all_relthresh-0.60_k-17", "manual_labels.txt"),
        help="Path to manual dimension labels file.",
    )
    parser.add_argument("--nc-threshold", type=float, default=0.2, help="Noise ceiling threshold (0-1).")
    parser.add_argument(
        "--outdir",
        default=pjoin("results", "fmri_experiment", "things_rsquared", "group_plots"),
        help="Base output directory. Each mode is saved under outdir/eval-<mode>/nc-<thr>.",
    )
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        default=["all", "balanced", "face", "nonface"],
        choices=["all", "balanced", "face", "nonface"],
        help="Which evaluation modes to plot.",
    )
    args = parser.parse_args()

    subs_csv = ",".join(args.subs)
    for mode in args.eval_modes:
        res_root_mode = mode_results_root(args.results_root, mode)
        outdir_mode = mode_outdir(args.outdir, mode)
        print(f"\n=== Plotting eval mode: {mode} ===")

        run(
            [
                sys.executable,
                "scripts/fmri_experiment/things_rsquared/plot_things_rsquared_group.py",
                "--subs",
                *args.subs,
                "--results-root",
                res_root_mode,
                "--thingsmri-dir",
                args.thingsmri_dir,
                "--nc-threshold",
                str(args.nc_threshold),
                "--labels-path",
                args.labels_path,
                "--outdir",
                outdir_mode,
            ]
        )
        run(
            [
                sys.executable,
                "scripts/fmri_experiment/things_rsquared/regenerate_group_deltaR2_beyond_faces_boxplot.py",
                "--subs",
                subs_csv,
                "--results-root",
                res_root_mode,
                "--thingsmri-dir",
                args.thingsmri_dir,
                "--nc-threshold",
                str(args.nc_threshold),
                "--labels-path",
                args.labels_path,
                "--outdir",
                outdir_mode,
            ]
        )
        run(
            [
                sys.executable,
                "scripts/fmri_experiment/things_rsquared/regenerate_group_r2_beyond_faces_with_baseline_boxplot.py",
                "--subs",
                subs_csv,
                "--results-root",
                res_root_mode,
                "--thingsmri-dir",
                args.thingsmri_dir,
                "--nc-threshold",
                str(args.nc_threshold),
                "--labels-path",
                args.labels_path,
                "--outdir",
                outdir_mode,
            ]
        )


if __name__ == "__main__":
    main()

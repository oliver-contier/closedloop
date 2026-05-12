#!/usr/bin/env bash
# Produce aggregate plots for euclidean_distance, zscore-voxels, cross-split,
# with permutation 10000 (betas and contrasts). Run after run_permutation_euclidean_zscore_crosssplit.sh has finished.

set -e
cd "$(dirname "$0")/../../.."
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
[ -d "${PROJECT_ROOT}/.venv" ] && . "${PROJECT_ROOT}/.venv/bin/activate"

RESULTS_ROOT="$PROJECT_ROOT/results/fmri_experiment/pattern_correlation_roi-FFA_denoised-compcor_crosssplit"
SUBJECTS="02 03 04 05 06 07"

for STAT in betas contrasts; do
  OUT_DIR="$RESULTS_ROOT/aggregate_plots_FFA_euclidean_${STAT}_crosssplit_voxzscored_perm10000"
  python scripts/fmri_experiment/pattern_analysis/plot_aggregate_correlations.py \
    --results_root "$RESULTS_ROOT" \
    --output_dir "$OUT_DIR" \
    --subjects $SUBJECTS \
    --statstype "$STAT" \
    --similarity-metric euclidean_distance \
    --split-method crosssplit \
    --zscore-voxels \
    --denoising-approach compcor \
    --permutation \
    --n-permutations 10000 \
    --shuffle-split odd
done

#!/usr/bin/env bash
# Run pattern correlation with euclidean_distance, zscore-voxels, cross-split,
# and permutation testing (10000) for both betas and contrasts.
# Usage: run_permutation_euclidean_zscore_crosssplit.sh [subjects]
#   subjects: default "02 03 04 05 06 07"

set -e
cd "$(dirname "$0")/../../.."
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
[ -d "${PROJECT_ROOT}/.venv" ] && . "${PROJECT_ROOT}/.venv/bin/activate"

SUBJECTS="${1:-02 03 04 05 06 07}"

for SUB in $SUBJECTS; do
  for STAT in betas contrasts; do
    python scripts/fmri_experiment/pattern_analysis/run_pattern_correlation.py \
      --project_root "$PROJECT_ROOT" \
      --sub "$SUB" \
      --statstype "$STAT" \
      --roi-name FFA \
      --denoising-approach compcor \
      --similarity-metric euclidean_distance \
      --zscore-voxels \
      --n-jobs 8 \
      --permutation \
      --n-permutations 10000 \
      --shuffle-split odd
  done
done

#!/usr/bin/env bash
# Train the sparse, non-negative similarity embedding model on the simulated
# odd-one-out triplets derived from in-silico FFA responses.
#
# This wraps the deep_embeddings package
# (https://github.com/florianmahner/object-dimensions, originally
# https://github.com/ViCCo-Group/SPoSE), with hyperparameters set to those
# described in the paper Methods (Similarity embedding model):
#   - L1 penalty beta:        0.0003
#   - learning rate:          0.001
#   - initial dimensionality: 120
#   - training/test split:    90/10 of 4 million triplets
#
# Usage: ./spose_ffa_ontriplets.sh [n_triplets_in_millions] [init_dim]
#   defaults: 4 million triplets, 120 initial dimensions
#
# The training was repeated 20 times with different random seeds, see
# Methods (Reproducible group-level embedding).

NTRIPS_MIO=${1:-4}
INIT_DIM=${2:-120}

python ../../../deep_embeddings/main.py \
    --identifier ffadims \
    --triplet_path ../../data/preproc/simthresh-0.0_trans-zvox/triplets/triplets_${NTRIPS_MIO}mio/ \
    --log_path ../../results/simthresh-0.0_trans-zvox \
    --method deterministic \
    --init_dim ${INIT_DIM} \
    --beta 0.0003 \
    --lr 0.001 \
    --modality behavior \
    --tensorboard

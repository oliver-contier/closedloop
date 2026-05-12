# closedloop

Analysis code for the project

> **Closed-loop hypothesis generation and testing for characterizing representations in visually-responsive brain regions** — Oliver Contier, N. Apurva Ratan Murty, Martin N. Hebart.

This repository contains the code used to produce the results of the
project. It implements a closed-loop framework that
(i) generates *in silico* FFA response patterns to natural images,
(ii) derives interpretable representational dimensions from those patterns,
and (iii) tests those dimensions in two complementary in-vivo experiments
(THINGS-data encoding and a targeted block-design fMRI experiment).

## Datasets used / produced

| Resource | Where to get it |
| --- | --- |
| THINGS image database | https://osf.io/jum2f |
| THINGS-data fMRI (validation) | https://doi.org/10.18112/openneuro.ds004192.v1.0.5 |
| Targeted fMRI validation experiment | https://doi.org/10.18112/openneuro.ds007791.v1.0.0 |
| In-silico FFA dimensions, human dimension ratings, face labels | This repository — see [`data/`](data/) |

The deposited assets in `data/` are:

- `data/embedding/emb_filtered.npy` — the final 15-dimension embedding of THINGS images
- `data/embedding/dimension_labels.txt` — author-assigned dimension names
- `data/embedding/pred_dims_from_subjs.csv` — cross-subject validation *R*²
- `data/labeling_task_data/*.csv` — raw human dimension ratings (N = 12 raters)
- `data/face_labels/FACE.csv` — manual binary face labels per THINGS image

See [`data/README.md`](data/README.md) for column-level documentation.

## Repository layout

```
closedloop/
├── src/closedloop/              Python package with shared analysis code
│   ├── triplets.py              Odd-one-out triplet generation
│   ├── consensus.py             Dimension reliability + clustering
│   ├── inspect_spose.py         Loaders for SPoSE / deep_embeddings results
│   ├── validate.py              Cross-subject dimension validation
│   ├── encoding.py              Fractional-ridge voxel-wise encoding
│   ├── data.py                  THINGS / THINGS-data loaders
│   ├── facelabels.py            Binary face-label loaders
│   ├── maths.py, plotting.py, utils.py
│   └── fmri/
│       ├── analyze_experiment.py   GLM workflow + data loader
│       └── roi.py                  ROI masking utilities
│
├── scripts/
│   ├── tripletize/              Generate 4 M euclidean-distance triplets
│   ├── run_deep_embeddings/     Train SPoSE; build group-level consensus
│   ├── labeling_task/           Word clouds from human dimension labels
│   └── fmri_experiment/
│       ├── top_images/          Select stimuli for the targeted experiment
│       ├── preproc/             dcm2bids + fMRIPrep wrappers
│       ├── first_level/         Block-design + fLoc GLMs
│       ├── things_rsquared/     THINGS-data encoding (leave-3-sessions-out)
│       └── pattern_analysis/    Pattern selectivity in the targeted experiment
│
├── fMRI_experiment_code/
│   ├── ffadims_block/           PsychoPy block-design presentation
│   └── pyfLoc/                  Stanford VPNL fLoc localizer (vendored)
│
└── labeling_task/
    └── pythonanywhere_app/      Flask web app used to collect dimension labels
```

## Pipeline overview

### 1. In-silico FFA responses → representational dimensions

1. **Preprocess in-silico responses** (z-score per voxel):
   ```bash
   python scripts/preproc_data.py
   ```
2. **Generate 4 M odd-one-out triplets per subject** using euclidean
   distances between voxel response patterns:
   ```bash
   bash scripts/tripletize/tripletize_euclidean.sh
   ```
3. **Train the sparse, non-negative similarity embedding** (β = 0.0003,
   learning rate = 0.001, 120 initial dimensions; 20 random seeds, 90/10
   triplet split). This wraps
   [object-dimensions](https://github.com/florianmahner/object-dimensions):
   ```bash
   bash scripts/run_deep_embeddings/spose_ffa_ontriplets.sh 4 120
   ```
4. **Build the reliable, group-level embedding** (split-half reliability
   threshold *r* = 0.6, 17 agglomerative-cluster centroids; then keep only
   dimensions predictable from each individual subject with *R*² > 0.6,
   yielding the final 15 dimensions):
   ```bash
   python scripts/run_deep_embeddings/run_consensus_subject.py --subj all
   python scripts/run_deep_embeddings/run_validation_regression.py --subj all --k 17
   ```
5. **Optional**: collect human labels for each dimension and render word
   clouds. The Flask app under `labeling_task/pythonanywhere_app/` was
   used to gather labels from N=12 raters; aggregate with:
   ```bash
   python scripts/labeling_task/generate_dimension_wordclouds.py
   ```

### 2. THINGS-data validation (Figure 2)

Fit a fractional-ridge encoding model in FFA, using leave-3-sessions-out
cross-validation, fractions 0.05–1.0 (step 0.05), and top-75 vs.
bottom-75 dimension-wise contrasts:

```bash
for sub in 01 02 03; do
  python scripts/fmri_experiment/things_rsquared/run_trainpool_leave3_cv.py --sub "$sub"
done
```

Figure-generation scripts (R² normalised by single-trial noise ceiling,
contrasted against a one-predictor face-label baseline) live next to it:

- `regenerate_group_r2_face_vs_full_boxplot.py`
- `regenerate_subject_r2_face_vs_full_connected.py`
- `regenerate_group_r2_beyond_faces_with_baseline_boxplot.py`
- `plot_topbottom_contrasts_group.py`

### 3. Targeted fMRI experiment (Figure 3)

1. **Select stimuli** (top images per dimension, alternating split into
   two image sets disjoint at the object-concept level):
   ```bash
   python scripts/fmri_experiment/top_images/find_top_images_per_dimension.py
   python scripts/fmri_experiment/top_images/plot_mosaic_top_images.py
   ```
2. **Run the experiment** (PsychoPy block design + fLoc localizer):
   ```bash
   python fMRI_experiment_code/ffadims_block/experiment.py
   python fMRI_experiment_code/pyfLoc/fLoc.py
   ```
3. **Preprocess**:
   ```bash
   bash scripts/fmri_experiment/preproc/run_fmriprep.sh
   python scripts/deface_t1w.py    # before deposition on OpenNeuro
   ```
4. **First-level GLMs** (no smoothing, CompCor + motion + FD as
   nuisance regressors, double-gamma HRF):
   ```bash
   python scripts/fmri_experiment/first_level/analyze_blockdesign.py
   python scripts/fmri_experiment/first_level/analyze_floc.py
   ```
5. **Pattern selectivity** in FFA (sign-inverted euclidean distance,
   z-scored voxels, cross-split within/between similarity, voxels with
   split-half reliability *r* > 0.2, 10 000 permutations):
   ```bash
   bash scripts/fmri_experiment/pattern_analysis/run_permutation_euclidean_zscore_crosssplit.sh
   bash scripts/fmri_experiment/pattern_analysis/plot_aggregate_perm10000_euclidean_zscore_crosssplit.sh
   ```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

The similarity-embedding training step depends on
[object-dimensions](https://github.com/florianmahner/object-dimensions),
which should be installed separately. PsychoPy is only required if you
want to re-run the in-scanner experiment.

## Citing

If you use this code, please also cite the THINGS image database
(Hebart et al., 2019), the THINGS-data fMRI release (Hebart et al.,
2023), and the encoding model from Murty et al.

## License

MIT (see `LICENSE`).

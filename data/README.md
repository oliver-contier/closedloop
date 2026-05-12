# Data assets distributed with this repository

These small files are deposited as part of the closedloop release.
Large raw datasets (THINGS images, THINGS-data fMRI, the targeted fMRI
validation experiment) live on external archives — see the top-level
README for the relevant DOIs.

## `embedding/`

The validated group-level embedding of in-silico FFA dimensions
described in Methods (Reproducible group-level embedding).

| File | Description |
| --- | --- |
| `emb_filtered.npy` | Float64 numpy array, shape `(26107, 15)`. Rows correspond to THINGS images (sorted as in `closedloop.data.get_thingsimages_fnames`); columns are the 15 representational dimensions retained after split-half reliability filtering (*r* > 0.6), agglomerative clustering with k = 17, and the cross-subject *R*² > 0.6 prediction filter. |
| `dimension_labels.txt` | Comma-separated names assigned by the authors to each dimension, in the same order as columns of `emb_filtered.npy`. |
| `pred_dims_from_subjs.csv` | Per-subject and group-level *R*² for predicting each consensus dimension from individual subjects' voxel data (used to derive the *R*² > 0.6 filter). |

## `labeling_task_data/`

Per-rater dimension labels collected with the Flask app under
`labeling_task/pythonanywhere_app/`. These are the inputs to
`scripts/labeling_task/generate_dimension_wordclouds.py` and the source
of the word clouds in Fig. 1C.

| File | Description |
| --- | --- |
| `responses_long.csv` | Long-format dimension labels: `token, dim_id, trial_index, label, updated_at, is_submitted`. Each `label` is the comma-separated free-text response that rater provided for the 8×8 grid of top images for that dimension. |

## `face_labels/`

| File | Description |
| --- | --- |
| `FACE.csv` | Two columns (`image_name`, `label`) giving a manually curated binary face/no-face label for each of the 26,107 THINGS images. Used to construct the one-predictor face-selectivity baseline encoding model (Methods: THINGS-data validation analysis). |

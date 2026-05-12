#!/usr/bin/env python
"""Train-pool leave-3-session-out CV plus top/bottom-k FFA contrasts."""

import argparse
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from fracridge import FracRidgeRegressor
from nilearn.masking import unmask

from closedloop.data import ThingsmriLoader
from closedloop.encoding import _find_trial_indices_in_embedding


def parse_fracs(fracs_str):
    if ":" in fracs_str:
        start, stop, step = map(float, fracs_str.split(":"))
        return np.arange(start, stop + 1e-8, step)
    return np.array([float(x) for x in fracs_str.split(",") if x.strip()], dtype=float)


def zscore_train_test(train, test):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def zscore_all(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (arr - mean) / std


def build_ffa_mask(voxdata):
    return ((voxdata["lFFA"] == 1) | (voxdata["rFFA"] == 1)).to_numpy()


def make_folds(sessions, holdout_sessions, seed):
    sessions = np.array(sorted(sessions), dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sessions)
    if len(shuffled) % holdout_sessions:
        raise ValueError("Number of sessions must be divisible by holdout_sessions.")
    return [np.sort(fold) for fold in np.array_split(shuffled, len(shuffled) // holdout_sessions)]


def fit_predict_cv(X, y, session_vec, folds, fracs):
    nvox = y.shape[1]
    sse_by_frac = np.zeros((len(fracs), nvox), dtype=np.float64)
    tss = np.zeros(nvox, dtype=np.float64)

    for heldout in folds:
        test_mask = np.isin(session_vec, heldout)
        train_mask = ~test_mask
        X_train, X_test = zscore_train_test(X[train_mask], X[test_mask])
        y_train = zscore_all(y[train_mask])
        y_test = zscore_all(y[test_mask])
        reg = FracRidgeRegressor(fracs=fracs, fit_intercept=True, normalize=False)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)  # n_test x n_fracs x n_vox
        tss += np.sum(y_test**2, axis=0)
        for frac_i in range(len(fracs)):
            sse_by_frac[frac_i] += np.sum((y_test - pred[:, frac_i, :]) ** 2, axis=0)

    pooled_r2_by_frac = np.full_like(sse_by_frac, np.nan, dtype=np.float32)
    valid_tss = tss > 0
    pooled_r2_by_frac[:, valid_tss] = (
        1.0 - sse_by_frac[:, valid_tss] / tss[valid_tss]
    ).astype(np.float32)
    best_frac_i = np.nanargmax(pooled_r2_by_frac, axis=0)
    best_fracs = fracs[best_frac_i]
    r2_best = pooled_r2_by_frac[best_frac_i, np.arange(nvox)]

    return r2_best, best_fracs, pooled_r2_by_frac


def fit_betas_full_data(X, y, best_fracs):
    Xz = zscore_all(X)
    yz = zscore_all(y)
    nvox = y.shape[1]
    ndims = X.shape[1]
    betas = np.zeros((nvox, ndims), dtype=np.float32)
    for frac in np.unique(best_fracs):
        vox = np.where(best_fracs == frac)[0]
        reg = FracRidgeRegressor(fracs=float(frac), fit_intercept=True, normalize=False)
        reg.fit(Xz, yz[:, vox])
        betas[vox] = reg.coef_.T.astype(np.float32)
    return betas


def save_ffa_map(arr_ffa, ffa_mask, n_vox, brainmask, path):
    vec = np.full(n_vox, np.nan, dtype=np.float32)
    vec[ffa_mask] = arr_ffa.astype(np.float32)
    unmask(vec, brainmask).to_filename(path)


def save_beta_map(betas_ffa, ffa_mask, n_vox, brainmask, path):
    ndims = betas_ffa.shape[1]
    arr = np.full((n_vox, ndims), np.nan, dtype=np.float32)
    arr[ffa_mask, :] = betas_ffa.astype(np.float32)
    unmask(arr.T, brainmask).to_filename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True)
    parser.add_argument("--embedding", default=pjoin("data", "embedding", "emb_filtered.npy"))
    parser.add_argument("--face-labels", default=pjoin("data", "face_labels", "FACE.csv"))
    parser.add_argument("--thingsmri-dir", default=pjoin("data", "things-fmri"))
    parser.add_argument("--outdir", default=pjoin("results", "fmri_experiment", "things_rsquared", "eval-trainpool-leave3"))
    parser.add_argument("--fracs", default="0.05:1.0:0.05")
    parser.add_argument("--holdout-sessions", type=int, default=3)
    parser.add_argument("--fold-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=75)
    args = parser.parse_args()

    embedding = np.load(args.embedding)
    loader = ThingsmriLoader(args.thingsmri_dir)
    responses_df, stimdata, voxdata = loader.load_responses(args.sub)
    train_mask = (stimdata["trial_type"] == "train").to_numpy()
    stim_train = stimdata.loc[train_mask].reset_index(drop=True)
    session_vec = stim_train["session"].to_numpy()

    trial_to_emb = _find_trial_indices_in_embedding(stimdata)
    X = embedding[trial_to_emb[train_mask]]
    voxel_ids_resp = responses_df["voxel_id"].to_numpy()
    ffa_mask = build_ffa_mask(voxdata)
    assert np.array_equal(voxel_ids_resp, voxdata["voxel_id"].to_numpy())
    y = responses_df.drop(columns="voxel_id").to_numpy().T[train_mask][:, ffa_mask]

    face_df = pd.read_csv(args.face_labels)
    face_map = dict(zip(face_df["image_name"].astype(str), face_df["label"].astype(str).str.lower().isin(["true", "1"])))
    face_vec = stim_train["stimulus"].astype(str).map(face_map).to_numpy(dtype=float)[:, None]

    nc_ffa = voxdata.loc[ffa_mask, "nc_singletrial"].to_numpy(dtype=np.float32) / 100.0
    nc_safe = nc_ffa.copy()
    nc_safe[nc_safe <= 0] = np.nan
    n_vox = responses_df.shape[0]
    n_ffa = y.shape[1]
    ndims = X.shape[1]
    fracs = parse_fracs(args.fracs)
    folds = make_folds(np.unique(session_vec), args.holdout_sessions, args.fold_seed)

    subdir = pjoin(args.outdir, f"sub-{args.sub}")
    betas_dir = pjoin(subdir, "betas_and_fracs")
    single_dir = pjoin(subdir, "single_dimension")
    contrast_dir = pjoin(subdir, "top_bottom_k")
    for d in (subdir, betas_dir, single_dir, contrast_dir):
        os.makedirs(d, exist_ok=True)
    brainmask = pjoin(args.thingsmri_dir, "unionmasks", f"sub-{args.sub}_space-T1w_brainmask.nii.gz")

    print(f"sub-{args.sub}: leave-{args.holdout_sessions}-session CV folds={[f.tolist() for f in folds]}")
    r2_full, best_full, r2_full_by_frac = fit_predict_cv(X, y, session_vec, folds, fracs)
    betas_full = fit_betas_full_data(X, y, best_full)
    np.save(pjoin(subdir, f"sub-{args.sub}_r2_full_ffa.npy"), (r2_full / nc_safe).astype(np.float32))
    np.save(pjoin(subdir, f"sub-{args.sub}_r2_full_raw_ffa.npy"), r2_full.astype(np.float32))
    np.save(pjoin(betas_dir, f"sub-{args.sub}_betas_full_ffa.npy"), betas_full)
    np.save(pjoin(betas_dir, f"sub-{args.sub}_best_fracs_ffa.npy"), best_full.astype(np.float32))
    np.save(pjoin(betas_dir, f"sub-{args.sub}_r2_full_by_frac_ffa.npy"), r2_full_by_frac.astype(np.float32))
    save_ffa_map(r2_full / nc_safe, ffa_mask, n_vox, brainmask, pjoin(subdir, f"sub-{args.sub}_r2_full.nii.gz"))
    save_ffa_map(r2_full, ffa_mask, n_vox, brainmask, pjoin(subdir, f"sub-{args.sub}_r2_full_raw.nii.gz"))
    save_beta_map(betas_full, ffa_mask, n_vox, brainmask, pjoin(betas_dir, f"sub-{args.sub}_betas_full.nii.gz"))

    r2_single = np.full((ndims, n_ffa), np.nan, dtype=np.float32)
    r2_single_raw = np.full((ndims, n_ffa), np.nan, dtype=np.float32)
    betas_single = np.full((ndims, n_ffa), np.nan, dtype=np.float32)
    best_single = np.full((ndims, n_ffa), np.nan, dtype=np.float32)
    for dim in range(ndims):
        r2_d, best_d, _ = fit_predict_cv(X[:, [dim]], y, session_vec, folds, fracs)
        betas_d = fit_betas_full_data(X[:, [dim]], y, best_d)[:, 0]
        r2_single_raw[dim] = r2_d.astype(np.float32)
        r2_single[dim] = (r2_d / nc_safe).astype(np.float32)
        betas_single[dim] = betas_d.astype(np.float32)
        best_single[dim] = best_d.astype(np.float32)
        save_ffa_map(r2_single[dim], ffa_mask, n_vox, brainmask, pjoin(single_dir, f"sub-{args.sub}_dim-{dim + 1:02d}_r2_single.nii.gz"))
        save_ffa_map(r2_single_raw[dim], ffa_mask, n_vox, brainmask, pjoin(single_dir, f"sub-{args.sub}_dim-{dim + 1:02d}_r2_single_raw.nii.gz"))
        save_ffa_map(betas_single[dim], ffa_mask, n_vox, brainmask, pjoin(single_dir, f"sub-{args.sub}_dim-{dim + 1:02d}_beta_single.nii.gz"))
        print(f"dim {dim + 1:02d}: mean CV R2/nc={np.nanmean(r2_single[dim]):.6f}")
    np.save(pjoin(single_dir, f"sub-{args.sub}_r2_single_ffa.npy"), r2_single)
    np.save(pjoin(single_dir, f"sub-{args.sub}_r2_single_raw_ffa.npy"), r2_single_raw)
    np.save(pjoin(single_dir, f"sub-{args.sub}_betas_single_ffa.npy"), betas_single)
    np.save(pjoin(single_dir, f"sub-{args.sub}_best_fracs_single_ffa.npy"), best_single)

    r2_face, best_face, _ = fit_predict_cv(face_vec, y, session_vec, folds, fracs)
    betas_face = fit_betas_full_data(face_vec, y, best_face)[:, 0]
    np.save(pjoin(subdir, f"sub-{args.sub}_r2_face_label_ffa.npy"), (r2_face / nc_safe).astype(np.float32))
    np.save(pjoin(subdir, f"sub-{args.sub}_r2_face_label_raw_ffa.npy"), r2_face.astype(np.float32))
    save_ffa_map(r2_face / nc_safe, ffa_mask, n_vox, brainmask, pjoin(subdir, f"sub-{args.sub}_r2_face_label.nii.gz"))
    save_ffa_map(r2_face, ffa_mask, n_vox, brainmask, pjoin(subdir, f"sub-{args.sub}_r2_face_label_raw.nii.gz"))
    np.save(pjoin(betas_dir, f"sub-{args.sub}_betas_face_label_ffa.npy"), betas_face.astype(np.float32))

    yz = zscore_all(y)
    contrast = np.full((ndims, n_ffa), np.nan, dtype=np.float32)
    contrast_meta = []
    for dim in range(ndims):
        order = np.argsort(X[:, dim])
        bottom = order[: args.top_k]
        top = order[-args.top_k :]
        contrast[dim] = (yz[top].mean(axis=0) - yz[bottom].mean(axis=0)).astype(np.float32)
        contrast_meta.append(
            {
                "dim": dim + 1,
                "top_k": args.top_k,
                "top_score_mean": float(X[top, dim].mean()),
                "bottom_score_mean": float(X[bottom, dim].mean()),
                "top_binary_face_count": int(face_vec[top].sum()),
                "bottom_binary_face_count": int(face_vec[bottom].sum()),
            }
        )
        save_ffa_map(contrast[dim], ffa_mask, n_vox, brainmask, pjoin(contrast_dir, f"sub-{args.sub}_dim-{dim + 1:02d}_topbottom_contrast.nii.gz"))
    np.save(pjoin(contrast_dir, f"sub-{args.sub}_topbottom-k{args.top_k}_contrast_ffa.npy"), contrast)
    pd.DataFrame(contrast_meta).to_csv(pjoin(contrast_dir, f"sub-{args.sub}_topbottom-k{args.top_k}_metadata.csv"), index=False)

    pd.DataFrame(
        [
            {
                "sub": args.sub,
                "folds": ";".join(",".join(map(str, f.tolist())) for f in folds),
                "holdout_sessions": args.holdout_sessions,
                "top_k": args.top_k,
                "x_zscored": True,
                "x_zscore_source": "fold_train_for_cv_full_trainpool_for_final_betas",
                "y_zscored": True,
                "y_zscore_source": "fold_train_and_fold_test_separately_for_cv_full_trainpool_for_final_betas",
                "r2_cv_aggregation": "pooled_out_of_fold",
                "fracridge_normalize": False,
                "nc_source": "nc_singletrial",
            }
        ]
    ).to_csv(pjoin(subdir, f"sub-{args.sub}_leave3_metadata.csv"), index=False)
    print(f"Finished sub-{args.sub}; saved to {subdir}")


if __name__ == "__main__":
    main()

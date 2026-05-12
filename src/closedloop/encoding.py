import numpy as np
from tqdm import tqdm
from closedloop.data import get_thingsimages_fnames, ThingsmriLoader
from closedloop.maths import regress_out, pearsonr_nd
from os.path import basename, pardir
from fracridge import FracRidgeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import pandas as pd
from nilearn.masking import unmask
import os
from os.path import join as pjoin
from scipy.stats import zscore


class FracRidgeVoxelwise:
    """
    A class for performing voxel-wise fractional ridge regression on fMRI data.

    This class encapsulates the functionality needed to tune, fit, and evaluate
    fractional ridge regression models on a per-voxel basis in fMRI datasets. It
    allows for tuning of the regularization parameter (alpha) using K-Fold
    cross-validation, fitting the final model, and evaluating its performance.
    Optionally, it can compute partial correlations between predicted and observed
    responses after accounting for other predictors.

    Parameters:
    - n_splits (int): Number of splits for K-Fold cross-validation.
    - test_size (float, optional): Proportion of the dataset to include in the test split.
      If set to 0, both training and testing are performed on the entire dataset.
      Defaults to 1/12.
    - fracs (np.ndarray, optional): Array of fractional values to be used for ridge
      regularization. Defaults to np.arange(0.01, 1.01, 0.01).
    - run_pcorr (bool, optional): If True, computes partial correlations after model fitting.
      Defaults to False.
    - fracridge_kws (dict, optional): Additional keyword arguments to be passed to the
      FracRidgeRegressor. Defaults to {"fit_intercept": True, "normalize": True}.

    Methods:
    - tune_alpha: Tunes the regularization parameter (alpha) for the model.
    - fit_evaulate: Fits the model and evaluates its performance on test data.
    - fit_pcorrs: Computes partial correlations between predicted and observed responses.
    - tune_and_eval: A high-level method that combines tuning, fitting, and evaluating.

    The class is designed to be used with fMRI data, where each voxel's response is predicted
    from a set of features (e.g., stimulus properties). The fractional ridge approach allows
    for fine-grained control over the amount of regularization applied to each voxel's model.
    """

    def __init__(
        self,
        n_splits,
        test_size=0.0833,  # = 1/12
        fracs=np.arange(0.01, 1.01, 0.01),
        run_pcorr=False,
        fracridge_kws={"fit_intercept": True, "normalize": True},
    ):
        self.n_splits = n_splits
        self.fracs = fracs
        self.nfracs = len(fracs)
        self.fr = FracRidgeRegressor(fracs=fracs, **fracridge_kws)
        self.test_size = test_size
        self.run_pcorr = run_pcorr

    def tune_alpha(self, X_train, y_train):
        nvox = y_train.shape[1]
        kfold = KFold(n_splits=self.n_splits, shuffle=False)
        r2s = np.zeros((self.n_splits, self.nfracs, nvox))
        for fold_i, (tune_inds, val_inds) in tqdm(
            enumerate(kfold.split(X_train)),
            desc="tuning regularization parameter",
            total=self.n_splits,
        ):
            X_tune, y_tune = X_train[tune_inds], y_train[tune_inds]
            X_val, y_val = X_train[val_inds], y_train[val_inds]
            self.fr.fit(X_tune, y_tune)
            pred = self.fr.predict(X_val)
            for frac_i in range(self.nfracs):
                r2s[fold_i, frac_i] = r2_score(
                    y_val, pred[:, frac_i, :], multioutput="raw_values"
                )
        r2_train = r2s.mean(0)
        best_fracinds = np.argmax(r2_train, axis=0)
        best_fracs = self.fracs[best_fracinds]
        return best_fracs, r2_train

    def fit_evaulate(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        best_fracs,
    ):
        nvox, ndims = y_train.shape[1], X_train.shape[1]
        betas = np.zeros((nvox, ndims))
        r2_test = np.zeros(nvox)
        unique_fracs = np.unique(best_fracs)
        for frac in tqdm(
            unique_fracs,
            desc="fitting final model (for each relevant frac parameter)",
            total=len(unique_fracs),
        ):
            vox_is = np.where(best_fracs == frac)[0]
            self.fr.fracs = frac
            self.fr.fit(X_train, y_train[:, vox_is])
            betas_ = self.fr.coef_.T
            betas[vox_is] = betas_
            pred = self.fr.predict(X_test)
            r2_test[vox_is] = r2_score(
                y_test[:, vox_is], pred, multioutput="raw_values"
            )
        return betas, r2_test

    def fit_pcorrs(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        best_fracs,
    ):
        # TODO: integrate with fit_evaluate, we don't have to do the voxel indexing twice.
        print(
            "\nDetermining partial correlations. Make sure your X and y are standardized.\n"
        )
        nvox, ndims = y_train.shape[1], X_train.shape[1]
        pcorrs = np.zeros((nvox, ndims))
        unique_fracs = np.unique(best_fracs)
        for frac in tqdm(
            unique_fracs,
            desc="Voxel sets (depending on frac)",
            total=len(unique_fracs),
            leave=True,
        ):
            # select relevant voxels
            self.fr.fracs = frac
            vox_is = np.where(best_fracs == frac)[0]
            y_train_thisfrac, y_test_thisfrac = y_train[:, vox_is], y_test[:, vox_is]
            for pred_i in tqdm(
                range(ndims), total=ndims, desc="predictors", leave=False
            ):
                # partial out other predictors from both this predictor and the response
                X_train_nuisance, X_test_nuisance = np.delete(
                    X_train, pred_i, axis=1
                ), np.delete(X_test, pred_i, axis=1)
                X_train_ = regress_out(
                    X_train_nuisance, X_train[:, pred_i].reshape(-1, 1)
                )
                X_test_ = regress_out(X_test_nuisance, X_test[:, pred_i].reshape(-1, 1))
                y_train_ = regress_out(X_train_nuisance, y_train_thisfrac)
                y_test_ = regress_out(X_test_nuisance, y_test_thisfrac)
                # fit model
                self.fr.fit(X_train_, y_train_)
                # determine correlation
                pred = self.fr.predict(X_test_)
                pcorrs[vox_is, pred_i] = pearsonr_nd(y_test_, pred)
        return pcorrs

    def tune_and_eval(self, X, y):
        """Tune alpha and fit final model"""
        if self.test_size:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y
        best_fracs, r2_train = self.tune_alpha(X_train, y_train)
        betas, r2_test = self.fit_evaulate(X_train, X_test, y_train, y_test, best_fracs)
        if self.run_pcorr:
            pcorrs = self.fit_pcorrs(X_train, X_test, y_train, y_test, best_fracs)
        else:
            pcorrs = np.zeros(r2_test.shape)
        return betas, r2_train, r2_test, best_fracs, pcorrs


def _find_trial_indices_in_embedding(
    stimdata_df, thingsimage_fnames=get_thingsimages_fnames()
):
    thingsimage_fnames = np.array([basename(fn) for fn in thingsimage_fnames])
    trial_indices = []
    for target in tqdm(stimdata_df.stimulus, "finding trial indices"):
        hits = np.where(thingsimage_fnames == target)
        assert len(hits[0]) == 1, f"Found {target} in things-images {len(hits)} times"
        trial_indices.append(hits[0][0])
    return np.array(trial_indices)


def embedding_to_X(
    embedding: np.ndarray, stimdata_df: pd.DataFrame, faces_onehot: np.ndarray = None
):
    trial_indices = _find_trial_indices_in_embedding(stimdata_df)
    X = embedding[trial_indices]
    if faces_onehot:
        nonface_trial_mask = np.invert(faces_onehot[trial_indices])
        X = X[nonface_trial_mask]
        n_excluded = np.sum(np.invert(nonface_trial_mask))
        print(f"excluded {n_excluded} trials with faces")
        return X, nonface_trial_mask
    else:
        return X


def fit_embedding_to_subject(
    embedding,
    thingsmri_sub,
    exclude_facetrials=False,
    zscore_y=True,
    zscore_x=True,
    frvox_kws=dict(n_splits=11, fracs=np.arange(0.1, 1.01, 0.1)),
    outbasedir=pjoin(pardir, pardir, "results", "encoding"),
    thingsmri_dir=pjoin(pardir, pardir, "data", "thingsmri"),
):
    """
    Fits an embedding model to a subject's fMRI data from the ThingsMRI dataset.

    This function processes the embedding and fMRI data, applies fractional ridge
    regression to predict brain responses, and saves the results to disk. It supports
    options to exclude trials with faces, z-score normalization of both features and
    responses, and customization of the fractional ridge regression parameters.

    Parameters:
    - embedding (np.ndarray): A numpy array containing the embedding data.
    - thingsmri_sub (str): The subject identifier in the ThingsMRI dataset.
    - exclude_facetrials (bool, optional): If True, trials with faces are excluded from the analysis. Defaults to False.
    - zscore_y (bool, optional): If True, applies z-score normalization to the response data (y). Defaults to True.
    - zscore_x (bool, optional): If True, applies z-score normalization to the feature data (X). Defaults to True.
    - frvox_kws (dict, optional): A dictionary of keyword arguments for the FracRidgeVoxelwise class.
      Defaults to {'n_splits': 11, 'fracs': np.arange(0.1, 1.01, 0.1)}.
    - outbasedir (str, optional): Base directory for saving output results. Defaults to two levels up from the current directory in 'results/encoding'.
    - thingsmri_dir (str, optional): Directory containing the ThingsMRI dataset. Defaults to two levels up from the current directory in 'data/thingsmri'.

    Returns:
    - None: The function saves the results to disk and does not return any value.

    The output files, including beta weights, training and testing R-squared values, best fractional ridge parameters, and partial correlations,
    are saved in the NIfTI format in the specified output directory.
    """
    outdir = pjoin(outbasedir, f"sub-{thingsmri_sub}")
    print(f"making results directory {outdir}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("loading fmri data")
    mriloader = ThingsmriLoader(thingsmri_dir)
    responses, stimdata, _ = mriloader.load_responses(thingsmri_sub)
    y = responses.drop(columns="voxel_id").to_numpy().T

    print("making design matrix")
    if not exclude_facetrials:
        X = embedding_to_X(embedding, stimdata)
    else:
        from closedloop.facelabels import load_facelabels

        print("loading face labels")
        _, _, _, faces_onehot = load_facelabels()
        print("excluding trials with faces from X")
        X, nonface_trial_mask = embedding_to_X(
            embedding,
            stimdata,
            faces_onehot=faces_onehot,
        )
        print("excluding trials with faces from y")
        y = y[nonface_trial_mask]

    if zscore_x:
        print("zscoring X")
        X = zscore(X, axis=0)
    if zscore_y:
        print("zscoring y")
        y = zscore(y, axis=0)

    print("start fitting fracridge")
    frvox = FracRidgeVoxelwise(**frvox_kws)
    betas, r2_train, r2_test, best_fracs, pcorrs = frvox.tune_and_eval(X, y)

    print("unmasking files and saving to disk")
    bm = mriloader.get_brainmask(thingsmri_sub)
    for arr, nam in zip(
        [betas.T, r2_train, r2_test, best_fracs, pcorrs.T],
        ["betas", "r2_train", "r2_test", "best_fracs", "pcorrs"],
    ):
        img = unmask(arr, bm)
        img.to_filename(pjoin(outdir, f"sub-{thingsmri_sub}_{nam}.nii.gz"))
    return None

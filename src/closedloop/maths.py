import numpy as np
from typing import Union
from tqdm import tqdm
import warnings
from itertools import combinations
from sklearn.linear_model import LinearRegression

def columnwise_correlation(A, B):
    """
    calculate correlation between each column in matrix A with each column in B
    """
    # center
    A = A - np.mean(A, axis=0)
    B = B - np.mean(B, axis=0)
    # Calculate norms
    norm_A = np.linalg.norm(A, axis=0)
    norm_B = np.linalg.norm(B, axis=0)
    # Outer product of norms
    norm_product = np.outer(norm_A, norm_B)
    # Calculate the correlation matrix
    corrs = (A.T @ B) / norm_product
    return corrs

def vectorized_pearsonr(base: np.ndarray, comp: np.ndarray) -> Union[float, np.ndarray]:
    """Alterntive to scipy.stats.pearsonr that is vectorized over the first dimension for
    fast pairwise correlation calculation."""
    if base.shape != comp.shape:
        raise ValueError(
            "Input arrays must have the same dimensions; "
            f"base.shape = {base.shape}, comp.shape = {comp.shape}"
        )
    if base.ndim < 2:
        base = base[:, None]
    if comp.ndim < 2:
        comp = comp[:, None]
    n = base.shape[1]
    covariance = np.cov(base.T, comp.T, ddof=1)
    base_std = np.sqrt(covariance[:n, :n].diagonal())
    comp_std = np.sqrt(covariance[n:, n:].diagonal())
    pearson_r = covariance[:n, n:] / np.outer(base_std, comp_std)
    return pearson_r


def compute_all_pairwise_corrmats(mats):
    """
    Compute pairwise correlation matrices between all members in a list of matrices.

    Args:
        mats (list): List of 2D arrays. Length of second dimension must be identical across all arrays.

    Returns:
        corrmats (list): List of corrleation matrices. Length is n(n-1)/2, where n is the length of mats.
    """
    
    corrmats = [
        columnwise_correlation(mats[i], mats[j]) 
        for i, j in combinations(range(len(mats)), 2)
        ]
    return corrmats


def regress_out(
        x: np.ndarray, y: np.ndarray, dtype=np.single,
        lr_kws: dict = dict(copy_X=True, fit_intercept=True, n_jobs=-1)
) -> np.ndarray:
    reg = LinearRegression(**lr_kws)
    reg.fit(x, y)
    resid = y - reg.predict(x)
    return resid.astype(dtype)


def pearsonr_nd(arr1: np.ndarray, arr2: np.ndarray, alongax: int = 0) -> np.ndarray:
    """
    Pearson correlation between respective variables in two arrays.
    arr1 and arr2 are 2d arrays with the same number of columns. Rows correspond to observations, columns to variables.
    Returns:
        correlations: np.ndarray (shape nvariables)
    """
    # center each feature
    arr1_c = arr1 - arr1.mean(axis=alongax)
    arr2_c = arr2 - arr2.mean(axis=alongax)
    # get sum of products for each voxel (numerator)
    numerators = np.sum(arr1_c * arr2_c, axis=alongax)
    # denominator
    arr1_sds = np.sqrt(np.sum(arr1_c**2, axis=alongax))
    arr2_sds = np.sqrt(np.sum(arr2_c**2, axis=alongax))
    denominators = arr1_sds * arr2_sds
    # for many voxels, this division will raise RuntimeWarnings for divide by zero. Ignore these.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = numerators / denominators
    return r
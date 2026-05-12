#!/usr/bin/env python
"""
fMRI pattern correlation analysis within a defined ROI.

Analyzes fMRI data (betas/contrasts) to assess representational structure of experimental 
conditions using odd/even imageset splits. Calculates:

1. Within-condition correlation: Pattern reliability across splits
2. Between-condition correlation: Cross-split or same-split methods  
3. Selectivity indices: Simple, contrast, separability, distinctiveness
4. Optional Fisher Z-transform for correlations
5. Bootstrap confidence intervals or permutation testing

Supports multiple similarity metrics (Pearson, Spearman, Euclidean variants).
Requires --splitimageset approach for independent data splits.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean
from sklearn.utils import shuffle as sk_shuffle

from closedloop.fmri.roi import apply_roi_mask
from closedloop.fmri.analyze_experiment import DataLoaderExperiment


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
# Define metrics calculated within run_pattern_correlation
SELECTIVITY_METRICS = [
    'simple_selectivity',
    'contrast_index',
    'separability',
    'distinctiveness'
]
NON_SELECTIVITY_METRICS = [
    'within_corrs',
    'mean_between_corrs',
    'pairwise_corrs'
]
ALL_METRICS = NON_SELECTIVITY_METRICS + SELECTIVITY_METRICS


def compute_voxel_reliability(
    betas: np.ndarray,
    condinfo: pd.DataFrame,
    min_conditions: int = 2,
) -> np.ndarray:
    """
    Compute split-half voxel-wise reliability using odd/even imagesets.

    For each voxel, we:
    - Compute condition-mean responses separately for odd and even imagesets.
    - Correlate the odd vs. even condition vectors across all conditions that
      have data in both splits.

    Returns
    -------
    reliability : np.ndarray, shape (n_voxels,)
        Pearson r per voxel. Voxels with insufficient data or zero variance
        in either split are returned as NaN.
    """
    if betas is None or betas.size == 0:
        return np.array([])

    if 'dim' not in condinfo.columns or 'imageset' not in condinfo.columns:
        logger.warning(
            "condinfo missing required columns 'dim' and/or 'imageset'; "
            "cannot compute voxel reliability. Returning NaNs."
        )
        return np.full(betas.shape[1], np.nan)

    conditions = sorted(condinfo['dim'].unique())
    odd_mask_all = condinfo['imageset'] == 'odd'
    even_mask_all = condinfo['imageset'] == 'even'

    patterns_odd = []
    patterns_even = []

    for cond in conditions:
        cond_mask = condinfo['dim'] == cond
        idx_odd = condinfo.index[cond_mask & odd_mask_all]
        idx_even = condinfo.index[cond_mask & even_mask_all]

        if len(idx_odd) == 0 or len(idx_even) == 0:
            continue

        patterns_odd.append(betas[idx_odd].mean(axis=0))
        patterns_even.append(betas[idx_even].mean(axis=0))

    if not patterns_odd or not patterns_even:
        logger.warning(
            "No conditions with data in both odd and even imagesets; "
            "cannot compute voxel reliability. Returning NaNs."
        )
        return np.full(betas.shape[1], np.nan)

    patterns_odd = np.vstack(patterns_odd)
    patterns_even = np.vstack(patterns_even)

    n_conditions_effective = patterns_odd.shape[0]
    if n_conditions_effective < min_conditions:
        logger.warning(
            f"Only {n_conditions_effective} conditions with data in both splits; "
            "fewer than min_conditions for reliability. Returning NaNs."
        )
        return np.full(betas.shape[1], np.nan)

    # Vectorized Pearson correlation across conditions for each voxel
    odd_mean = np.nanmean(patterns_odd, axis=0)
    even_mean = np.nanmean(patterns_even, axis=0)
    odd_centered = patterns_odd - odd_mean
    even_centered = patterns_even - even_mean

    numerator = np.nansum(odd_centered * even_centered, axis=0)
    odd_var = np.nansum(odd_centered ** 2, axis=0)
    even_var = np.nansum(even_centered ** 2, axis=0)
    denom = np.sqrt(odd_var * even_var)

    with np.errstate(invalid="ignore", divide="ignore"):
        reliability = numerator / denom

    reliability[denom == 0] = np.nan
    return reliability

def preprocess_zscore_voxels(betas: np.ndarray) -> np.ndarray:
    """Z-scores each voxel's activity across samples."""
    logger.info(f"Z-scoring voxels (columns) across samples (rows)...")
    scaler = StandardScaler()
    # StandardScaler works column-wise by default.
    betas_zscored = scaler.fit_transform(betas)
    logger.info(f"Finished z-scoring voxels.")
    return betas_zscored

def _get_split_patterns(
    betas: np.ndarray,
    condinfo: pd.DataFrame,
    condition: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculates average spatial patterns for a condition in 'odd' and 'even' imagesets.
    
    Returns: (pattern_odd, pattern_even) - average patterns or None if no data.
    """
    idx_odd = condinfo[(condinfo['dim'] == condition) & (condinfo['imageset'] == 'odd')].index
    idx_even = condinfo[(condinfo['dim'] == condition) & (condinfo['imageset'] == 'even')].index
    
    pattern_odd = np.mean(betas[idx_odd], axis=0) if len(idx_odd) > 0 else None
    pattern_even = np.mean(betas[idx_even], axis=0) if len(idx_even) > 0 else None
    
    # Ensure patterns are 1D arrays
    if pattern_odd is not None and pattern_odd.ndim == 0: pattern_odd = np.array([pattern_odd])
    if pattern_even is not None and pattern_even.ndim == 0: pattern_even = np.array([pattern_even])

    return pattern_odd, pattern_even

def calculate_similarity(x: Optional[np.ndarray], y: Optional[np.ndarray], metric: str = 'pearson') -> float:
    """
    Calculates similarity/dissimilarity between two vectors.
    
    Metrics: 'pearson', 'spearman', 'euclidean_distance', 'euclidean_inverse', 'euclidean_rbf'
    Returns np.nan for invalid inputs (None, mismatched shapes, constant vectors for correlations).
    """
    if x is None or y is None or x.shape[0] < 1 or y.shape[0] < 1 or x.shape != y.shape:
        # Allow single-point vectors for distance, but not < 1
        if x is not None and metric in ['pearson', 'spearman'] and x.shape[0] < 2:
             return np.nan # Correlations need at least 2 points
        elif x is None or y is None or x.shape != y.shape:
             return np.nan # Mismatched shapes or None are always invalid

    # Handle constant vectors
    # Correlations are undefined for constant vectors
    is_constant_x = np.all(x == x[0]) if x.shape[0] > 0 else True
    is_constant_y = np.all(y == y[0]) if y.shape[0] > 0 else True
    if metric in ['pearson', 'spearman'] and (is_constant_x or is_constant_y):
        return np.nan
    # Euclidean distance between identical constant vectors is 0, which is valid.
    # Check added below for distance between non-identical constant vectors.

    try:
        if metric == 'pearson':
            # Clipping for numerical stability with arctanh later
            clip_val = 1.0 - 1e-9 # Slightly less than 1
            x_clipped = np.clip(x, -clip_val, clip_val)
            y_clipped = np.clip(y, -clip_val, clip_val)
            r, _ = pearsonr(x_clipped, y_clipped)
            value = r
        elif metric == 'spearman':
            clip_val = 1.0 - 1e-9 # Slightly less than 1
            x_clipped = np.clip(x, -clip_val, clip_val)
            y_clipped = np.clip(y, -clip_val, clip_val)
            r, _ = spearmanr(x_clipped, y_clipped)
            value = r
        elif metric.startswith('euclidean'):
            if is_constant_x and is_constant_y and x[0] != y[0]:
                # Handle non-identical constant vectors for distance
                # Use np.linalg.norm which handles scalar differences correctly
                dist = np.linalg.norm(x - y)
            elif x.shape[0] == 0: # Handle empty vectors
                 return np.nan
            else:
                # Use scipy's euclidean distance for potentially higher dimension vectors
                dist = euclidean(x, y)

            if metric == 'euclidean_distance':
                value = dist
            elif metric == 'euclidean_inverse':
                value = 1.0 / (1.0 + dist)
            elif metric == 'euclidean_rbf':
                # Use gamma = 1 / n_features as a common default
                gamma = 1.0 / x.shape[0] if x.shape[0] > 0 else 1.0
                value = np.exp(-gamma * (dist ** 2))
            else:
                raise ValueError(f"Unsupported Euclidean metric variant: {metric}")

        else:
            raise ValueError(f"Unsupported similarity/dissimilarity metric: {metric}")

    except ValueError as e: # Catch potential errors from correlation/distance functions
        logger.error(f"Error calculating metric '{metric}' between vectors: {e}")
        return np.nan

    return value if np.isfinite(value) else np.nan

def run_pattern_correlation(
    betas: np.ndarray,
    condinfo: pd.DataFrame,
    labels: np.ndarray,
    same_split: bool = False,
    fisher_z: bool = True,
    similarity_metric: str = 'pearson'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Performs pairwise pattern similarity analysis using imageset splits.

    Calculates within-condition (reliability) and between-condition similarities based on
    'odd'/'even' imageset splits. Applies Fisher-Z transform if requested for correlations.
    Computes selectivity indices: simple, contrast, separability, distinctiveness.
    
    Cross-split (default): mean(sim(i_odd, j_even), sim(i_even, j_odd))
    Same-split: mean(sim(i_odd, j_odd), sim(i_even, j_even))
    
    Returns: within_metrics, mean_between_metrics, pairwise_metrics,
             simple_selectivity, contrast_index, separability, distinctiveness, condition_labels
    """
    conditions = sorted(condinfo['dim'].unique())
    n_conditions = len(conditions)

    within_metrics = np.full(n_conditions, np.nan)
    mean_between_metrics = np.full(n_conditions, np.nan)
    pairwise_metrics = np.full((n_conditions, n_conditions), np.nan)
    
    simple_selectivity = np.full(n_conditions, np.nan)
    contrast_index = np.full(n_conditions, np.nan)
    separability = np.full(n_conditions, np.nan)
    distinctiveness = np.full(n_conditions, np.nan)

    condition_patterns_odd = {}
    condition_patterns_even = {}
    condition_labels_map = {}

    # Pre-calculate average patterns for each condition and split
    logger.info("Calculating average patterns for odd/even splits...")
    for i, cond in enumerate(conditions):
        pattern_odd, pattern_even = _get_split_patterns(betas, condinfo, cond)
        condition_patterns_odd[cond] = pattern_odd
        condition_patterns_even[cond] = pattern_even
        try:
            condition_labels_map[cond] = labels[int(cond)-1]
        except IndexError:
            condition_labels_map[cond] = f"Cond{cond}"
        
        if pattern_odd is None or pattern_even is None:
            logger.warning(f"Condition {cond} ('{condition_labels_map[cond]}') missing data in odd or even split. Within-metric will be NaN.")

    # Calculate metrics
    logger.info(f"Calculating within- and between-condition metrics using '{similarity_metric}'...")
    is_correlation_metric = similarity_metric in ['pearson', 'spearman']

    for i, cond1 in enumerate(conditions):
        pattern1_odd = condition_patterns_odd[cond1]
        pattern1_even = condition_patterns_even[cond1]

        # Calculate within-condition metric (reliability)
        within_metric = calculate_similarity(pattern1_odd, pattern1_even, metric=similarity_metric)
        if fisher_z and is_correlation_metric and not np.isnan(within_metric):
            # Apply Fisher-Z only to correlation metrics
            within_metric = np.arctanh(np.clip(within_metric, -1 + 1e-9, 1 - 1e-9))

        within_metrics[i] = within_metric
        pairwise_metrics[i, i] = within_metric # Store on diagonal

        current_between_metrics = []
        for j, cond2 in enumerate(conditions):
            if i == j:
                continue

            pattern2_odd = condition_patterns_odd[cond2]
            pattern2_even = condition_patterns_even[cond2]

            # Calculate between-condition metric
            if not same_split: # Default: Cross-split
                metric_oe = calculate_similarity(pattern1_odd, pattern2_even, metric=similarity_metric)
                metric_eo = calculate_similarity(pattern1_even, pattern2_odd, metric=similarity_metric)
                if fisher_z and is_correlation_metric:
                    if not np.isnan(metric_oe): metric_oe = np.arctanh(np.clip(metric_oe, -1 + 1e-9, 1 - 1e-9))
                    if not np.isnan(metric_eo): metric_eo = np.arctanh(np.clip(metric_eo, -1 + 1e-9, 1 - 1e-9))
                valid_metrics = [m for m in [metric_oe, metric_eo] if not np.isnan(m)]
                between_metric = np.mean(valid_metrics) if valid_metrics else np.nan
            else: # Same-split
                metric_odd = calculate_similarity(pattern1_odd, pattern2_odd, metric=similarity_metric)
                metric_even = calculate_similarity(pattern1_even, pattern2_even, metric=similarity_metric)
                if fisher_z and is_correlation_metric:
                    if not np.isnan(metric_odd): metric_odd = np.arctanh(np.clip(metric_odd, -1 + 1e-9, 1 - 1e-9))
                    if not np.isnan(metric_even): metric_even = np.arctanh(np.clip(metric_even, -1 + 1e-9, 1 - 1e-9))
                valid_metrics = [m for m in [metric_odd, metric_even] if not np.isnan(m)]
                between_metric = np.mean(valid_metrics) if valid_metrics else np.nan

            pairwise_metrics[i, j] = between_metric
            if not np.isnan(between_metric):
                current_between_metrics.append(between_metric)

        # Calculate summary stats for between-metrics for cond1
        if current_between_metrics:
            mean_between_metrics[i] = np.mean(current_between_metrics)
        else:
             logger.warning(f"No valid between-condition metrics found for condition {cond1} ('{condition_labels_map[cond1]}').")


    # <<< START INVERSION BLOCK >>>
    # Invert distance metrics so higher = better before calculating selectivity
    if similarity_metric == 'euclidean_distance':
        logger.info("Inverting Euclidean distance metrics for consistent interpretation (higher = better).")
        within_metrics *= -1
        mean_between_metrics *= -1
        pairwise_metrics *= -1
    # <<< END INVERSION BLOCK >>>

    # Calculate selectivity indices
    logger.info("Calculating selectivity indices...")
    for i in range(n_conditions):
        within = within_metrics[i]
        mean_between = mean_between_metrics[i]

        if not np.isnan(within) and not np.isnan(mean_between):
            simple_selectivity[i] = within - mean_between

            # Avoid division by zero
            denominator_contrast = within + mean_between
            if not np.isclose(denominator_contrast, 0):
                 contrast_index[i] = simple_selectivity[i] / denominator_contrast
            else:
                 contrast_index[i] = np.nan

            denominator_separability = 1.0 - mean_between
            if not np.isclose(denominator_separability, 0):
                separability[i] = simple_selectivity[i] / denominator_separability
            else:
                separability[i] = np.nan

            denominator_distinct = within
            if not np.isclose(denominator_distinct, 0):
                distinctiveness[i] = simple_selectivity[i] / denominator_distinct
            else:
                 # Handle case where within is 0
                 if not np.isnan(mean_between) and np.isclose(mean_between, 0):
                     distinctiveness[i] = 0.0
                 elif not np.isnan(mean_between) and not np.isclose(mean_between, 0):
                      distinctiveness[i] = np.nan
                 else:
                      distinctiveness[i] = np.nan

    condition_labels_list = [condition_labels_map[cond] for cond in conditions]

    # Return potentially inverted metrics and derived selectivity indices
    return (within_metrics, mean_between_metrics, pairwise_metrics,
            simple_selectivity, contrast_index, separability, distinctiveness,
            condition_labels_list)


def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    title: str,
    filename: Union[str, Path],
    labels: List[str],
    cmap: str = "coolwarm", # Diverging colormap
    vmin: float = -1.0,
    vmax: float = 1.0
) -> None:
    """
    Plots and saves correlation heatmap with annotations and diverging colormap.
    """
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    # Optional: Mask upper triangle if you only want to show lower triangle + diagonal
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=.5,
        cbar=True,
        square=True,
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        center=0.0 # Center colormap at 0
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_selectivity_bar(
    selectivity_values: np.ndarray,
    condition_labels: List[str],
    title: str,
    ylabel: str,
    output_file: Union[str, Path]
) -> None:
    """
    Generates bar plot of selectivity values per condition, sorted in descending order.
    """
    valid_indices = ~np.isnan(selectivity_values)
    if not np.any(valid_indices):
        logger.warning(f"All selectivity values are NaN for '{title}'. Cannot plot.")
        plt.figure(figsize=(10, 6))
        plt.title(f"{title}(No valid data)")
        plt.savefig(output_file)
        plt.close()
        return

    values_to_plot = selectivity_values[valid_indices]
    labels_to_plot = [condition_labels[i] for i in np.where(valid_indices)[0]]
    
    sort_idx = np.argsort(values_to_plot)[::-1]
    sorted_labels = [labels_to_plot[i] for i in sort_idx]
    sorted_values = values_to_plot[sort_idx]

    x_coords = np.arange(len(sorted_labels))
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x=x_coords, y=sorted_values, color='skyblue', edgecolor='black')
    plt.xlabel("Condition")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x_coords, sorted_labels, rotation=60, ha='right')
    # Add a horizontal line at 0 for reference
    plt.axhline(0, color='grey', linestyle='--') 
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def _run_bootstrap_iteration(
    iteration: int,
    betas: np.ndarray,
    condinfo: pd.DataFrame,
    labels: np.ndarray,
    conditions: List[int],
    same_split: bool,
    fisher_z: bool,
    similarity_metric: str
) -> Tuple[Dict[str, np.ndarray], Optional[List[str]]]:
    """
    Run single bootstrap iteration: resample within condition/imageset and compute metrics.
    """
    # Create bootstrapped data by resampling within each condition and imageset
    bootstrapped_condinfo = condinfo.copy()
    bootstrapped_betas = np.empty_like(betas)

    for cond in conditions:
        for split in ['odd', 'even']:
            indices = condinfo[(condinfo['dim'] == cond) & (condinfo['imageset'] == split)].index
            if len(indices) > 0:
                # Resample with replacement
                resampled_indices = np.random.choice(indices, size=len(indices), replace=True)
                # Map original indices to their positions in bootstrapped_betas
                positions = np.where(np.isin(condinfo.index, indices))[0]
                if len(positions) == len(indices):
                     for i, pos in enumerate(positions):
                          bootstrapped_betas[pos] = betas[resampled_indices[i % len(resampled_indices)]]
                else:
                     logger.error(f"Bootstrap Error: Mismatch for cond {cond}, split {split}.")
                     return None, None # Indicate error

    # Run pattern analysis on bootstrapped data
    try:
        # Rename results variables for clarity
        (within_metrics, mean_between_metrics, pairwise_metrics,
         simple_selectivity, contrast_index, separability, distinctiveness,
         condition_labels) = run_pattern_correlation(
            bootstrapped_betas, bootstrapped_condinfo, labels,
            same_split=same_split, fisher_z=fisher_z,
            similarity_metric=similarity_metric
        )

        # Store results for this iteration using updated names
        results = {
            'within_corrs': within_metrics, # Keep keys consistent for outer function
            'mean_between_corrs': mean_between_metrics,
            'pairwise_corrs': pairwise_metrics,
            'simple_selectivity': simple_selectivity,
            'contrast_index': contrast_index,
            'separability': separability,
            'distinctiveness': distinctiveness
        }

        return results, condition_labels if iteration == 0 else None
    except Exception as e:
        logger.error(f"Error in bootstrap iteration {iteration}: {e}", exc_info=True)
        # For error handling, return None to indicate failure
        return None, None

def bootstrap_pattern_correlation(
    betas: np.ndarray,
    condinfo: pd.DataFrame,
    labels: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    same_split: bool = False,
    fisher_z: bool = True,
    similarity_metric: str = 'pearson',
    random_seed: Optional[int] = None,
    n_jobs: int = 10
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Performs bootstrapped pattern analysis to estimate confidence intervals.

    Resamples data with replacement within each condition/imageset, computes metrics,
    and calculates confidence intervals and significance (CI > 0 for selectivity).
    
    Returns: bootstrap_results, mean_results, ci_results, significance_results, condition_labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    conditions = sorted(condinfo['dim'].unique())
    n_conditions = len(conditions)
    condition_labels_list = []

    # Prepare storage for bootstrap results
    # Use ALL_METRICS defined globally (these keys remain the same for consistency)
    bootstrap_results = {
        metric: np.full((n_iterations, n_conditions), np.nan)
        for metric in ALL_METRICS if metric != 'pairwise_corrs'
    }
    bootstrap_results['pairwise_corrs'] = np.full((n_iterations, n_conditions, n_conditions), np.nan)

    # Run bootstrap iterations in parallel
    logger.info(f"Running {n_iterations} bootstrap iterations with {n_jobs} parallel jobs using '{similarity_metric}' metric...")

    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_bootstrap_iteration)(
            iteration, betas, condinfo, labels, conditions, same_split, fisher_z,
            similarity_metric=similarity_metric
        ) for iteration in tqdm(range(n_iterations), desc="Bootstrap Progress")
    )

    # Process results
    valid_iterations = 0
    for iteration, (result, cond_labels) in enumerate(parallel_results):
        if result is not None:
            valid_iterations += 1
            for metric in ALL_METRICS:
                if metric in result: # Check if metric was successfully calculated in the iteration
                    if metric == 'pairwise_corrs':
                         # Check shape consistency before assigning
                         if result[metric].shape == (n_conditions, n_conditions):
                             bootstrap_results[metric][iteration] = result[metric]
                         else:
                             logger.warning(f"Shape mismatch for pairwise_corrs in bootstrap iteration {iteration}. Expected {(n_conditions, n_conditions)}, got {result[metric].shape}. Filling with NaN.")
                             bootstrap_results[metric][iteration].fill(np.nan)
                    else:
                         # Ensure result[metric] is broadcastable and correct length
                         if len(result[metric]) == n_conditions:
                             bootstrap_results[metric][iteration, :] = result[metric]
                         else:
                              logger.warning(f"Length mismatch for {metric} in bootstrap iteration {iteration}. Expected {n_conditions}, got {len(result[metric])}. Filling with NaN.")
                              bootstrap_results[metric][iteration].fill(np.nan)


            if iteration == 0 and cond_labels is not None: # Get labels only once
                condition_labels_list = cond_labels
        # No 'else' needed, NaNs remain for failed iterations

    if valid_iterations < n_iterations:
         logger.warning(f"Completed {valid_iterations} valid bootstrap iterations out of {n_iterations} requested.")
    if valid_iterations == 0:
         logger.error("No bootstrap iterations completed successfully. Cannot calculate results.")
         # Return empty/NaN structures
         empty_means = {metric: np.full(n_conditions if metric != 'pairwise_corrs' else (n_conditions, n_conditions), np.nan) for metric in ALL_METRICS}
         empty_cis = {metric: {'lower': np.full(n_conditions if metric != 'pairwise_corrs' else (n_conditions, n_conditions), np.nan),
                               'upper': np.full(n_conditions if metric != 'pairwise_corrs' else (n_conditions, n_conditions), np.nan)} for metric in ALL_METRICS}
         empty_sig = {metric: np.full(n_conditions, False) for metric in SELECTIVITY_METRICS}
         return bootstrap_results, empty_means, empty_cis, empty_sig, condition_labels_list


    # <<< START INVERSION BLOCK >>>
    # Invert distance metric bootstrap results so higher = better before calculating stats
    if similarity_metric == 'euclidean_distance':
        logger.info("Inverting Euclidean distance bootstrap results (multiplying by -1) for consistent interpretation (higher = better).")
        for metric_key in ALL_METRICS:
            if metric_key in bootstrap_results:
                # NaNs remain NaNs after multiplication
                bootstrap_results[metric_key] *= -1
    # <<< END INVERSION BLOCK >>>

    # Calculate mean and confidence intervals using potentially inverted bootstrap data
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    # Calculate mean across bootstrap iterations
    mean_results = { # Iterate through all metric keys
        metric: np.nanmean(bootstrap_results[metric], axis=0)
        for metric in ALL_METRICS
    }

    # Calculate confidence intervals
    ci_results = { # Iterate through all metric keys
        metric: {
            'lower': np.nanpercentile(bootstrap_results[metric], lower_percentile, axis=0),
            'upper': np.nanpercentile(bootstrap_results[metric], upper_percentile, axis=0)
        }
        for metric in ALL_METRICS
    }

    # Calculate significance (lower CI > 0) specifically for selectivity metrics
    # This interpretation assumes higher = better. For euclidean_distance, significance might be defined differently (e.g., upper CI < some_threshold).
    # Sticking to lower CI > 0 for now, but interpretation depends on the metric.
    significance_results = {}
    for metric in SELECTIVITY_METRICS: # Use predefined list of selectivity metric keys
        if metric in ci_results: # Check if CIs were calculated
            lower_ci = ci_results[metric]['lower']
            # Ensure lower_ci is not NaN before comparison
            significance_results[metric] = (~np.isnan(lower_ci)) & (lower_ci > 0)
        else:
             significance_results[metric] = np.full(n_conditions, False) # Default to False if CIs missing

    return bootstrap_results, mean_results, ci_results, significance_results, condition_labels_list

def plot_selectivity_bar_with_ci(
    selectivity_values: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    condition_labels: List[str],
    title: str,
    ylabel: str,
    output_file: Union[str, Path]
) -> None:
    """
    Generates bar plot of selectivity indices with confidence intervals.
    """
    valid_indices = ~np.isnan(selectivity_values)
    if not np.any(valid_indices):
        logger.warning(f"All selectivity values are NaN for '{title}'. Cannot plot.")
        plt.figure(figsize=(10, 6))
        plt.title(f"{title}(No valid data)")
        plt.savefig(output_file)
        plt.close()
        return
    
    values_to_plot = selectivity_values[valid_indices]
    ci_lower_to_plot = ci_lower[valid_indices]
    ci_upper_to_plot = ci_upper[valid_indices]
    labels_to_plot = [condition_labels[i] for i in np.where(valid_indices)[0]]
    
    # Sort values in descending order
    sort_idx = np.argsort(values_to_plot)[::-1]
    sorted_labels = [labels_to_plot[i] for i in sort_idx]
    sorted_values = values_to_plot[sort_idx]
    sorted_ci_lower = ci_lower_to_plot[sort_idx]
    sorted_ci_upper = ci_upper_to_plot[sort_idx]
    
    x_coords = np.arange(len(sorted_labels))
    
    plt.figure(figsize=(12, 7))
    # Plot bars based on the mean of the bootstrap distribution
    plt.bar(x_coords, sorted_values, color='skyblue', edgecolor='black')
    
    # Plot error bars representing the percentile confidence interval (e.g., 2.5th to 97.5th).
    # Note: These CIs are not necessarily symmetric around the mean bar height.
    for i, (x, y, low, high) in enumerate(zip(x_coords, sorted_values, sorted_ci_lower, sorted_ci_upper)):
        plt.plot([x, x], [low, high], color='black', linewidth=1.5)  # Vertical line
        plt.plot([x-0.1, x+0.1], [low, low], color='black', linewidth=1.5)  # Lower cap
        plt.plot([x-0.1, x+0.1], [high, high], color='black', linewidth=1.5)  # Upper cap
    
    plt.xlabel("Condition")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x_coords, sorted_labels, rotation=60, ha='right')
    plt.axhline(0, color='grey', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# --- Permutation Testing Functions --- #

def _shuffle_condinfo_within_split(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to shuffle 'dim' column within a subset (one imageset)."""
    df = df.copy()
    # Shuffle the 'dim' column randomly
    df['dim'] = sk_shuffle(df['dim'].values, random_state=np.random.RandomState())
    return df

def _run_permutation_iteration(
    iteration: int,
    betas: np.ndarray,
    original_condinfo: pd.DataFrame,
    labels: np.ndarray,
    conditions: List[int], # List of unique conditions
    same_split: bool,
    fisher_z: bool,
    similarity_metric: str,
    shuffle_split: Optional[str] = None  # New parameter: 'odd', 'even', 'both', or None (both)
) -> Optional[Dict[str, np.ndarray]]: # Return only selectivity indices
    """
    Run single permutation iteration: shuffle condition labels within specified imageset(s).
    """
    try:
        # Create a deep copy to avoid modifying the original dataframe outside the scope
        shuffled_condinfo = original_condinfo.copy()

        # --- Shuffle condition labels ('dim') within specified imageset(s) --- #
        if shuffle_split is None or shuffle_split == "both":
            # Original behavior: shuffle both splits independently
            shuffled_dims = shuffled_condinfo.groupby('imageset')['dim']\
                                            .transform(lambda x: np.random.permutation(x))
            shuffled_condinfo['dim'] = shuffled_dims
        else:
            # New behavior: shuffle only the specified split
            if shuffle_split not in ['odd', 'even']:
                raise ValueError(f"shuffle_split must be 'odd', 'even', 'both', or None, got: {shuffle_split}")
            
            # Only shuffle the specified imageset
            mask = shuffled_condinfo['imageset'] == shuffle_split
            if mask.sum() > 0:  # Make sure we have data for this split
                original_dims = shuffled_condinfo.loc[mask, 'dim'].values
                shuffled_dims_subset = np.random.permutation(original_dims)
                shuffled_condinfo.loc[mask, 'dim'] = shuffled_dims_subset
            else:
                logger.warning(f"No data found for imageset '{shuffle_split}' in iteration {iteration}")

        # Run pattern analysis on original betas with shuffled labels
        # We only need the selectivity indices for the null distribution
        (_within, _mean_between, _pairwise,
         perm_simple, perm_contrast, perm_sep, perm_distinct,
         _cond_labels) = run_pattern_correlation(
            betas, shuffled_condinfo, labels, # Use original betas, shuffled condinfo
            same_split=same_split, fisher_z=fisher_z,
            similarity_metric=similarity_metric
        )

        # Return only the selectivity metrics needed for the null distribution
        perm_results = {
            'simple_selectivity': perm_simple,
            'contrast_index': perm_contrast,
            'separability': perm_sep,
            'distinctiveness': perm_distinct
        }
        return perm_results

    except Exception as e:
        logger.error(f"Error in permutation iteration {iteration}: {e}", exc_info=True)
        return None # Indicate failure

def permutation_test_pattern_correlation(
    betas: np.ndarray,
    condinfo: pd.DataFrame,
    labels: np.ndarray,
    n_permutations: int = 1000,
    same_split: bool = False,
    fisher_z: bool = True,
    similarity_metric: str = 'pearson',
    random_seed: Optional[int] = None,
    n_jobs: int = 10,
    shuffle_split: Optional[str] = None  # New parameter: 'odd', 'even', 'both', or None (both)
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Performs permutation testing to estimate p-values for selectivity indices.

    Shuffles condition labels within specified imageset(s), recalculates metrics,
    and compares true metrics to null distribution for p-values.
    
    Returns: true_results, p_values, condition_labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    conditions = sorted(condinfo['dim'].unique())
    n_conditions = len(conditions)

    shuffle_desc = f"shuffle_{shuffle_split}_only" if shuffle_split and shuffle_split != "both" else "shuffle_both_splits"
    logger.info(f"--- Running Permutation Test ({n_permutations} iterations, {shuffle_desc}) ---")

    # 1. Calculate True Metrics
    logger.info("Calculating true metrics...")
    try:
        (true_within, true_mean_between, true_pairwise,
         true_simple, true_contrast, true_sep, true_distinct,
         condition_labels_list) = run_pattern_correlation(
            betas, condinfo, labels,
            same_split=same_split, fisher_z=fisher_z,
            similarity_metric=similarity_metric
        )
        true_results = {
            'within_corrs': true_within, # Use consistent keys
            'mean_between_corrs': true_mean_between,
            'pairwise_corrs': true_pairwise,
            'simple_selectivity': true_simple,
            'contrast_index': true_contrast,
            'separability': true_sep,
            'distinctiveness': true_distinct
        }
    except Exception as e:
        logger.error(f"Failed to calculate true metrics: {e}", exc_info=True)
        raise # Re-raise the exception as we can't proceed

    # 2. Run Permutations in Parallel
    logger.info(f"Running {n_permutations} permutations with {n_jobs} parallel jobs...")

    # Prepare storage for permuted selectivity metrics
    permuted_results_storage = {
        metric: np.full((n_permutations, n_conditions), np.nan)
        for metric in SELECTIVITY_METRICS # Only store selectivity metrics
    }

    parallel_results = Parallel(n_jobs=n_jobs)( # Use n_jobs from args
        delayed(_run_permutation_iteration)(
            iteration, betas, condinfo, labels, conditions, same_split, fisher_z,
            similarity_metric, shuffle_split  # Pass the new parameter
        ) for iteration in tqdm(range(n_permutations), desc="Permutation Progress")
    )

    # Process results and store selectivity indices
    valid_permutations = 0
    for iteration, result in enumerate(parallel_results):
        if result is not None:
            valid_permutations += 1
            for metric_key in SELECTIVITY_METRICS:
                if metric_key in result and result[metric_key] is not None:
                     if len(result[metric_key]) == n_conditions:
                         permuted_results_storage[metric_key][iteration, :] = result[metric_key]
                     else:
                          logger.warning(f"Length mismatch for permuted {metric_key} in iteration {iteration}. Expected {n_conditions}, got {len(result[metric_key])}. Storing NaN.")
                          permuted_results_storage[metric_key][iteration, :].fill(np.nan)
                else:
                     permuted_results_storage[metric_key][iteration, :].fill(np.nan)

    if valid_permutations < n_permutations:
        logger.warning(f"Completed {valid_permutations} valid permutation iterations out of {n_permutations} requested.")
    if valid_permutations == 0:
        logger.error("No permutation iterations completed successfully. Cannot calculate p-values.")
        # Return true results but empty/NaN p-values
        nan_p_values = {metric: np.full(n_conditions, np.nan) for metric in SELECTIVITY_METRICS}
        return true_results, nan_p_values, condition_labels_list

    # 3. Calculate P-values
    logger.info("Calculating p-values...")
    p_values = {
        metric: np.full(n_conditions, np.nan) for metric in SELECTIVITY_METRICS
    }

    for metric_key in SELECTIVITY_METRICS:
        true_metric_values = true_results[metric_key]
        permuted_metric_distribution = permuted_results_storage[metric_key]

        for cond_idx in range(n_conditions):
            true_val = true_metric_values[cond_idx]
            if np.isnan(true_val):
                 p_values[metric_key][cond_idx] = np.nan
                 continue

            # Get the distribution for this condition, excluding NaNs from failed permutations
            perm_dist_cond = permuted_metric_distribution[:, cond_idx]
            perm_dist_cond_valid = perm_dist_cond[~np.isnan(perm_dist_cond)]
            num_valid_perms_for_cond = len(perm_dist_cond_valid)

            if num_valid_perms_for_cond == 0:
                 p_values[metric_key][cond_idx] = np.nan
                 continue

            # Calculate p-value: proportion of permutations >= true value
            # Add 1 to numerator and denominator for stability
            n_extreme_or_equal = np.sum(perm_dist_cond_valid >= true_val)
            if num_valid_perms_for_cond > 0: # Ensure no division by zero
                p_val = n_extreme_or_equal / num_valid_perms_for_cond
            else: # Should have been caught by the earlier check, but as a safeguard
                p_val = np.nan
            p_values[metric_key][cond_idx] = p_val

    return true_results, p_values, condition_labels_list

# --- End Permutation Testing Functions --- #

def main(
    project_root_str: str,
    sub: str,
    statstype: str,
    roi_name: str,
    same_split: bool,
    fisher_z: bool,
    zscore_voxels: bool,
    # Updated args for bootstrap/permutation
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    permutation: bool = False,
    n_permutations: int = 1000,
    # End updated args
    similarity_metric: str = 'pearson',
    random_seed: Optional[int] = None,
    n_jobs: int = 10,
    shuffle_split: Optional[str] = None,  # New parameter: 'odd', 'even', 'both', or None (both)
    denoising_approach: str = 'compcor',
    reliability_threshold: float = 0.3,
) -> None:
    """
    Main pipeline for pattern similarity analysis with optional bootstrap/permutation testing.
    
    Loads data, applies ROI mask, runs analysis, saves results and generates plots.
    Can use bootstrap confidence intervals OR permutation p-values (mutually exclusive).
    """
    # --- Argument Validation --- #
    if bootstrap and permutation:
        raise ValueError("Cannot use both --bootstrap and --permutation simultaneously.")

    project_root = Path(project_root_str).resolve()
    
    # Determine split method string for directory naming
    split_method_dir = "samesplit" if same_split else "crosssplit"
    output_dir = project_root / 'results' / 'fmri_experiment' / f'pattern_correlation_roi-{roi_name}_denoised-{denoising_approach}_{split_method_dir}' / f'sub-{sub}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # Create a base filename suffix
    split_method_str = "samesplit" if same_split else "crosssplit"
    is_correlation = similarity_metric in ['pearson', 'spearman']
    z_transform_str = "_fisherz" if fisher_z and is_correlation else ""
    zscore_str = "_voxzscored" if zscore_voxels else ""
    core_prefix = f"sub-{sub}_stat-{statstype}_metric-{similarity_metric}_{split_method_str}{z_transform_str}{zscore_str}"

    test_suffix = "" # Suffix for bootstrap or permutation
    if bootstrap:
        test_suffix = f"_bootstrap{n_bootstrap}"
    elif permutation:
        # Handle "both" vs None for shuffle_split
        if shuffle_split == "both":
            shuffle_suffix = "_shuffleboth"
        elif shuffle_split is not None:
            shuffle_suffix = f"_shuffle{shuffle_split}"
        else:
            shuffle_suffix = "_shuffleboth"  # fallback for None
        test_suffix = f"_permtest{n_permutations}{shuffle_suffix}"

    rel_suffix = ""
    if reliability_threshold is not None and reliability_threshold > 0.0:
        rel_suffix = f"_relthr-{reliability_threshold:.2f}"

    base_suffix = f"{core_prefix}{test_suffix}{rel_suffix}_denoised-{denoising_approach}"
    reliability_suffix = f"{core_prefix}_denoised-{denoising_approach}"
    logger.info(f"Using base suffix for output files: {base_suffix}")

    logger.info(f"Loading data for subject {sub}...")
    try:
        dl = DataLoaderExperiment(project_root, denoising_approach=denoising_approach)
        niftis, condinfo, ffa_files, labels_all = dl.load_analysis_data(sub=sub, statstype=statstype, roi_name=roi_name)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    logger.info("Applying ROI mask...")
    try:
        betas, mask_img_resampled, bg_img = apply_roi_mask(niftis, ffa_files)
    except Exception as e:
        logger.error(f"Error applying mask: {e}")
        return

    n_voxels = betas.shape[1] if betas is not None else 0
    logger.info(f"Number of voxels in ROI before reliability filtering: {n_voxels}")

    if zscore_voxels:
        betas = preprocess_zscore_voxels(betas)

    # --- Voxel-wise reliability and optional filtering --- #
    try:
        voxel_reliability = compute_voxel_reliability(betas, condinfo)
    except Exception as e:
        logger.error(f"Error computing voxel reliability: {e}", exc_info=True)
        voxel_reliability = np.full(n_voxels, np.nan)

    # Save reliability values (independent of threshold / test type)
    try:
        np.save(output_dir / f'{reliability_suffix}_voxel_reliability.npy', voxel_reliability)
    except Exception as e:
        logger.error(f"Error saving voxel reliability array: {e}", exc_info=True)

    # Plot distribution of reliability values for inspection
    try:
        valid_rel = voxel_reliability[~np.isnan(voxel_reliability)]
        if valid_rel.size > 0:
            plt.figure(figsize=(8, 6))
            sns.histplot(valid_rel, bins=30, kde=False)
            plt.xlabel("Voxel split-half reliability (r)")
            plt.ylabel("Count")
            plt.title(f"Voxel reliability distribution (sub-{sub}, ROI {roi_name})")
            if reliability_threshold is not None and reliability_threshold > 0.0:
                plt.axvline(
                    reliability_threshold,
                    color="red",
                    linestyle="--",
                    label=f"threshold = {reliability_threshold:.2f}",
                )
                plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f'{reliability_suffix}_voxel_reliability_hist.png')
            plt.close()
        else:
            logger.warning("All voxel reliability values are NaN; skipping reliability histogram.")
    except Exception as e:
        logger.error(f"Error plotting voxel reliability histogram: {e}", exc_info=True)

    # Apply reliability-based voxel mask
    if roi_name == 'wm':
        # For WM control analyses, keep a fixed number of the *least* reliable voxels
        logger.info("WM control ROI: selecting lowest 400 reliability voxels within WM mask.")
        try:
            valid_idx = np.where(~np.isnan(voxel_reliability))[0]
            if valid_idx.size == 0:
                logger.warning(
                    "No valid voxel reliability values found in WM ROI; "
                    "using all voxels in WM mask."
                )
                reliable_mask = np.ones_like(voxel_reliability, dtype=bool)
            else:
                n_select = min(400, int(valid_idx.size))
                # argsort returns indices from lowest to highest reliability
                sorted_idx = valid_idx[np.argsort(voxel_reliability[valid_idx])]
                selected_idx = sorted_idx[:n_select]
                reliable_mask = np.zeros_like(voxel_reliability, dtype=bool)
                reliable_mask[selected_idx] = True

            n_voxels_kept = int(np.nansum(reliable_mask))
            logger.info(
                f"WM control voxel selection: keeping {n_voxels_kept}/{n_voxels} voxels "
                f"with lowest split-half reliability."
            )

            try:
                np.save(output_dir / f'{base_suffix}_voxel_mask.npy', reliable_mask)
                counts_df = pd.DataFrame(
                    [{
                        "subject": sub,
                        "roi": roi_name,
                        "similarity_metric": similarity_metric,
                        "split_method": split_method_str,
                        "denoising_approach": denoising_approach,
                        "reliability_threshold": reliability_threshold,
                        "selection_strategy": "lowest_400_reliability_voxels",
                        "n_voxels_total": int(n_voxels),
                        "n_voxels_kept": int(n_voxels_kept),
                    }]
                )
                counts_df.to_csv(
                    output_dir / f'{base_suffix}_voxel_counts.csv',
                    index=False,
                )
            except Exception as e:
                logger.error(f"Error saving WM voxel selection mask/counts: {e}", exc_info=True)

            if n_voxels_kept > 0:
                betas = betas[:, reliable_mask]
            else:
                logger.warning("WM control voxel selection kept 0 voxels; using all WM voxels instead.")
        except Exception as e:
            logger.error(f"Error during WM control voxel selection: {e}", exc_info=True)
            logger.info("Falling back to using all voxels in WM ROI.")
    else:
        # Default behavior: keep voxels above reliability threshold (if any)
        if reliability_threshold is not None and reliability_threshold > 0.0:
            reliable_mask = voxel_reliability >= reliability_threshold
            n_voxels_kept = int(np.nansum(reliable_mask))
            if n_voxels_kept == 0:
                logger.warning(
                    f"No voxels passed reliability threshold {reliability_threshold:.2f}; "
                    "downstream metrics will be NaN."
                )
            else:
                logger.info(
                    f"Applying voxel reliability threshold {reliability_threshold:.2f}: "
                    f"keeping {n_voxels_kept}/{n_voxels} voxels "
                    f"({(n_voxels_kept / max(n_voxels, 1)) * 100:.1f}%)."
                )
            try:
                np.save(output_dir / f'{base_suffix}_voxel_mask.npy', reliable_mask)
                counts_df = pd.DataFrame(
                    [{
                        "subject": sub,
                        "roi": roi_name,
                        "similarity_metric": similarity_metric,
                        "split_method": split_method_str,
                        "denoising_approach": denoising_approach,
                        "reliability_threshold": reliability_threshold,
                        "n_voxels_total": int(n_voxels),
                        "n_voxels_kept": int(n_voxels_kept),
                    }]
                )
                counts_df.to_csv(
                    output_dir / f'{base_suffix}_voxel_counts.csv',
                    index=False,
                )
            except Exception as e:
                logger.error(f"Error saving voxel reliability mask/counts: {e}", exc_info=True)

            if n_voxels_kept > 0:
                betas = betas[:, reliable_mask]
            else:
                # Keep betas as-is (all NaNs will propagate through metrics)
                pass
        else:
            logger.info("No voxel reliability threshold applied; using all voxels in ROI.")

    metric_name = similarity_metric.replace("_", " ").capitalize()
    logger.info(f"Running pattern analysis using {metric_name} metric...")
    logger.info(f"Using {'Same-Split' if same_split else 'Cross-Split'} between-condition metrics.")
    if fisher_z and is_correlation:
        logger.info("Applying Fisher Z-transform to correlation metrics.")
    elif fisher_z and not is_correlation:
        logger.info("Fisher Z-transform not applicable for non-correlation metric; ignoring.")

    # --- Run Analysis (Bootstrap, Permutation, or Simple) --- #
    results = {}
    significance_results = {} # For bootstrap CIs > 0
    p_values = {} # For permutation tests
    condition_labels = []

    try:
        if bootstrap:
            logger.info(f"Using bootstrapping with {n_bootstrap} iterations and {confidence_level*100}% confidence level.")
            # Run bootstrap
            bootstrap_dist, mean_results, ci_results, significance_results, condition_labels = bootstrap_pattern_correlation(
                betas, condinfo, labels_all,
                n_iterations=n_bootstrap,
                confidence_level=confidence_level,
                same_split=same_split,
                fisher_z=fisher_z,
                similarity_metric=similarity_metric,
                random_seed=random_seed,
                n_jobs=n_jobs
            )
            results = mean_results # Store mean results for saving/logging
            # Bootstrap specific files (dist, CIs, significance) will be saved later

        elif permutation:
            shuffle_info = f"shuffling {shuffle_split} split only" if shuffle_split else "shuffling both splits"
            logger.info(f"Using permutation testing with {n_permutations} iterations ({shuffle_info}).")
            # Run permutation test
            true_results, p_values, condition_labels = permutation_test_pattern_correlation(
                betas, condinfo, labels_all,
                n_permutations=n_permutations,
                same_split=same_split,
                fisher_z=fisher_z,
                similarity_metric=similarity_metric,
                random_seed=random_seed,
                n_jobs=n_jobs,
                shuffle_split=shuffle_split  # Pass the new parameter
            )
            results = true_results # Store true results for saving/logging
            # p-values will be saved later

        else: # Neither bootstrap nor permutation
            logger.info("Running single analysis (no bootstrap or permutation).")
            # Run simple analysis
            (within, mean_between, pairwise,
             simple_sel, contrast_idx, sep, distinct,
             condition_labels) = run_pattern_correlation(
                betas, condinfo, labels_all, same_split=same_split, fisher_z=fisher_z,
                similarity_metric=similarity_metric
            )
            results = { # Store results in the same dictionary format
                 'within_corrs': within,
                 'mean_between_corrs': mean_between,
                 'pairwise_corrs': pairwise,
                 'simple_selectivity': simple_sel,
                 'contrast_index': contrast_idx,
                 'separability': sep,
                 'distinctiveness': distinct
            }

    except Exception as e:
        logger.error(f"Pattern analysis error: {e}", exc_info=True)
        return

    # --- Log and Save Results --- #
    logger.info(f"--- Pattern Analysis Results ({metric_name}) ---")
    # Log main metrics from the results dictionary
    for key, value in results.items():
        if key != 'pairwise_corrs': # Don't log the full matrix
             log_label = key.replace("_corrs", "_metrics").replace("_", " ").title()
             logger.info(f"{log_label}: {np.round(value, 3)}")

    # Log significance if available
    if p_values:
        logger.info("--- Permutation Test P-values ---")
        for key, value in p_values.items():
            log_label = key.replace("_", " ").title()
            logger.info(f"{log_label} p-values: {np.round(value, 4)}")
    if significance_results:
         logger.info("--- Bootstrap Significance (Lower CI > 0) ---")
         for key, value in significance_results.items():
             log_label = key.replace("_", " ").title()
             logger.info(f"{log_label} Significant: {value}")

    try:
        # Save standard metrics (means if bootstrap, true values if permutation/simple)
        np.save(output_dir / f'{base_suffix}_within_metrics.npy', results['within_corrs'])
        np.save(output_dir / f'{base_suffix}_mean_between_metrics.npy', results['mean_between_corrs'])
        np.save(output_dir / f'{base_suffix}_pairwise_metrics.npy', results['pairwise_corrs'])
        np.save(output_dir / f'{base_suffix}_simple_selectivity.npy', results['simple_selectivity'])
        np.save(output_dir / f'{base_suffix}_contrast_index.npy', results['contrast_index'])
        np.save(output_dir / f'{base_suffix}_separability.npy', results['separability'])
        np.save(output_dir / f'{base_suffix}_distinctiveness.npy', results['distinctiveness'])
        np.savetxt(output_dir / f'sub-{sub}_condition_labels.txt', condition_labels, fmt='%s')
        condinfo.to_csv(output_dir / f'sub-{sub}_condinfo_{similarity_metric}.csv', index=False)

        # Save bootstrap specific results
        if bootstrap:
            # Save bootstrap distributions and CIs using consistent keys
            for metric_key in ALL_METRICS:
                if metric_key in bootstrap_dist:
                     metric_file_suffix = metric_key.replace('_corrs','_metrics')
                     # Save bootstrap distribution
                     np.save(output_dir / f'{base_suffix}_{metric_file_suffix}_bootstrap.npy', bootstrap_dist[metric_key])
                     # Save CIs
                     if metric_key in ci_results:
                         np.save(output_dir / f'{base_suffix}_{metric_file_suffix}_ci_lower.npy', ci_results[metric_key]['lower'])
                         np.save(output_dir / f'{base_suffix}_{metric_file_suffix}_ci_upper.npy', ci_results[metric_key]['upper'])
            # Save significance results (Lower CI > 0)
            for metric_key in SELECTIVITY_METRICS:
                if metric_key in significance_results:
                    np.save(output_dir / f'{base_suffix}_{metric_key}_significant_gt_0.npy', significance_results[metric_key])
                else:
                    logger.warning(f"Significance results for metric '{metric_key}' not found, cannot save significance file.")

        # Save permutation specific results
        elif permutation:
            for metric_key in SELECTIVITY_METRICS:
                if metric_key in p_values:
                    np.save(output_dir / f'{base_suffix}_{metric_key}_pvalue.npy', p_values[metric_key])
                else:
                     logger.warning(f"P-value results for metric '{metric_key}' not found, cannot save p-value file.")

        # --- Plotting --- #
        # Plot Pairwise Matrix (using mean/true values)
        pairwise_metrics = results['pairwise_corrs'] # Get from results dict
        plot_title_matrix = f'Subject {sub} Pairwise Pattern {metric_name} ({split_method_str.upper()}, {statstype})'
        matrix_plot_file = output_dir / f'{base_suffix}_pairwise_metric_matrix.png'

        # Determine plot parameters (same logic as before)
        if similarity_metric == 'euclidean_distance':
            cmap_matrix = "viridis"
            vmin_matrix, vmax_matrix = np.nanmin(pairwise_metrics), np.nanmax(pairwise_metrics)
            center_matrix = None
        elif similarity_metric in ['pearson', 'spearman']:
            cmap_matrix = "coolwarm"
            vmin_matrix, vmax_matrix = -0.5, 0.5
            center_matrix = 0.0
        elif similarity_metric in ['euclidean_inverse', 'euclidean_rbf']:
             cmap_matrix = "viridis"
             vmin_matrix, vmax_matrix = 0.0, 1.0
             center_matrix = None
        else: # Fallback
            cmap_matrix = "viridis"
            vmin_matrix, vmax_matrix = np.nanmin(pairwise_metrics), np.nanmax(pairwise_metrics)
            center_matrix = None

        # Ensure vmin < vmax for plotting
        if np.isnan(vmin_matrix) or np.isnan(vmax_matrix) or vmin_matrix >= vmax_matrix:
             logger.warning(f"Could not determine valid range for heatmap ({vmin_matrix=}, {vmax_matrix=}). Setting default range [-0.1, 0.1].")
             vmin_matrix, vmax_matrix = -0.1, 0.1
             if center_matrix is not None: center_matrix = 0.0
        else:
             if center_matrix is None and vmax_matrix > vmin_matrix:
                 range_val = vmax_matrix - vmin_matrix
                 vmin_matrix = vmin_matrix - 0.05 * range_val
                 vmax_matrix = vmax_matrix + 0.05 * range_val

        plot_correlation_matrix(
            pairwise_metrics, plot_title_matrix, matrix_plot_file, condition_labels,
            cmap=cmap_matrix, vmin=vmin_matrix, vmax=vmax_matrix
        )
        logger.info(f"Pairwise metric heatmap saved to {matrix_plot_file}")

        # Plot selectivity indices bars
        plot_title_base = f'Subject {sub} {{metric_label}} ({statstype}, {metric_name})'
        if bootstrap:
            # Plot with CIs if bootstrap was run
            logger.info("Plotting selectivity indices with confidence intervals...")
            for metric_key in SELECTIVITY_METRICS:
                 if metric_key in results and metric_key in ci_results:
                     metric_label = metric_key.replace("_", " ").title()
                     plot_selectivity_bar_with_ci(
                         results[metric_key], # Use mean results stored earlier
                         ci_results[metric_key]['lower'],
                         ci_results[metric_key]['upper'],
                         condition_labels,
                         plot_title_base.format(metric_label=metric_label) + f' with {confidence_level*100:.0f}% CI',
                         f'{metric_label}',
                         output_dir / f'{base_suffix}_{metric_key}_with_ci_bar.png'
                     )
                 else:
                      logger.warning(f"Could not plot {metric_key} with CI for subject {sub} as mean or CI data is missing.")
        else:
            # Plot simple bars if permutation or no stats run
            # (Could add p-value annotations here later by modifying plot_selectivity_bar)
            logger.info("Plotting selectivity indices (standard bars)...")
            for metric_key in SELECTIVITY_METRICS:
                 metric_label = metric_key.replace("_", " ").title()
                 metric_value = results.get(metric_key)
                 if metric_value is not None:
                     plot_selectivity_bar(metric_value, condition_labels,
                                        plot_title_base.format(metric_label=metric_label),
                                        f'{metric_label}',
                                        output_dir / f'{base_suffix}_{metric_key}_bar.png')
                 else:
                      logger.warning(f"Could not plot {metric_key} ({metric_label}) for subject {sub} as data is missing.")

    except Exception as e:
        logger.error(f"Error saving results/plots: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="fMRI Pattern Analysis with Bootstrap/Permutation testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--project_root", type=str, default="../..", help="Path to the project root directory.")
    parser.add_argument("--sub", type=str, required=True, help="Subject ID (e.g., '02').")
    parser.add_argument("--statstype", type=str, default='contrasts', choices=['betas', 'contrasts'], help="Type of statistic to load.")
    parser.add_argument("--same-split", action='store_true', default=False, help="Use same-split metric calculation.")
    parser.add_argument("--raw-values", dest='fisher_z', action='store_false', default=True, help="Use raw metric values instead of Fisher Z (for correlations).")
    parser.add_argument("--zscore-voxels", action='store_true', default=True, help="Z-score voxels before computing metrics (default: True).")
    parser.add_argument("--no-zscore-voxels", dest='zscore_voxels', action='store_false', help="Disable voxel z-scoring.")
    parser.add_argument("--similarity-metric", type=str, default='euclidean_distance', choices=['pearson', 'spearman', 'euclidean_distance', 'euclidean_inverse', 'euclidean_rbf'], help="Metric for pattern comparison.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n-jobs", type=int, default=10, help="Number of parallel jobs.")

    stat_test_group = parser.add_mutually_exclusive_group()
    stat_test_group.add_argument(
        "--bootstrap",
        action='store_true',
        default=False,
        help="Use bootstrapping to estimate confidence intervals."
    )
    stat_test_group.add_argument(
        "--permutation",
        action='store_true',
        default=True,
        help="Use permutation testing to estimate p-values."
    )
    parser.add_argument(
        "--no-permutation",
        action='store_true',
        help="Run single analysis only (no permutation or bootstrap)."
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1_000,
        help="Number of bootstrap iterations (default: 1000)."
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals (0-1, default: 0.95)."
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10_000,
        help="Number of permutation iterations (default: 10000, matches paper)."
    )
    parser.add_argument(
        "--shuffle-split",
        type=str,
        choices=['odd', 'even', "both"],
        default='odd',
        help="Which imageset to shuffle for permutation testing ('odd', 'even', or 'both')"
    )
    parser.add_argument(
        "--roi-name",
        type=str,
        choices=['FFA', 'wm'],
        default='FFA',
        help="Name of ROI to use for pattern correlation."
    )
    parser.add_argument(
        "--denoising-approach",
        type=str,
        choices=['motion', 'compcor', 'aroma'],
        default='compcor',
        help="Denoising approach to use for loading data."
    )
    parser.add_argument(
        "--reliability-threshold",
        type=float,
        default=0.2,
        help="Minimum voxel split-half reliability (r) required to keep a voxel. "
             "Default 0.2 matches the paper (Methods: Pattern selectivity analysis). "
             "If 0 or negative, all voxels in the ROI are used."
    )

    args = parser.parse_args()
    if args.no_permutation:
        args.permutation = False

    main(
        args.project_root,
        args.sub,
        args.statstype,
        args.roi_name,
        args.same_split,
        args.fisher_z,
        args.zscore_voxels,
        args.bootstrap,
        args.n_bootstrap,
        args.confidence_level,
        args.permutation,
        args.n_permutations,
        args.similarity_metric,
        args.random_seed,
        args.n_jobs,
        args.shuffle_split,
        args.denoising_approach,
        args.reliability_threshold,
    ) 
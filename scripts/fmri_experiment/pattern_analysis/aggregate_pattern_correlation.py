#!/usr/bin/env python
"""
Script for aggregating and visualizing pattern correlation results across subjects.

This script:
1. Loads pattern correlation results for multiple subjects
2. Aggregates the results (mean, SEM) across subjects
3. Generates summary plots and statistics
4. Saves aggregated results for further analysis
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_subject_results(
    project_root: Path,
    sub: str,
    statstype: str,
    same_split: bool,
    fisher_z: bool,
    zscore_voxels: bool,
    bootstrap: bool = False,
    n_bootstrap: int = 10000,
    reliability_threshold: float = 0.3,
) -> Tuple[Dict[str, np.ndarray], List[str], Optional[Dict[str, Dict[str, np.ndarray]]]]:
    """
    Load pattern correlation results for a single subject.
    
    Args:
        project_root: Path to project root directory
        sub: Subject ID
        statstype: Type of statistic ('betas' or 'contrasts')
        same_split: Whether same-split correlations were used
        fisher_z: Whether Fisher Z-transform was used
        zscore_voxels: Whether voxels were z-scored
        bootstrap: Whether bootstrapping was used
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Tuple containing:
        - Dictionary of results arrays
        - List of condition labels
        - Optional dictionary of bootstrap confidence intervals
    """
    # Construct base filename suffix
    split_method_str = "samesplit" if same_split else "crosssplit"
    z_transform_str = "_fisherz" if fisher_z else ""
    zscore_str = "_voxzscored" if zscore_voxels else ""
    rel_suffix = ""
    if reliability_threshold is not None and reliability_threshold > 0.0:
        rel_suffix = f"_relthr-{reliability_threshold:.2f}"
    bootstrap_str = f"_bootstrap{n_bootstrap}" if bootstrap else ""
    base_suffix = f"sub-{sub}_stat-{statstype}_corr-{split_method_str}{z_transform_str}{zscore_str}{bootstrap_str}{rel_suffix}"
    
    # Load results
    results_dir = project_root / 'results' / 'fmri_experiment' / 'pattern_correlation' / f'sub-{sub}'
    
    # Load main results
    results = {
        'within_corrs': np.load(results_dir / f'{base_suffix}_within_corrs.npy'),
        'mean_between_corrs': np.load(results_dir / f'{base_suffix}_mean_between_corrs.npy'),
        'max_between_corrs': np.load(results_dir / f'{base_suffix}_max_between_corrs.npy'),
        'pairwise_corrs': np.load(results_dir / f'{base_suffix}_pairwise_corrs.npy'),
        'simple_selectivity': np.load(results_dir / f'{base_suffix}_simple_selectivity.npy'),
        'strict_selectivity': np.load(results_dir / f'{base_suffix}_strict_selectivity.npy'),
        'normalized_selectivity': np.load(results_dir / f'{base_suffix}_normalized_selectivity.npy')
    }
    
    # Load condition labels
    condition_labels = np.loadtxt(results_dir / f'sub-{sub}_condition_labels.txt', dtype=str).tolist()
    
    # Load bootstrap confidence intervals if available
    ci_results = None
    if bootstrap:
        ci_results = {}
        for metric in ['within_corrs', 'mean_between_corrs', 'max_between_corrs', 
                      'simple_selectivity', 'strict_selectivity', 'normalized_selectivity']:
            ci_results[metric] = {
                'lower': np.load(results_dir / f'{base_suffix}_{metric}_ci_lower.npy'),
                'upper': np.load(results_dir / f'{base_suffix}_{metric}_ci_upper.npy')
            }
        ci_results['pairwise_corrs'] = {
            'lower': np.load(results_dir / f'{base_suffix}_pairwise_corrs_ci_lower.npy'),
            'upper': np.load(results_dir / f'{base_suffix}_pairwise_corrs_ci_upper.npy')
        }
    
    return results, condition_labels, ci_results

def aggregate_results(
    project_root: Path,
    subjects: List[str],
    statstype: str,
    same_split: bool,
    fisher_z: bool,
    zscore_voxels: bool,
    bootstrap: bool = False,
    n_bootstrap: int = 10000,
    reliability_threshold: float = 0.3,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], Optional[Dict[str, Dict[str, np.ndarray]]]]:
    """
    Aggregate pattern correlation results across subjects.
    
    Args:
        project_root: Path to project root directory
        subjects: List of subject IDs
        statstype: Type of statistic ('betas' or 'contrasts')
        same_split: Whether same-split correlations were used
        fisher_z: Whether Fisher Z-transform was used
        zscore_voxels: Whether voxels were z-scored
        bootstrap: Whether bootstrapping was used
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Tuple containing:
        - Dictionary of mean results across subjects
        - Dictionary of SEM results across subjects
        - List of condition labels
        - Optional dictionary of bootstrap confidence intervals
    """
    # Initialize storage
    all_results = {}
    all_ci_results = {} if bootstrap else None
    condition_labels = None
    
    # Load results for each subject
    for sub in tqdm(subjects, desc="Loading subject results"):
        results, labels, ci_results = load_subject_results(
            project_root, sub, statstype, same_split, fisher_z, zscore_voxels,
            bootstrap, n_bootstrap, reliability_threshold
        )
        
        # Store results
        for metric, values in results.items():
            if metric not in all_results:
                all_results[metric] = []
            all_results[metric].append(values)
        
        # Store CI results if available
        if bootstrap and ci_results is not None:
            for metric, ci in ci_results.items():
                if metric not in all_ci_results:
                    all_ci_results[metric] = {'lower': [], 'upper': []}
                all_ci_results[metric]['lower'].append(ci['lower'])
                all_ci_results[metric]['upper'].append(ci['upper'])
        
        # Store condition labels (should be same for all subjects)
        if condition_labels is None:
            condition_labels = labels
        elif labels != condition_labels:
            logger.warning(f"Condition labels mismatch for subject {sub}")
            # Use the intersection of labels
            condition_labels = sorted(set(condition_labels) & set(labels))
    
    # Calculate mean and SEM across subjects
    mean_results = {}
    sem_results = {}
    for metric, values in all_results.items():
        # Ensure all arrays have the same shape
        values = np.array(values)
        if values.ndim > 1:
            # For 2D arrays (like pairwise_corrs), ensure all have same shape
            max_shape = max(arr.shape[0] for arr in values)
            padded_values = []
            for arr in values:
                if arr.shape[0] < max_shape:
                    # Pad with NaN if needed
                    pad_shape = (max_shape - arr.shape[0],) + arr.shape[1:]
                    padded = np.pad(arr, ((0, pad_shape[0]), (0, 0)), 
                                  mode='constant', constant_values=np.nan)
                    padded_values.append(padded)
                else:
                    padded_values.append(arr)
            values = np.array(padded_values)
        
        mean_results[metric] = np.nanmean(values, axis=0)
        sem_results[metric] = stats.sem(values, axis=0, nan_policy='omit')
    
    # Calculate mean CI results if available
    mean_ci_results = None
    if bootstrap and all_ci_results:
        mean_ci_results = {}
        for metric, ci in all_ci_results.items():
            mean_ci_results[metric] = {
                'lower': np.nanmean(ci['lower'], axis=0),
                'upper': np.nanmean(ci['upper'], axis=0)
            }
    
    return mean_results, sem_results, condition_labels, mean_ci_results

def plot_aggregated_results(
    mean_results: Dict[str, np.ndarray],
    sem_results: Dict[str, np.ndarray],
    condition_labels: List[str],
    output_dir: Path,
    statstype: str,
    same_split: bool,
    fisher_z: bool,
    zscore_voxels: bool,
    subjects: List[str],
    project_root: Path,
    bootstrap: bool = False,
    ci_results: Optional[Dict[str, Dict[str, np.ndarray]]] = None
) -> None:
    """
    Generate plots of aggregated pattern correlation results.
    
    Args:
        mean_results: Dictionary of mean results across subjects
        sem_results: Dictionary of SEM results across subjects
        condition_labels: List of condition labels
        output_dir: Directory to save plots
        statstype: Type of statistic used
        same_split: Whether same-split correlations were used
        fisher_z: Whether Fisher Z-transform was used
        zscore_voxels: Whether voxels were z-scored
        subjects: List of subject IDs
        project_root: Path to project root directory
        bootstrap: Whether bootstrapping was used
        ci_results: Optional dictionary of bootstrap confidence intervals
    """
    # Create base filename suffix
    split_method_str = "samesplit" if same_split else "crosssplit"
    z_transform_str = "_fisherz" if fisher_z else ""
    zscore_str = "_voxzscored" if zscore_voxels else ""
    base_suffix = f"stat-{statstype}_corr-{split_method_str}{z_transform_str}{zscore_str}"
    
    # Set the style without grid
    sns.set_style("ticks")
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        mean_results['pairwise_corrs'],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=condition_labels,
        yticklabels=condition_labels,
        linewidths=.5,
        cbar=True,
        square=True,
        vmin=-1.0,
        vmax=1.0,
        center=0.0
    )
    plt.title(f"Mean Pairwise Pattern Correlations ({split_method_str.upper()}, {statstype})")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / f'mean_{base_suffix}_pairwise_correlation_matrix.png')
    plt.close()
    
    # Plot selectivity indices
    for metric, title in [
        ('simple_selectivity', 'Simple Selectivity'),
        ('strict_selectivity', 'Strict Selectivity'),
        ('normalized_selectivity', 'Normalized Selectivity')
    ]:
        # Get individual subject values
        subject_values = []
        for sub in subjects:
            results, _, _ = load_subject_results(
                project_root, sub, statstype, same_split, fisher_z, zscore_voxels
            )
            subject_values.append(results[metric])
        
        # Convert to DataFrame for stripplot
        df = pd.DataFrame({
            'condition': np.repeat(condition_labels, len(subjects)),
            'value': np.concatenate(subject_values),
            'subject': np.tile([f'Sub-{sub}' for sub in subjects], len(condition_labels))
        })
        
        # Sort by mean values
        mean_values = df.groupby('condition')['value'].mean()
        sorted_conditions = mean_values.sort_values(ascending=False).index
        df['condition'] = pd.Categorical(df['condition'], categories=sorted_conditions, ordered=True)
        
        # Create figure
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        # Plot the means as bars
        bar_width = 0.5
        mean_values = df.groupby('condition')['value'].mean()
        bars = plt.bar(
            mean_values.index, mean_values.values,
            color='none', edgecolor='black', linewidth=2, width=bar_width, zorder=1, alpha=0.7
        )
        
        # Add a horizontal line at 0
        plt.axhline(0, color='grey', linestyle='--', linewidth=2, zorder=2)
        
        # Show each observation with a stripplot
        sns.stripplot(
            data=df,
            y='value', x='condition',
            hue='subject', dodge=True, alpha=0.8, zorder=3,
            palette='tab10', size=8, jitter=0.2
        )
        
        # Set y-axis limits for normalized selectivity
        if metric == 'normalized_selectivity':
            plt.ylim(-30, 30)
        
        plt.xlabel("Condition")
        plt.ylabel(title)
        plt.title(f"Mean {title} Across Subjects ({statstype})")
        plt.xticks(rotation=60, ha='right')
        
        # Get only the subject handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels, loc='upper right', 
                  ncol=2, fontsize=10, frameon=True, title='Subjects')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'mean_{base_suffix}_{metric}_bar.png')
        plt.close()
    
    # Plot within vs between correlations
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Get individual subject values
    within_values = []
    between_values = []
    for sub in subjects:
        results, _, _ = load_subject_results(
            project_root, sub, statstype, same_split, fisher_z, zscore_voxels
        )
        within_values.append(results['within_corrs'])
        between_values.append(results['mean_between_corrs'])
    
    # Convert to DataFrame for stripplot
    df = pd.DataFrame({
        'condition': np.repeat(condition_labels, len(subjects) * 2),
        'value': np.concatenate([np.concatenate(within_values), np.concatenate(between_values)]),
        'type': np.repeat(['Within', 'Between'], len(condition_labels) * len(subjects)),
        'subject': np.tile([f'Sub-{sub}' for sub in subjects], len(condition_labels) * 2)
    })
    
    # Calculate means for each condition and type
    mean_within = df[df['type'] == 'Within'].groupby('condition')['value'].mean()
    mean_between = df[df['type'] == 'Between'].groupby('condition')['value'].mean()
    
    # Ensure we have the same conditions in both means
    all_conditions = sorted(set(mean_within.index) | set(mean_between.index))
    mean_within = mean_within.reindex(all_conditions)
    mean_between = mean_between.reindex(all_conditions)
    
    # Plot the means as bars
    width = 0.35
    x = np.arange(len(all_conditions))
    
    within_bars = plt.bar(x - width/2, mean_within.values, width,
                         color='none', edgecolor='black', linewidth=2, zorder=1, alpha=0.7)
    between_bars = plt.bar(x + width/2, mean_between.values, width,
                          color='none', edgecolor='black', linewidth=2, zorder=1, alpha=0.7)
    
    # Show each observation with a stripplot
    sns.stripplot(
        data=df,
        y='value', x='condition', hue='subject',
        dodge=True, alpha=0.8, zorder=3,
        palette='tab10', size=8, jitter=0.2
    )
    
    plt.xlabel("Condition")
    plt.ylabel("Correlation")
    plt.title(f"Within vs Between-Condition Correlations ({statstype})")
    plt.xticks(x, all_conditions, rotation=45, ha='right')
    
    # Get only the subject handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='upper right', 
              ncol=2, fontsize=10, frameon=True, title='Subjects')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'mean_{base_suffix}_within_vs_between.png')
    plt.close()

def main(
    project_root_str: str,
    subjects: List[str],
    statstype: str,
    same_split: bool,
    fisher_z: bool,
    zscore_voxels: bool,
    bootstrap: bool = False,
    n_bootstrap: int = 10000,
    reliability_threshold: float = 0.3,
) -> None:
    """
    Main function to aggregate and visualize pattern correlation results.
    
    Args:
        project_root_str: Path to project root directory
        subjects: List of subject IDs
        statstype: Type of statistic ('betas' or 'contrasts')
        same_split: Whether same-split correlations were used
        fisher_z: Whether Fisher Z-transform was used
        zscore_voxels: Whether voxels were z-scored
        bootstrap: Whether bootstrapping was used
        n_bootstrap: Number of bootstrap iterations
    """
    project_root = Path(project_root_str).resolve()
    output_dir = project_root / 'results' / 'fmri_experiment' / 'pattern_correlation' / 'aggregated'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Aggregating results for subjects: {subjects}")
    logger.info(f"Using {'Same-Split' if same_split else 'Cross-Split'} correlations")
    if fisher_z:
        logger.info("Using Fisher Z-transformed correlations")
    if bootstrap:
        logger.info(f"Using bootstrapped results with {n_bootstrap} iterations")
    
    # Aggregate results
    mean_results, sem_results, condition_labels, ci_results = aggregate_results(
        project_root, subjects, statstype, same_split, fisher_z, zscore_voxels,
        bootstrap, n_bootstrap, reliability_threshold
    )
    
    # Save aggregated results
    np.save(output_dir / 'mean_results.npy', mean_results)
    np.save(output_dir / 'sem_results.npy', sem_results)
    np.savetxt(output_dir / 'condition_labels.txt', condition_labels, fmt='%s')
    if bootstrap and ci_results is not None:
        np.save(output_dir / 'ci_results.npy', ci_results)
    
    # Generate plots
    plot_aggregated_results(
        mean_results, sem_results, condition_labels, output_dir,
        statstype, same_split, fisher_z, zscore_voxels,
        subjects, project_root, bootstrap, ci_results
    )
    
    logger.info("Aggregation complete. Results saved to: " + str(output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Pattern Correlation Results")
    parser.add_argument("--project_root", type=str, default="../..", 
                       help="Path to project root directory")
    parser.add_argument("--subjects", type=str, nargs='+', required=True,
                       help="List of subject IDs (e.g., '02 03 04')")
    parser.add_argument("--statstype", type=str, default='contrasts',
                       choices=['betas', 'contrasts'],
                       help="Type of statistic ('betas' or 'contrasts')")
    parser.add_argument("--same-split", action='store_true', default=False,
                       help="Use same-split between-condition correlations")
    parser.add_argument("--raw-correlations", dest='fisher_z', action='store_false',
                       default=True, help="Use raw correlations instead of Fisher Z-transformed")
    parser.add_argument("--zscore-voxels", action='store_true', default=True,
                       help="Use z-scored voxel patterns (default: True).")
    parser.add_argument("--no-zscore-voxels", dest='zscore_voxels', action='store_false',
                       help="Disable z-scored voxel patterns.")
    parser.add_argument("--bootstrap", action='store_true', default=False,
                       help="Use bootstrapped results")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                       help="Number of bootstrap iterations")
    parser.add_argument(
        "--reliability-threshold",
        type=float,
        default=0.3,
        help="Voxel reliability threshold (r) used in the subject-level analysis. "
             "Must match the value passed to run_pattern_correlation.py."
    )
    
    args = parser.parse_args()
    
    main(
        args.project_root,
        args.subjects,
        args.statstype,
        args.same_split,
        args.fisher_z,
        args.zscore_voxels,
        args.bootstrap,
        args.n_bootstrap,
        args.reliability_threshold,
    ) 
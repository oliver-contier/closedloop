import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from closedloop.maths import vectorized_pearsonr
from typing import Dict
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
)


def get_consensus_khosla(
    embeddings,
    rho=0.3,
    tau=1.0,
    kmeans_k=0,
    sort_by_sum=True,
    return_score=False,
    score="calinski_harabasz",
    plot_distance_histograms=False,
    kmeans_kwargs=dict(n_init=10, random_state=1),
):
    """Get consensus components from embeddings.
    Args:
        embeddings: list of embeddings, each element is (np.ndarray) (nimages, ndimensions). Embeddings can have different numbers of dimensions.
        rho (float): proportion of local neighbors considered
        tau (float): threshold for average distance within cluster
        kmeans_k (int): Dimensionality of the resulting consensus embedding. if 0, use average dimensionality of input embeddings.
        sort_by_sum (bool): whether to sort dimensions of consensus embedding by sum of weights
        return_score (bool): whether to return silhouette score, in which case function returns tuple (consensus, silscore). Only relevant if return_score is True.
        score (str): which score to evaluate solution. 'wcss' (within-cluster sum of squares) or 'silhouette'.
        plot_distance_histograms (bool): whether to plot distance distributions
    Returns:
        consensus (np.ndarray): (nimages, ndimensions)
    """
    assert score in ("wcss", "silhouette", "calinski_harabasz")
    # stack embeddings across runs
    embs_concat = np.hstack(embeddings)
    nims, ndims_total = embs_concat.shape
    nruns = len(embeddings)
    avg_ndims = ndims_total / nruns
    # rescale dimensions to unit norm
    embs_scaled = embs_concat / np.linalg.norm(embs_concat, axis=0)
    # compute distances between all dimensions
    dists = euclidean_distances(embs_scaled.T)
    # find local neighbors
    # how many local neighbors, given rho?
    l = int(rho * avg_ndims)
    dists[np.diag_indices_from(dists)] = np.inf  # to avoid self-matches
    dists_nn = np.partition(dists, l, axis=1)[:, :l]
    dists[np.diag_indices_from(dists)] = 0  # back to zero diagonal
    if plot_distance_histograms:
        for i, dim_dists in enumerate(dists):
            plt.hist(dim_dists)
            plt.title(f"distance distribution for dimension {i}")
            plt.show()
    # calculate average distance within nighborhoods
    dists_avg = np.mean(dists_nn, axis=1)
    # filter dimensions by average distance, given tau
    dims_select_inds = np.where(dists_avg < tau)[0]
    # print(f"{dims_select_inds.shape[0]} dimensions across runs (out of {nruns * ncomps}) passed consistency threshold (tau = {tau})")
    dims_select = embs_scaled[:, dims_select_inds]
    # also filter distances
    dists_select = dists[dims_select_inds, :][:, dims_select_inds]
    # run kmeans on filtered dimension
    nclust = kmeans_k if kmeans_k else avg_ndims
    kmeans = KMeans(n_clusters=nclust, **kmeans_kwargs)
    kmeans.fit(dists_select)
    # Take median of each cluster as consensus dimension
    consensus = np.zeros((nclust, nims))
    singular_clusters = []  # count clusters with only one component
    for clust_i in range(nclust):
        cluster_is = kmeans.labels_ == clust_i
        if cluster_is.sum() == 1:
            dim_consensus = dims_select[:, cluster_is].squeeze()
            singular_clusters.append(clust_i)
        else:
            dim_consensus = np.median(dims_select[:, cluster_is], axis=1)
        consensus[clust_i] = dim_consensus
    # print(f"found {len(singular_clusters)} clusters with only one component")
    # sort dimensions by sum of weights
    consensus = consensus.T
    dim_sortinds = np.argsort(consensus.sum(axis=0))
    consensus = consensus[:, dim_sortinds]
    if sort_by_sum:
        sortinds = np.flip(np.argsort(consensus.sum(axis=0)))
        consensus = consensus[:, sortinds]
    if return_score:
        if score == "silhouette":
            silscore = silhouette_score(dists_select, kmeans.labels_)
        elif score == "wcss":
            silscore = kmeans.inertia_
        elif score == "calinski_harabasz":
            silscore = calinski_harabasz_score(dists_select, kmeans.labels_)
        return consensus, silscore
    else:
        return consensus


def gridsearch_initialdims_tau_khosla(
    embeddings,
    ndims=np.array([20, 50, 70, 100, 120]),
    taus=np.linspace(0.5, 1.5, 11),
):
    """Evaluate silhouette score for different values of initial dimensionality and tau.

    Args:
        ndims (_type_, optional): _description_. Defaults to np.array([20,50,70,100,120]).
        taus (_type_, optional): _description_. Defaults to np.linspace(0.5, 1.5, 11).

    Returns:
        _type_: _description_
    """
    assert np.max(ndims) <= embeddings[0].shape[1]
    scores = np.zeros((len(ndims), len(taus)))
    for di, ndim in tqdm(enumerate(ndims), desc="optimize ndim", total=len(ndims)):
        embs_ = [e[:, :ndim] for e in embeddings]
        for ti, tau in tqdm(
            enumerate(taus), desc="optimize tau", leave=False, total=len(taus)
        ):
            _, silscore = get_consensus_khosla(
                embs_, rho=0.3, tau=tau, return_score=True
            )
            scores[di, ti] = silscore
    return scores


def gridsearch_k_tau_khosla(  # doesn't work as well as optimizing initial_dims and tau
    embeddings,
    ks=np.arange(10, 101, 10),
    taus=np.linspace(0.5, 1.5, 11),
):
    """
    Evaluate silhouette score for different values of k and tau of the consensus embedding.

    Args:
        embeddings (list): List of embeddings, each element is (np.ndarray) (nimages, ndimensions)
        ks (iterable): k-values. Defaults to np.arange(10, 101, 10)
        taus (iterable): tau-values. Defaults to np.linspace(0.5, 1.5, 11).

    Returns:
        np.array (len(ks), len(taus)): array of silhouette scores
    """
    # assert ks[-1] <= embeddings[0].shape[1]
    scores = np.zeros((len(ks), len(taus)))
    for ti, tau in tqdm(
        enumerate(taus), desc="optimize tau", leave=True, total=len(taus)
    ):
        for ki, k in tqdm(enumerate(ks), desc="optimize k", leave=False, total=len(ks)):
            _, silscore = get_consensus_khosla(
                embeddings, rho=0.3, tau=tau, kmeans_k=k, return_score=True
            )
            scores[ki, ti] = silscore
    return scores


def split_half_reliability(embeddings: np.ndarray, identifiers: str) -> Dict[str, list]:
    """
    Compute the split-half reliability of each dimension for each model.

    The method is as follows:
    1. Split the data objects in half using an odd and even mask.
    2. For a given model run, iterate across all dimensions $i$:
        - Identify the dimension in all other models $k$ that has the highest correlation with dimension $i$,
          calculated using the odd-masked data.
        - For model $k$, correlate this identified dimension with dimension $i$ using the even-masked data.
    This process results in a sampling distribution of Pearson r coefficients across all other model seeds.
    The sampling distribution of Pearson r is then transformed using Fisher-z so that it becomes z-scored (i.e.,
    normally distributed). The mean of this sampling distribution is taken as the average z-transformed
    reliability score. Finally, this score is inverted to get the average Pearson r reliability score.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings in the shape of (n_embeddings, n_objects, n_dimensions).

    identifiers : str
        Identifiers of the embeddings, which could be model names or seeds.

    Returns
    -------
    dict
        A dictionary with identifiers as keys and Pearson r values across all dimensions.


    TODO - Take the pruned dimensions to calc fisher z!
    """
    assert (
        embeddings.ndim == 3
    ), "Embeddings must be 3-dimensional (n_embeddings, n_objects, n_dimensions)"
    n_embeddings, n_objects, n_dimensions = embeddings.shape

    np.random.seed(42)
    odd_mask = np.random.choice([True, False], size=n_objects)
    even_mask = np.invert(odd_mask)
    reliability_per_dim = defaultdict(list)

    for i in tqdm(range(n_embeddings)):
        ident = identifiers[i]
        reproducibility_across_embeddings = np.zeros((n_embeddings, n_dimensions))

        for j in range(n_embeddings):
            if i == j:
                continue

            emb_i = embeddings[i]
            emb_j = embeddings[j]

            # For each dim i, correlate with all other dims j
            corr_ij = vectorized_pearsonr(emb_i[odd_mask], emb_j[odd_mask])

            # For each dim i, find the dim j with the highest correlation
            highest_corrs = np.argmax(corr_ij, axis=1)

            # For each dim i, correlate with the dim j with the highest correlation across even objects
            even_corrs = np.zeros(n_dimensions)
            for k in range(n_dimensions):
                base_even = emb_i[even_mask][:, k]
                dim_match = highest_corrs[k]  # dim j with highest corr to dim i
                comp_even = emb_j[even_mask][:, dim_match]
                even_corrs[k] = vectorized_pearsonr(base_even, comp_even)

            reproducibility_across_embeddings[j] = even_corrs

        z_transformed = np.arctanh(reproducibility_across_embeddings)
        average = np.mean(z_transformed, axis=0)
        back_transformed = np.tanh(average)
        reliability_per_dim[ident] = back_transformed

    return reliability_per_dim


def make_npruned_dict(results):
    n_pruned_dict = {}
    for res in results:
        ident = f'subj-{res["subj"]}_lmdb-{res["beta"]:.4f}_seed-{res["seed"]}'
        n_pruned = np.load(res["embfile"])["pruned_weights"].shape[1]
        n_pruned_dict[ident] = n_pruned
        res["n_pruned"] = n_pruned
        res["ident"] = ident
    return n_pruned_dict


def find_mean_reliability(
    reliability_per_dim: Dict[str, list], n_pruned: Dict[str, int]
) -> Dict[str, float]:
    mean_reliabilities = {}
    for key, values in reliability_per_dim.items():
        ndim = n_pruned[key]
        values = values[:ndim]
        mean_reliabilities[key] = np.mean(values)
    return mean_reliabilities


def cluster_evaluate_ks(
    data, ks, clustering_model, #gap_nboot=3, #kmeans_kws=dict(n_init=10, random_state=0)
):
    """
    Calculate fit of sklearn clustering model for a range of k-values.
    """
    # Models for which this function works
    allowed_models = (KMeans, SpectralClustering, AgglomerativeClustering)
    was_allowed = [isinstance(clustering_model, allowed) for allowed in allowed_models]
    assert np.any(was_allowed)
    # if isinstance(clustering_model, AgglomerativeClustering):
    #     clustering_model.set_params(linkage='ward')
    # initialize results data frame
    scores_df = pd.DataFrame({"k": [], "gap": []})
    for k in tqdm(ks, leave=True, desc="k"):
        # Fit cluster to original data and calculate wcss
        # km = KMeans(k, **kmeans_kws)
        clustering_model.set_params(n_clusters=k)
        clustering_model.fit(data)
        # wcss = copy(clustering_model.inertia_)
        # calculate gap statistics via bootstrapping
        # if gap_nboot > 0:
        #     wcss_boots = np.zeros(gap_nboot)
        #     # km_gap = KMeans(k, **kmeans_kws)
        #     for b in tqdm(range(gap_nboot), desc="references", leave=False):
        #         randomReference = np.random.random_sample(size=data.shape)
        #         clustering_model.fit(randomReference)
        #         wcss_boots[b] = clustering_model.inertia_
        #     # gap statistic
        #     gap = np.log(np.mean(wcss_boots)) - np.log(wcss)
        # else:
        #     gap = 0
        # Assign this loop's gap statistic to gaps
        scores_df = scores_df.append(
            {
                "k": k,
                "silhouette": silhouette_score(data, clustering_model.labels_),
                "calinski_harabasz": calinski_harabasz_score(data, clustering_model.labels_),
                "davies_bouldin": davies_bouldin_score(data, clustering_model.labels_),
                # "wcss": wcss,
                # "gap": gap,
            },
            ignore_index=True,
        )
    return scores_df

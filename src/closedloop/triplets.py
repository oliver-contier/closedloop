"""
author: Shu Fujimori
"""
import os

# os.environ["OMP_NUM_THREADS"] = "20"
# os.environ["MKL_NUM_THREADS"] = "20"
# os.environ["NUMEXPR_NUM_THREADS"] = "20"
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform


# def filter_triplets(rnd_samples: np.ndarray, n_samples: float) -> np.ndarray:
#     """filter for unique triplets (i, j, k have to be different indices)"""

#     def is_set_(triplet: np.ndarray) -> bool:
#         return len(np.unique(triplet)) == len(triplet)

#     rnd_samples = np.asarray(list(filter(is_set_, rnd_samples)))
#     rnd_samples = np.unique(rnd_samples, axis=0)
#     # shuffle for fix bug
#     rnd_samples_index = np.random.sample(
#         list(range(len(rnd_samples))), len(rnd_samples)
#     )
#     rnd_samples = rnd_samples[rnd_samples_index, :][: int(n_samples)]
#     if len(rnd_samples) != n_samples:
#         raise Exception(
#             f"Increase sampling_constant. Current data size is {len(rnd_samples)}, excepted is {n_samples}"
#         )
#     return rnd_samples

def filter_triplets(rnd_samples: np.ndarray, n_samples: float) -> np.ndarray:
    """filter for unique triplets (i, j, k have to be different indices)"""

    def is_set_(triplet: np.ndarray) -> bool:
        return len(np.unique(triplet)) == len(triplet)

    rnd_samples = np.asarray(list(filter(is_set_, rnd_samples)))
    rnd_samples = np.unique(rnd_samples, axis=0)[: int(n_samples)]
    return rnd_samples


def min_max_normalize(x):
    min_x = x.min()
    max_x = x.max()
    x_norm = (x - min_x) / (max_x - min_x)
    return x_norm


def calc_S(X:np.ndarray, dist_method:str):
    assert dist_method in ("dot", "cosine", "correlation", "euclidean")
    if dist_method == "dot":
        S = X @ X.T
    elif dist_method == "cosine":
        S = squareform(pdist(X, metric=dist_method))
    elif dist_method == "correlation":
        S = squareform(pdist(X, metric=dist_method))
    elif dist_method == "euclidean":
        D = squareform(pdist(X, metric=dist_method))
        print('converting euclidean distances to similarities')
        D = min_max_normalize(S)
        S = 1 - D


def calc_triplets(
    out_path: str,
    n_samples: float,
    data: np.ndarray,
    dist_method: str,
) -> None:
    """create triplets of object embedding similarities, and for each triplet find the odd-one-out"""
    sampling_constant = n_samples / 10
    # load input data (e.g., word embeddings, image features)
    X = data
    # create similarity matrix
    print('computing S')
    if dist_method == "dot":
        S = X @ X.T
    elif dist_method == "cosine":
        S = squareform(pdist(X, metric=dist_method))
    elif dist_method == "correlation":
        S = squareform(pdist(X, metric=dist_method))
    elif dist_method == "euclidean":
        D = squareform(pdist(X, metric=dist_method))
        print('minmaxing, converting to similarities')
        D = min_max_normalize(D)
        S = 1 - D

    N = S.shape[0]
    # draw random triplet samples
    print('drawing random samples')
    rnd_samples = np.random.randint(N, size=(n_samples + int(sampling_constant), 3))
    print("rnd_samples", rnd_samples)
    # filter for unique triplets and remove all duplicates
    print('filtering for unique triplets')
    rnd_samples = filter_triplets(rnd_samples, n_samples)
    print("rnd_samples.shape", rnd_samples.shape)  # (6952695, 3)
    triplets = np.zeros((int(n_samples), 3), dtype=int)
    print('iterating over triplets, find ooo')
    for idx, [i, j, k] in enumerate(tqdm(rnd_samples)):
        odd_one_outs = np.asarray([k, j, i])
        sims = np.array([S[i, j], S[i, k], S[j, k]])
        # simply use the argmax to (deterministically) find the odd-one-out choice
        choices = odd_one_outs[np.argsort(sims)]
        triplets[idx] = choices
    print('creating output directory')
    os.makedirs(out_path, exist_ok=True)
    rnd_indices = np.random.permutation(len(triplets))
    train_triplets = triplets[rnd_indices[: int(len(rnd_indices) * 0.9)]]
    test_triplets = triplets[rnd_indices[int(len(rnd_indices) * 0.9) :]]
    print(
        "Train data",
        train_triplets.shape,
        "Test data",
        test_triplets.shape,
    )
    # save as npy
    with open(os.path.join(out_path, "train_90.npy"), "wb") as train_file:
        np.save(train_file, train_triplets)
    print(os.path.join(out_path, "train_90.npy"))
    with open(os.path.join(out_path, "test_10.npy"), "wb") as test_file:
        np.save(test_file, test_triplets)
    print(os.path.join(out_path, "test_10.npy"))
    # save as txt
    np.savetxt(os.path.join(out_path, "train_90.txt"), train_triplets, fmt="%d")
    print(os.path.join(out_path, "train_90.txt"))
    np.savetxt(os.path.join(out_path, "test_10.txt"), test_triplets, fmt="%d")
    print(os.path.join(out_path, "test_10.txt"))
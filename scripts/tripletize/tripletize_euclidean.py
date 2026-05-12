from closedloop.triplets import calc_triplets
from os.path import join as pjoin
from os import pardir
import numpy as np

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="all")
    parser.add_argument("--simthresh", type=str, default="0.0")
    parser.add_argument("--trans", type=str, default="zvox")
    parser.add_argument("--ntrips", type=int, default=4_000_000)
    args = parser.parse_args()
    
    nmiostr = f"{int(args.ntrips/1_000_000)}mio"

    print("loading data")
    data = np.load(
        pjoin(
            pardir,
            pardir,
            "data",
            "preproc",
            f"simthresh-{args.simthresh}_trans-{args.trans}",
            f"subj-{args.sub}_features.npy",
        )
    )
    out_path = pjoin(
        pardir,
        pardir,
        "data",
        "preproc",
        f"simthresh-{args.simthresh}_trans-{args.trans}",
        "triplets",
        f"triplets_subj-{args.sub}_tripdist-euclidean",
        f"triplets_{nmiostr}",
    )

    print("starting tripletization")
    calc_triplets(
        out_path=out_path,
        n_samples=args.ntrips,
        data=data,
        dist_method="euclidean",
    )

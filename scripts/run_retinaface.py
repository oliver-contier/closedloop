from closedloop.data import get_thingsimages_fnames
import numpy as np
import os
from os.path import join as pjoin
import json
from closedloop.utils import NpEncoder
from closedloop.plotting import plot_list_of_images
from closedloop.facelabels import run_retinafaces_parallel


def main(
    threshold = .95,
    out_basedir = pjoin(os.pardir, 'data', 'facelabels'),
    test_nims = 0,
    plot_dpi = 150,
    n_jobs=20,
):
    # get things images
    things_fnames = get_thingsimages_fnames()
    if test_nims:  # sample random images for exploring different thresholds
        things_fnames = things_fnames[np.random.randint(0, len(things_fnames), test_nims)]
    # make outdir
    outdir = pjoin(out_basedir, f'thresh-{threshold:.2f}')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # run retinaface
    hits_dicts, hits_inds, hits_onehot = run_retinafaces_parallel(
        things_fnames, threshold, n_jobs=n_jobs
        )
    hits_fnames = things_fnames[hits_onehot]
    # save data
    np.save(pjoin(outdir, 'hits_fnames.npy'), hits_fnames)
    np.save(pjoin(outdir, 'hits_onehot.npy'), hits_onehot)
    np.save(pjoin(outdir, 'hits_inds.npy'), hits_inds)
    with open(pjoin(outdir, 'hits_dicts.json'), 'w') as f:
        json.dump(hits_dicts, f, cls=NpEncoder)
    # make plot
    fig = plot_list_of_images(hits_fnames)
    fig.savefig(pjoin(outdir, 'hits.png'), dpi=plot_dpi)
    return None


if __name__ == "__main__":
    main()
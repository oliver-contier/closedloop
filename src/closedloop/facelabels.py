from tqdm import tqdm
import numpy as np
from os.path import join as pjoin
from os.path import pardir, dirname
import json
from joblib import Parallel, delayed


def run_detect_faces(fname, threshold):
    # Heavy detection dependencies are imported lazily so that downstream
    # analyses (load_facelabels) work without retinaface / opencv installed.
    import cv2
    from retinaface.RetinaFace import detect_faces

    img = cv2.imread(fname)
    result = detect_faces(img, threshold=threshold)
    if type(result) == dict:
        return result
    else:
        assert type(result) == tuple, f'unrecognized return type from detect_faces {type(result)}'
        return None


def run_retinafaces_parallel(fnames, threshold, n_jobs=10):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_detect_faces)(fname, threshold) 
        for fname in tqdm(fnames)
        )
    results = [(i, r) for i, r in enumerate(results) if r != None]
    hits_inds, hits_dicts = zip(*results)
    hits_inds = np.array(hits_inds)
    hits_onehot = np.zeros(len(fnames), dtype=bool)
    hits_onehot[hits_inds] = 1
    return hits_dicts, hits_inds, hits_onehot


def load_facelabels(
    datadir=pjoin(dirname(__file__), pardir, pardir, "data"),
    facethresh = .90
    ):
    facelabels_dir = pjoin(datadir, 'facelabels', f'thresh-{facethresh:.2f}')
    hits_dicts_json = pjoin(facelabels_dir, 'hits_dicts.json')
    with open(hits_dicts_json, 'r') as f:
        hits_dicts = json.load(f)
    hits_fnames = np.load(pjoin(facelabels_dir, 'hits_fnames.npy'))
    faces_inds = np.load(pjoin(facelabels_dir, 'hits_inds.npy'))
    faces_onehot = np.load(pjoin(facelabels_dir, 'hits_onehot.npy')).astype(bool)
    return hits_dicts, hits_fnames, faces_inds, faces_onehot
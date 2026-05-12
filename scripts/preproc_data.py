from os.path import join as pjoin
from os.path import pardir, dirname, exists
from os import makedirs
import numpy as np
from closedloop.data import remove_outlier_features, load_data, npsave_mkdir
from tqdm import tqdm
from scipy.stats import zscore


def preproc(
    simthresh=0.,
    datadir=pjoin(dirname(__file__), pardir, 'data'),
):
    subjects = ['1', '2', '3', '4', 'all']
    for subj in tqdm(subjects, desc='subjects'):
        print('Loading Data')
        x_ = load_data(subj=subj, datadir=datadir)
        if simthresh:
            print(f'removing outlier voxels, threshold {simthresh}')
            x = remove_outlier_features(x_, simthresh=simthresh)
        else:
            x = x_
        # print('negative shift')
        # xneg = x.copy()
        # xneg -= xneg.max()
        # print('positive shift')
        # xpos = x - x.min()
        print('voxel-wise z-scoring')
        zvox = zscore(x, 0)
        # print('positive shift of z-scored data')
        # zvoxpos = zvox + np.absolute(zvox.min(0))
        # print('negative shift of z-scored data')
        # zvoxneg = zvox - np.absolute(zvox.max(0))
        
        # arrs = [x, xneg, xpos, zvox, zvoxpos, zvoxneg]
        # transnames = ['none', 'negshift', 'posshift', 'zvox', 'zvoxpos', 'zvoxneg']
        arrs = [x, zvox]
        transnames = ['none', 'zvox']
        
        for arr, transname in zip(arrs, transnames):
            outf = pjoin(
                datadir, 'preproc', f"simthresh-{simthresh:.1f}_trans-{transname}", f'subj-{subj}_features.npy'
                )
            print(f'saving data to {outf}')
            npsave_mkdir(arr, outf)
    return None


if __name__ == '__main__':
    preproc()

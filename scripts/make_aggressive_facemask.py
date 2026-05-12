"""Create a more aggressive pydeface facemask that also zeroes the orbital
(eye) region. We extend the existing pydeface mask by setting to 0 any voxel
in the anterior-inferior region of MNI space (anterior to y_cut, below z_cut).
"""
import nibabel as nib
import numpy as np
from pathlib import Path

PYDEFACE_DATA = Path(
    "/Users/contier/projects/ffadims/ffadims_1/.venv/lib/python3.11/"
    "site-packages/pydeface/data"
)
SRC = PYDEFACE_DATA / "facemask.nii.gz"
DST = "/Users/contier/projects/ffadims/ffadims_1/scripts/facemask_aggressive.nii.gz"

# MNI-space cutoffs (mm).  Voxels with y > Y_CUT and z < Z_CUT are zeroed.
# Eyes sit roughly y ~ +25..+50, z ~ -20..+5.
Y_CUT = 15.0
Z_CUT = 15.0

img = nib.load(str(SRC))
data = img.get_fdata().astype(np.float32)
aff = img.affine

ii, jj, kk = np.meshgrid(
    np.arange(data.shape[0]),
    np.arange(data.shape[1]),
    np.arange(data.shape[2]),
    indexing="ij",
)
coords = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4)
mm = coords @ aff.T
mm = mm.reshape(data.shape + (4,))
y = mm[..., 1]
z = mm[..., 2]
zero_region = (y > Y_CUT) & (z < Z_CUT)

new = data.copy()
new[zero_region] = 0
nib.Nifti1Image(new, aff, img.header).to_filename(DST)

print(f"original mask kept voxels: {int(data.sum())}")
print(f"new mask kept voxels:      {int(new.sum())}")
print(f"additionally zeroed:       {int((data > 0).sum() - (new > 0).sum())}")
print(f"saved {DST}")

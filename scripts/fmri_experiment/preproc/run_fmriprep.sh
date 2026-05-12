#!/usr/bin/env bash

# subject=$1
nprocs=10
maxmem=15

# When this script is in scripts/fmri_experiment/preproc, go up three levels to project root
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
rawdatadir="${project_root}/data/fmri_experiment/bids"
derivsdir="${project_root}/data/fmri_experiment/bids/derivatives/fmriprep"
workdir="${project_root}/scripts/fmri_experiment/fmriprep_wdir"
fs_license_dir="${project_root}/data/freesurfer_license"

docker run -ti --rm \
  --memory="${maxmem}g" \
  -v "$rawdatadir":/data:ro \
  -v "$derivsdir":/out \
  -v "$workdir":/work \
  -v "$fs_license_dir":/fs_license_dir \
  nipreps/fmriprep:23.2.2 \
  --fs-no-reconall \
  --nprocs "$nprocs" \
  --output-spaces T1w func MNI152NLin2009cAsym \
  --fs-license-file /fs_license_dir/license.txt \
  -w /work /data /out participant

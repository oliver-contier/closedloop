"""Deface all T1w images in-place using pydeface."""
import subprocess
import sys
from glob import glob
from pathlib import Path

ROOT = "/Users/contier/projects/ffadims/ffadims_1/data/fmri_experiment/openneuro"

t1ws = sorted(glob(f"{ROOT}/sub-*/ses-*/anat/*_T1w.nii.gz"))
print(f"defacing {len(t1ws)} T1w files\n")

for i, t1 in enumerate(t1ws, 1):
    p = Path(t1)
    tmp = p.with_name(p.name.replace(".nii.gz", "_defaced.nii.gz"))
    print(f"[{i}/{len(t1ws)}] {p.name}")
    r = subprocess.run(
        ["pydeface", "--force", "--outfile", str(tmp), str(p)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(f"pydeface failed on {p}")
    tmp.replace(p)
    print(f"    -> overwrote {p.name}")

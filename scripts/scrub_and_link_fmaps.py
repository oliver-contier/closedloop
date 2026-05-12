"""Scrub PII from JSON sidecars and add IntendedFor to fmap sidecars.

PII fields removed from every sidecar:
    InstitutionAddress, DeviceSerialNumber, StationName,
    AcquisitionTime, ProcedureStepDescription

fmap -> BOLD mapping (option A, temporal-split): for each session, the two
phasediff fmaps bracket the BOLDs in time. Each BOLD is assigned to the
temporally-closest fmap (midpoint split). IntendedFor is written to all three
sidecars per fmap (phasediff, magnitude1, magnitude2).
"""
import json
from glob import glob
from os.path import basename, join as pjoin

ROOT = "/Users/contier/projects/ffadims/ffadims_1/data/fmri_experiment/openneuro"
PII_FIELDS = [
    "InstitutionAddress",
    "DeviceSerialNumber",
    "StationName",
    "AcquisitionTime",
    "ProcedureStepDescription",
]


def parse_time(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def session_fmap_intendedfor(sub_dir, ses):
    """Return {fmap_run_label: [IntendedFor relpaths]} for one session."""
    ses_dir = pjoin(sub_dir, ses)
    bolds = sorted(glob(pjoin(ses_dir, "func", "*_bold.nii.gz")))
    fmaps = sorted(glob(pjoin(ses_dir, "fmap", "*_phasediff.json")))
    assert len(fmaps) == 2, f"expected 2 phasediff fmaps in {ses_dir}, got {len(fmaps)}"

    fmap_times = {}
    for f in fmaps:
        d = json.load(open(f))
        run = basename(f).split("_run-")[1].split("_")[0]
        fmap_times[run] = parse_time(d["AcquisitionTime"])
    runs_sorted = sorted(fmap_times, key=lambda r: fmap_times[r])
    early_run, late_run = runs_sorted
    midpoint = (fmap_times[early_run] + fmap_times[late_run]) / 2

    assigned = {early_run: [], late_run: []}
    for bold in bolds:
        sidecar = bold.replace(".nii.gz", ".json")
        d = json.load(open(sidecar))
        t = parse_time(d["AcquisitionTime"])
        target = early_run if t < midpoint else late_run
        relpath = f"{ses}/func/{basename(bold)}"
        assigned[target].append(relpath)
    return assigned


def update_fmap_sidecars(sub_dir, ses, assigned):
    for run, intended in assigned.items():
        for kind in ("phasediff", "magnitude1", "magnitude2"):
            fs = glob(pjoin(sub_dir, ses, "fmap", f"*_run-{run}_{kind}.json"))
            assert len(fs) == 1, f"missing {kind} run-{run} in {sub_dir}/{ses}"
            path = fs[0]
            d = json.load(open(path))
            d["IntendedFor"] = intended
            with open(path, "w") as fh:
                json.dump(d, fh, indent=2)


def scrub_pii(path):
    d = json.load(open(path))
    changed = False
    for k in PII_FIELDS:
        if k in d:
            del d[k]
            changed = True
    if changed:
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)


def main():
    subjects = sorted(glob(pjoin(ROOT, "sub-*")))
    for sub_dir in subjects:
        sessions = sorted(
            basename(p) for p in glob(pjoin(sub_dir, "ses-*")) if "/ses-" in p or True
        )
        sessions = [s for s in sessions if s.startswith("ses-")]
        for ses in sessions:
            assigned = session_fmap_intendedfor(sub_dir, ses)
            update_fmap_sidecars(sub_dir, ses, assigned)
            counts = {r: len(v) for r, v in assigned.items()}
            print(f"{basename(sub_dir)} {ses}  fmap->bold counts: {counts}")

    all_jsons = glob(pjoin(ROOT, "sub-*", "ses-*", "*", "*.json"))
    for j in all_jsons:
        scrub_pii(j)
    print(f"scrubbed PII from {len(all_jsons)} sidecars")


if __name__ == "__main__":
    main()

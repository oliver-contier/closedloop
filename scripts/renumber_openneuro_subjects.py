"""Renumber subjects in the openneuro/ folder: sub-02..07 -> sub-01..06.

Processes in increasing order so target IDs are always free.
Only touches files inside the openneuro/ directory.
"""
import os
from pathlib import Path

ROOT = Path("/Users/contier/projects/ffadims/ffadims_1/data/fmri_experiment/openneuro")
MAPPING = [(f"sub-{old:02d}", f"sub-{old - 1:02d}") for old in range(2, 8)]
TEXT_EXTS = {".json", ".tsv", ".md", ".txt"}


def rewrite_text(path: Path, old: str, new: str) -> None:
    try:
        content = path.read_text()
    except UnicodeDecodeError:
        return
    if old in content:
        path.write_text(content.replace(old, new))


def rename_subject(old: str, new: str) -> None:
    sub_dir = ROOT / old
    if not sub_dir.is_dir():
        print(f"  skip: {sub_dir} not a directory")
        return
    # Rename files (and rewrite text content) inside, walking bottom-up
    for dirpath, dirnames, filenames in os.walk(sub_dir, topdown=False):
        for fname in filenames:
            old_path = Path(dirpath) / fname
            if fname.endswith(".tsv") or fname.endswith(".json") or fname.endswith(".md") or fname.endswith(".txt") or fname == "README":
                rewrite_text(old_path, old, new)
            if old in fname:
                new_path = Path(dirpath) / fname.replace(old, new)
                old_path.rename(new_path)
    # Rename the subject directory itself
    sub_dir.rename(ROOT / new)
    print(f"  {old} -> {new}")


def main() -> None:
    print("renaming subject dirs and files:")
    for old, new in MAPPING:
        rename_subject(old, new)

    print("\nupdating top-level files:")
    for top in ("participants.tsv", "README"):
        p = ROOT / top
        if not p.exists():
            continue
        content = p.read_text()
        for old, new in MAPPING:
            content = content.replace(old, new)
        p.write_text(content)
        print(f"  {top}")


if __name__ == "__main__":
    main()

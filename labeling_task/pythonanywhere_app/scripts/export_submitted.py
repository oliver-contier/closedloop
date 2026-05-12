from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
RESPONSES_PATH = DATA_DIR / "responses_long.csv"
PARTICIPANTS_PATH = DATA_DIR / "participants.csv"
DEFAULT_OUT = DATA_DIR / "export_submitted_latest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export latest labels for submitted participants."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not RESPONSES_PATH.exists():
        raise FileNotFoundError(f"Missing {RESPONSES_PATH}")
    if not PARTICIPANTS_PATH.exists():
        raise FileNotFoundError(f"Missing {PARTICIPANTS_PATH}")

    participants = pd.read_csv(PARTICIPANTS_PATH)
    completed_tokens = set(
        participants.loc[participants["submitted"] == 1, "token"].astype(str)
    )

    if not completed_tokens:
        print("No submitted participants found.")
        return

    df = pd.read_csv(RESPONSES_PATH)
    df = df[df["token"].astype(str).isin(completed_tokens)].copy()
    if df.empty:
        print("No responses found for submitted participants.")
        return

    # Keep latest entry per token+dim_id
    df = df.sort_values("updated_at")
    df = df.drop_duplicates(subset=["token", "dim_id"], keep="last")

    cols = ["token", "trial_index", "dim_id", "label", "updated_at"]
    out_df = df[cols].sort_values(["token", "trial_index"])
    out_df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()

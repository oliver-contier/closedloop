from __future__ import annotations

import argparse
import csv
import secrets
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
TOKENS_PATH = DATA_DIR / "tokens.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate participant tokens.")
    parser.add_argument("--count", type=int, default=50, help="Number of tokens.")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://your-subdomain.pythonanywhere.com/t",
        help="Base URL printed to terminal.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokens = [secrets.token_urlsafe(12) for _ in range(args.count)]

    with TOKENS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["token"])
        writer.writeheader()
        for token in tokens:
            writer.writerow({"token": token})

    print(f"Wrote {len(tokens)} tokens to {TOKENS_PATH}")
    print("Share these links:")
    for token in tokens:
        print(f"{args.base_url}/{token}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List

from flask import Flask, abort, make_response, redirect, render_template, request, url_for


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = APP_DIR / "data"
TRIALS_PATH = DATA_DIR / "trials.csv"
TOKENS_PATH = DATA_DIR / "tokens.csv"
PARTICIPANTS_PATH = DATA_DIR / "participants.csv"
RESPONSES_PATH = DATA_DIR / "responses_long.csv"
EXAMPLES_DIR = APP_DIR / "static" / "examples"
EXAMPLES_MANIFEST = EXAMPLES_DIR / "examples.csv"

PARTICIPANT_HEADERS = [
    "token",
    "started_at",
    "last_seen_at",
    "completed_at",
    "submitted",
    "trial_order_json",
]
RESPONSE_HEADERS = [
    "token",
    "dim_id",
    "trial_index",
    "label",
    "updated_at",
    "is_submitted",
]

_CSV_LOCK = Lock()


def load_local_env() -> None:
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PARTICIPANTS_PATH.exists():
        with PARTICIPANTS_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PARTICIPANT_HEADERS)
            writer.writeheader()

    if not RESPONSES_PATH.exists():
        with RESPONSES_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESPONSE_HEADERS)
            writer.writeheader()


def load_trials() -> List[Dict[str, str]]:
    if not TRIALS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRIALS_PATH}. Run scripts/generate_grids.py first."
        )

    with TRIALS_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{TRIALS_PATH} is empty.")
    return rows


def load_valid_tokens() -> set[str]:
    if not TOKENS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TOKENS_PATH}. Run scripts/make_tokens.py first."
        )

    with TOKENS_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["token"].strip() for r in rows if r.get("token", "").strip()}


def load_participants() -> Dict[str, Dict[str, str]]:
    with PARTICIPANTS_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["token"]: r for r in rows}


def write_participants(participants: Dict[str, Dict[str, str]]) -> None:
    with PARTICIPANTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PARTICIPANT_HEADERS)
        writer.writeheader()
        for token in sorted(participants.keys()):
            writer.writerow(participants[token])


def append_response(row: Dict[str, str]) -> None:
    with RESPONSES_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESPONSE_HEADERS)
        writer.writerow(row)


def latest_labels_by_token(token: str) -> Dict[str, str]:
    latest: Dict[str, str] = {}
    with RESPONSES_PATH.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["token"] != token:
                continue
            latest[row["dim_id"]] = row["label"]
    return latest


def load_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    candidate_dirs: List[Path] = []
    if EXAMPLES_DIR.exists():
        candidate_dirs.append(EXAMPLES_DIR)
    if STATIC_DIR.exists():
        for p in sorted(STATIC_DIR.iterdir()):
            if p.is_dir() and "example" in p.name.lower() and p not in candidate_dirs:
                candidate_dirs.append(p)
    if not candidate_dirs:
        return examples

    # Prefer manifest-driven examples if available in any candidate directory.
    for examples_dir in candidate_dirs:
        manifest_path = examples_dir / "examples.csv"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image = row.get("image", "").strip()
                label = row.get("label", "").strip()
                if image:
                    examples.append(
                        {
                            "image_path": f"{examples_dir.name}/{image}",
                            "label_text": label or "Example labels not provided.",
                        }
                    )
        if examples:
            random.shuffle(examples)
            return examples

    # Fall back to deriving labels from filenames.
    for examples_dir in candidate_dirs:
        for path in sorted(examples_dir.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            label_text = path.stem.replace("__", " - ").replace("_", ", ")
            examples.append(
                {
                    "image_path": f"{examples_dir.name}/{path.name}",
                    "label_text": label_text,
                }
            )
    random.shuffle(examples)
    return examples


def make_app() -> Flask:
    load_local_env()
    app = Flask(__name__)
    app.secret_key = os.environ.get("APP_SECRET_KEY", "dev-secret-key-change-me")

    @app.route("/")
    def root():
        return render_template("start.html")

    @app.get("/claim")
    def claim_token():
        ensure_storage()
        valid_tokens = load_valid_tokens()
        if not valid_tokens:
            return render_template("no_tokens.html"), 503

        cookie_token = request.cookies.get("claimed_token", "").strip()
        if cookie_token and cookie_token in valid_tokens:
            return redirect(url_for("token_start", token=cookie_token))

        with _CSV_LOCK:
            participants = load_participants()
            used_tokens = set(participants.keys())
            available = sorted(valid_tokens - used_tokens)
            if not available:
                return render_template("no_tokens.html"), 503
            token = available[0]

        resp = make_response(redirect(url_for("token_start", token=token)))
        # Long-lived cookie so participants can reopen the same link.
        resp.set_cookie("claimed_token", token, max_age=60 * 60 * 24 * 180, samesite="Lax")
        return resp

    @app.get("/t/<token>")
    def token_start(token: str):
        ensure_storage()

        trials = load_trials()
        valid_tokens = load_valid_tokens()
        if token not in valid_tokens:
            return render_template("invalid.html"), 404

        now = utc_now_iso()
        with _CSV_LOCK:
            participants = load_participants()
            participant = participants.get(token)
            if participant is None:
                import random

                order = list(range(len(trials)))
                random.shuffle(order)
                participant = {
                    "token": token,
                    "started_at": now,
                    "last_seen_at": now,
                    "completed_at": "",
                    "submitted": "0",
                    "trial_order_json": json.dumps(order),
                }
            else:
                participant["last_seen_at"] = now
            participants[token] = participant
            write_participants(participants)

        order = json.loads(participant["trial_order_json"])
        labels = latest_labels_by_token(token)
        if participant["submitted"] == "1":
            return redirect(url_for("complete", token=token))
        examples = load_examples()
        return render_template(
            "intro.html",
            token=token,
            n_trials=len(order),
            examples=examples,
            answered=sum(
                1 for i in order if labels.get(trials[i]["dim_id"], "").strip()
            ),
        )

    @app.post("/t/<token>/begin")
    def begin_task(token: str):
        ensure_storage()
        trials = load_trials()
        valid_tokens = load_valid_tokens()
        if token not in valid_tokens:
            return render_template("invalid.html"), 404

        with _CSV_LOCK:
            participants = load_participants()
            participant = participants.get(token)
            if participant is None:
                return redirect(url_for("token_start", token=token))
            if participant["submitted"] == "1":
                return redirect(url_for("complete", token=token))
            participant["last_seen_at"] = utc_now_iso()
            participants[token] = participant
            write_participants(participants)

        order = json.loads(participant["trial_order_json"])
        labels = latest_labels_by_token(token)
        for idx in range(1, len(order) + 1):
            dim_id = trials[order[idx - 1]]["dim_id"]
            if not labels.get(dim_id, "").strip():
                return redirect(url_for("trial_view", token=token, trial_num=idx))
        return redirect(url_for("review", token=token))

    @app.get("/t/<token>/trial/<int:trial_num>")
    def trial_view(token: str, trial_num: int):
        ensure_storage()
        trials = load_trials()
        valid_tokens = load_valid_tokens()
        if token not in valid_tokens:
            return render_template("invalid.html"), 404

        with _CSV_LOCK:
            participants = load_participants()
            participant = participants.get(token)
            if participant is None:
                return redirect(url_for("token_start", token=token))
            participant["last_seen_at"] = utc_now_iso()
            participants[token] = participant
            write_participants(participants)

        if participant["submitted"] == "1":
            return redirect(url_for("complete", token=token))

        order = json.loads(participant["trial_order_json"])
        n_trials = len(order)
        if trial_num < 1 or trial_num > n_trials:
            abort(404)

        trial = trials[order[trial_num - 1]]
        labels = latest_labels_by_token(token)
        current_label = labels.get(trial["dim_id"], "")

        return render_template(
            "trial.html",
            token=token,
            trial_num=trial_num,
            n_trials=n_trials,
            trial=trial,
            current_label=current_label,
        )

    @app.post("/t/<token>/trial/<int:trial_num>/save")
    def trial_save(token: str, trial_num: int):
        ensure_storage()
        trials = load_trials()
        valid_tokens = load_valid_tokens()
        if token not in valid_tokens:
            return render_template("invalid.html"), 404

        direction = request.form.get("direction", "next")
        label = request.form.get("label", "").strip()

        with _CSV_LOCK:
            participants = load_participants()
            participant = participants.get(token)
            if participant is None:
                return redirect(url_for("token_start", token=token))
            if participant["submitted"] == "1":
                return redirect(url_for("complete", token=token))

            participant["last_seen_at"] = utc_now_iso()
            participants[token] = participant
            write_participants(participants)

            order = json.loads(participant["trial_order_json"])
            n_trials = len(order)
            if trial_num < 1 or trial_num > n_trials:
                abort(404)

            trial = trials[order[trial_num - 1]]
            append_response(
                {
                    "token": token,
                    "dim_id": trial["dim_id"],
                    "trial_index": str(trial_num),
                    "label": label,
                    "updated_at": utc_now_iso(),
                    "is_submitted": "0",
                }
            )

        if direction == "prev":
            next_trial = max(1, trial_num - 1)
            return redirect(url_for("trial_view", token=token, trial_num=next_trial))
        if direction == "review":
            return redirect(url_for("review", token=token))
        next_trial = min(n_trials, trial_num + 1)
        return redirect(url_for("trial_view", token=token, trial_num=next_trial))

    @app.get("/t/<token>/review")
    def review(token: str):
        ensure_storage()
        trials = load_trials()
        valid_tokens = load_valid_tokens()
        if token not in valid_tokens:
            return render_template("invalid.html"), 404

        with _CSV_LOCK:
            participants = load_participants()
            participant = participants.get(token)
            if participant is None:
                return redirect(url_for("token_start", token=token))
            if participant["submitted"] == "1":
                return redirect(url_for("complete", token=token))
            participant["last_seen_at"] = utc_now_iso()
            participants[token] = participant
            write_participants(participants)

        order = json.loads(participant["trial_order_json"])
        latest = latest_labels_by_token(token)
        rows = []
        all_complete = True
        for idx, trial_idx in enumerate(order, start=1):
            trial = trials[trial_idx]
            label = latest.get(trial["dim_id"], "").strip()
            if not label:
                all_complete = False
            rows.append(
                {
                    "trial_num": idx,
                    "dim_id": trial["dim_id"],
                    "label": label,
                }
            )
        return render_template(
            "review.html",
            token=token,
            rows=rows,
            all_complete=all_complete,
        )

    @app.post("/t/<token>/submit")
    def submit(token: str):
        ensure_storage()
        trials = load_trials()
        valid_tokens = load_valid_tokens()
        if token not in valid_tokens:
            return render_template("invalid.html"), 404

        with _CSV_LOCK:
            participants = load_participants()
            participant = participants.get(token)
            if participant is None:
                return redirect(url_for("token_start", token=token))
            if participant["submitted"] == "1":
                return redirect(url_for("complete", token=token))

            order = json.loads(participant["trial_order_json"])
            latest = latest_labels_by_token(token)

            for idx, trial_idx in enumerate(order, start=1):
                trial = trials[trial_idx]
                label = latest.get(trial["dim_id"], "").strip()
                if not label:
                    return redirect(url_for("trial_view", token=token, trial_num=idx))

            for idx, trial_idx in enumerate(order, start=1):
                trial = trials[trial_idx]
                append_response(
                    {
                        "token": token,
                        "dim_id": trial["dim_id"],
                        "trial_index": str(idx),
                        "label": latest[trial["dim_id"]].strip(),
                        "updated_at": utc_now_iso(),
                        "is_submitted": "1",
                    }
                )

            participant["submitted"] = "1"
            participant["completed_at"] = utc_now_iso()
            participant["last_seen_at"] = utc_now_iso()
            participants[token] = participant
            write_participants(participants)

        return redirect(url_for("complete", token=token))

    @app.get("/t/<token>/complete")
    def complete(token: str):
        ensure_storage()
        valid_tokens = load_valid_tokens()
        if token not in valid_tokens:
            return render_template("invalid.html"), 404
        return render_template("complete.html")

    return app


app = make_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

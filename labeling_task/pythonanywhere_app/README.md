# ffadims-labeling-app

Lightweight Flask app for collecting human labels for image-dimension grids.

## Purpose

This task is designed to help interpret embedding dimensions.

- Participants see **15 image grids** (one grid per dimension).
- For each grid, they enter **3 descriptive words**.
- Labels can later be aggregated (for example into word clouds) to summarize what each dimension captures.

## Task flow

Participants can enter via:

- `/claim` (recommended): auto-assigns one unique unused token
- `/t/<token>`: direct token link

Flow per participant:

1. Intro page with instructions and optional examples
2. 15 labeling trials
3. Review page
4. Final submit

Responses are written to CSV files in `data/`.

## Repository layout

- `app.py` - main Flask app, routing, token assignment, CSV storage
- `wsgi.py` - WSGI entrypoint for deployment
- `templates/` - intro/trial/review/complete pages
- `static/grids/` - generated grid stimuli (`d01.png` ... `d15.png`)
- `static/examples/` - optional intro examples
- `scripts/generate_grids.py` - generate grid images + `data/trials.csv`
- `scripts/make_tokens.py` - generate participant token pool
- `scripts/export_submitted.py` - create clean export of submitted results
- `DEPLOY_PYTHONANYWHERE.md` - deployment guide

## Local setup (reproducible)

From repo root:

```bash
cd labeling_task/pythonanywhere_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install h5py
```

Create a secret key:

```bash
echo "APP_SECRET_KEY=$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
)" > .env
```

Generate stimuli and trial metadata:

```bash
cd /path/to/ffadims
source labeling_task/pythonanywhere_app/.venv/bin/activate
python labeling_task/pythonanywhere_app/scripts/generate_grids.py
```

Generate tokens:

```bash
python labeling_task/pythonanywhere_app/scripts/make_tokens.py --count 100 --base-url "http://127.0.0.1:8000/t"
```

Run locally:

```bash
python labeling_task/pythonanywhere_app/app.py
```

Open:

- `http://127.0.0.1:8000/claim`

## Examples (optional)

Put example images in `static/examples/`.

Optional manifest file:

- `static/examples/examples.csv`
- columns: `image,label`

If no manifest is provided, the app derives example labels from filenames.

## PythonAnywhere deployment (summary)

1. Upload or clone this app folder on PythonAnywhere.
2. Create a virtualenv and install dependencies.
3. Set `.env` with `APP_SECRET_KEY`.
4. Configure Web app:
   - source code path
   - working directory
   - virtualenv path
   - static mapping `/static/ -> .../static`
   - WSGI file imports `app` from this folder (`wsgi.py`)
5. Run:
   - `scripts/generate_grids.py`
   - `scripts/make_tokens.py --count ... --base-url "https://<your-domain>/t"`
6. Reload the web app.

Detailed steps: see `DEPLOY_PYTHONANYWHERE.md`.

## Data files

Runtime:

- `data/participants.csv` - participant status per token
- `data/responses_long.csv` - trial-level edits and final submissions

Clean export:

```bash
source .venv/bin/activate
python scripts/export_submitted.py
```

Output:

- `data/export_submitted_latest.csv` (latest submitted label per token x dimension)

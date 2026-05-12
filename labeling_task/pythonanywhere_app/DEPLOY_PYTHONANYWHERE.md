# Deploy on PythonAnywhere

## 1) Upload project
- Upload `labeling_task/pythonanywhere_app` to your PythonAnywhere account.
- Keep this folder together (templates/static/scripts/data).

## 2) Create virtual environment
```bash
cd ~/path/to/pythonanywhere_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Generate stimuli and trials
From your repository root (where embedding + THINGS image paths resolve):
```bash
source .venv/bin/activate
python labeling_task/pythonanywhere_app/scripts/generate_grids.py
```
This creates:
- `static/grids/*.png`
- `data/trials.csv`

## 4) Generate participant tokens
```bash
python labeling_task/pythonanywhere_app/scripts/make_tokens.py --count 40 --base-url "https://YOURNAME.pythonanywhere.com/t"
```
This writes `data/tokens.csv` and prints personal links.

## 5) Configure web app
- In PythonAnywhere dashboard: **Web** -> **Add a new web app**.
- Choose Flask/manual config.
- Point WSGI file to:
  `.../labeling_task/pythonanywhere_app/wsgi.py`
- Ensure static files are enabled automatically (Flask serves `/static/...`).

## 6) Set secret key
- In the WSGI file (or dashboard env var), set:
  - `APP_SECRET_KEY` to a long random string.

## 7) Reload and test
- Reload the web app in dashboard.
- Open one token URL:
  `https://YOURNAME.pythonanywhere.com/t/<token>`

## 8) Export submitted responses
```bash
python labeling_task/pythonanywhere_app/scripts/export_submitted.py
```
Output:
- `data/export_submitted_latest.csv`

## Notes
- No identifying information is collected; only token + labels + timestamps.
- Participants can revisit their token link and edit labels until they press submit.
- Trial order is randomized once per token and then fixed.

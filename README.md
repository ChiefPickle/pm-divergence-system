# pm-divergence-system

Research validation stack: attention signals vs prediction-market dynamics (causal snapshots, synthetic backtest). Not production trading infrastructure.

See [`docs/research_charter.md`](docs/research_charter.md).

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Streamlit Community Cloud

The folder you open in Cursor must be **this repository** (the one with `apps/dashboard/app.py`), not a parent folder without Git.

1. Push the **latest** `main` branch to GitHub (this repo).
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), **Create app** → pick **GitHub** → authorize if needed.
3. Select repository: **`ChiefPickle/pm-divergence-system`** (or your fork).
4. Branch: **`main`**.
5. **Main file path:** `apps/dashboard/app.py`
6. **App URL** (optional): leave default or set a subdomain.

`requirements.txt` at the repo root is used automatically. The dashboard adds `src/` to `sys.path` at runtime; no extra Streamlit secrets are required for the MVP.

## Other commands

```bash
PYTHONPATH=src:. python backtests/run_backtest.py
PYTHONPATH=src:. uvicorn apps.api.main:app --reload --port 8000
PYTHONPATH=src:. streamlit run apps/dashboard/app.py
```

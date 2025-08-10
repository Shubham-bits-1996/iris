

# Iris ML Ops – End-to-End Project

![CI](https://github.com/Shubham-bits-1996/iris/actions/workflows/ci.yml/badge.svg)


A production-style Iris classifier with training + experiment tracking, FastAPI serving, Docker packaging, CI/CD, logging, metrics, and optional retraining on new data.

## TL;DR — Run with Docker (fast)

```bash
export DOCKERHUB_USERNAME=<your-dockerhub-username> && ./deploy.sh
# Then open http://localhost:8000/docs
```

## Features

- **Models & tracking:** Logistic Regression & Random Forest tracked in **MLflow**; best model exported to `model/best_iris_model.pkl`.
- **API:** **FastAPI** `/predict` returns class + probabilities with **Pydantic validation**.
- **Container:** Minimal **Docker** image with `uvicorn`.
- **CI/CD:** **GitHub Actions** lint/tests → build Docker image → push to Docker Hub → local deploy script.
- **Observability:** Rotating file logs, **SQLite** request log, Prometheus `/metrics`, optional Grafana dashboard.
- **Retraining (bonus):** GitHub Action retrains + rebuilds image when `data/**` changes.

---

## Repository structure

```
.
├─ app/
│  └─ main.py                 # FastAPI app (/predict, /metrics, logging, SQLite)
├─ model/
│  └─ best_iris_model.pkl     # exported best model used by the API
├─ tests/
│  └─ test_app.py             # unit test for /predict
├─ data/                      # place “new data” files here to trigger retrain workflow
├─ train.py                   # trains two models, logs to MLflow, exports best
├─ requirements.txt
├─ Dockerfile
├─ deploy.sh                  # pulls latest image, runs container (port 8000→80)
├─ .github/workflows/
│  ├─ ci.yml                  # lint + tests + docker build & push
│  └─ retrain.yml             # retrain & rebuild on data/** change
└─ (logs/                     # created at runtime: app.log + logs.db)
```

---

## Quickstart

### 1) Local (no Docker)

**Prereqs:** Python 3.9+

```bash
git clone https://github.com/Shubham-bits-1996/iris.git
cd iris
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Run tests & lint:**
```bash
flake8 .
pytest -q
```

**Train models & export best (optional):**
_Skip this if you just want to run the API — `model/best_iris_model.pkl` is already included._
```bash
python train.py
# MLflow logs go to ./mlruns
```

**Serve the API:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Try a prediction:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

**Docs:** http://localhost:8000/docs  
**Metrics:** http://localhost:8000/metrics

### 2) Docker (recommended)

**Build locally (optional):**
```bash
docker build -t <your-dockerhub-username>/iris-api:latest .
```

**Or pull & run the CI-built image using the helper script:**
```bash
export DOCKERHUB_USERNAME=<your-dockerhub-username>
./deploy.sh
# exposes API at http://localhost:8000, mounts ./logs into container
```

**Test:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":6.2,"sepal_width":2.8,"petal_length":4.8,"petal_width":1.8}'
```

**Stop & cleanup:**
```bash
docker stop iris-api && docker rm iris-api
```

---

## Experiment tracking (MLflow)

`train.py` trains LogisticRegression & RandomForest, logs params/metrics/models to `mlruns/`, and exports the best model to `model/best_iris_model.pkl`. To view the UI locally:

```bash
pip install mlflow
mlflow ui --backend-store-uri mlruns --port 5000
# open http://localhost:5000
```

---

## Observability

### File logs
A rotating file handler writes to `logs/app.log`. You’ll see startup messages, inputs, predicted class, probabilities, and latency.

```bash
tail -n 20 logs/app.log
```

### SQLite request log
Each `/predict` call is persisted to `logs/logs.db` in a `predictions` table (inputs, output, per-class probabilities, latency).

```bash
python - <<'PY'
import sqlite3; con=sqlite3.connect('logs/logs.db')
for r in con.execute("SELECT ts,class_name,latency_ms FROM predictions ORDER BY id DESC LIMIT 5"):
    print(r)
con.close()
PY
```

> When running via Docker, `deploy.sh` mounts `./logs` into the container so logs and DB stay on your host.

### Prometheus metrics
The API exposes `/metrics` with:
- `predict_requests_total` (Counter)
- `predict_latency_seconds` (Histogram)

**Run Prometheus locally:**
1) Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: "iris-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8000"]  # or "localhost:8000" if Prometheus not in Docker
```

2) Start Prometheus:
```bash
docker rm -f prometheus 2>/dev/null || true
docker run -d --name prometheus \
  -p 9090:9090 \
  -v "$(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus
```

**(Optional) Grafana dashboard:**
```bash
docker rm -f grafana 2>/dev/null || true
docker run -d --name grafana -p 3000:3000 grafana/grafana-oss
# Login: admin / admin → set password
# Add data source: Prometheus @ http://host.docker.internal:9090
# Example panels:
#  - rate(predict_requests_total[1m])
#  - histogram_quantile(0.95, sum(rate(predict_latency_seconds_bucket[5m])) by (le))
```

---

## API schema (validation)

`/predict` expects:

```json
{
  "sepal_length": float,  // 0 < value < 10
  "sepal_width":  float,  // 0 < value < 10
  "petal_length": float,  // 0 < value < 10
  "petal_width":  float   // 0 < value < 10
}
```

Out-of-range inputs return HTTP 422 with details.

Response:
```json
{
  "predicted_class": 0,
  "class_name": "setosa",
  "probabilities": {
    "setosa": 0.87, "versicolor": 0.12, "virginica": 0.01
  }
}
```

---

## CI/CD

- **ci.yml** runs on push/PR to `main`: installs deps → `flake8` → `pytest` → builds Docker image → pushes to Docker Hub.
- Requires repo secrets:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN` (Docker Hub access token with write scope)

**Note:** These secrets are only required for the GitHub Actions jobs that push images to Docker Hub; they are **not** needed to run the app locally.

- Image tags: `latest` and the commit `${{ github.sha }}`.  
- **deploy.sh** pulls `latest` and (re)runs the container.

---

## Retraining trigger (bonus)

Workflow **`retrain.yml`** runs when anything under `data/**` changes or via manual dispatch:

1) Runs `train.py` to produce a fresh `model/best_iris_model.pkl`.
2) Builds a new Docker image and pushes:
   - `…/iris-api:latest`
   - `…/iris-api:retrained-<commit-sha>`
3) Redeploy with:
```bash
export DOCKERHUB_USERNAME=<your-dockerhub-username>
./deploy.sh
```

---

## Environment variables

- `MODEL_PATH` — path to model file inside the container (default: `model/best_iris_model.pkl`)
- `LOG_DIR` — where to write logs (default: `logs`)
- `LOG_DB_PATH` — SQLite DB path (default: `logs/logs.db`)
- `PORT` — (deploy.sh) host port to map to container `80` (default: `8000`)

---

## Troubleshooting

- **Port 8000 already in use:** stop other processes or change `PORT` before running `deploy.sh`.
- **Docker “insufficient scopes”:** regenerate Docker Hub token with **write** access; update repo secret `DOCKERHUB_TOKEN`.
- **No logs in `logs/` when using Docker:** rerun `./deploy.sh` from the repo root (it mounts `$(pwd)/logs:/app/logs`).
- **CI flake8 errors:** run `flake8 .` locally; respect the config in `.flake8`.
- **MLflow UI empty:** make sure you ran `python train.py` and start the UI with `mlflow ui --backend-store-uri mlruns`.

---

## License

MIT
# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import time
import sqlite3
from datetime import datetime
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --- logging setup ---
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("iris_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"), maxBytes=1_000_000, backupCount=5
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# --- end logging setup ---

LOG_DB_PATH = os.getenv("LOG_DB_PATH", os.path.join(LOG_DIR, "logs.db"))


def init_db():
    with sqlite3.connect(LOG_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                sepal_length REAL NOT NULL,
                sepal_width REAL NOT NULL,
                petal_length REAL NOT NULL,
                petal_width REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                p_setosa REAL NOT NULL,
                p_versicolor REAL NOT NULL,
                p_virginica REAL NOT NULL,
                latency_ms REAL NOT NULL
            )
            """
        )


init_db()
logger.info("SQLite logging to %s", LOG_DB_PATH)


# --- prometheus metrics ---
REQUEST_COUNT = Counter("predict_requests_total", "Total predict requests")
REQUEST_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latency for /predict endpoint in seconds",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2],
)
# --- end metrics ---


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisResponse(BaseModel):
    predicted_class: int
    class_name: str
    probabilities: dict


app = FastAPI(title="Iris Classification API")

# Load the trained model
MODEL_PATH = os.getenv("MODEL_PATH", "model/best_iris_model.pkl")
model = joblib.load(MODEL_PATH)
logger.info("Model loaded from %s", MODEL_PATH)
target_names = ["setosa", "versicolor", "virginica"]


@app.post("/predict", response_model=IrisResponse)
def predict(request: IrisRequest):
    start = time.perf_counter()
    REQUEST_COUNT.inc()
    try:
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]])
        preds = model.predict(features)
        probs = model.predict_proba(features)[0]
        idx = int(preds[0])
        proba_dict = {target_names[i]: float(probs[i]) for i in range(len(target_names))}
        latency_ms = (time.perf_counter() - start) * 1000.0
        latency_sec = latency_ms / 1000.0
        REQUEST_LATENCY.observe(latency_sec)

        logger.info(
            "predict ok sepal_length=%.3f sepal_width=%.3f "
            "petal_length=%.3f petal_width=%.3f class=%s latency_ms=%.2f",
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
            target_names[idx],
            latency_ms,
        )
        logger.info("predict probs=%s", proba_dict)

        # persist to sqlite
        try:
            ts = datetime.utcnow().isoformat()
            with sqlite3.connect(LOG_DB_PATH) as conn:
                conn.execute(
                    """
                    INSERT INTO predictions
                    (ts, sepal_length, sepal_width, petal_length, petal_width,
                     predicted_class, class_name, p_setosa, p_versicolor, p_virginica, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        float(request.sepal_length),
                        float(request.sepal_width),
                        float(request.petal_length),
                        float(request.petal_width),
                        idx,
                        target_names[idx],
                        proba_dict["setosa"],
                        proba_dict["versicolor"],
                        proba_dict["virginica"],
                        latency_ms,
                    ),
                )
        except Exception as db_exc:
            logger.exception("sqlite insert failed: %s", str(db_exc))

        return IrisResponse(
            predicted_class=idx,
            class_name=target_names[idx],
            probabilities=proba_dict,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.exception("predict error latency_ms=%.2f error=%s", latency_ms, str(exc))
        raise


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

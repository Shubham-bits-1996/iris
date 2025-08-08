
import os
import tempfile
import joblib
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _prepare_temp_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = LogisticRegression(max_iter=200).fit(X, y)
    tmpdir = tempfile.mkdtemp()
    model_path = os.path.join(tmpdir, "temp_model.pkl")
    joblib.dump(clf, model_path)
    return model_path


def test_predict_endpoint():
    # Create a temporary model and point the app to it
    os.environ["MODEL_PATH"] = _prepare_temp_model()

    # Import after MODEL_PATH is set so app loads the right file
    from app.main import app

    client = TestClient(app)
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_class" in data
    assert "class_name" in data
    assert "probabilities" in data

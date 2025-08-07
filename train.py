import os
import mlflow
import joblib
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss



def load_data(test_size=0.2, random_state=42):
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_log(model, params, X_train, X_test, y_train, y_test, run_name):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        model.set_params(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        ll  = log_loss(y_test, proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("log_loss", ll)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Capture run_id before exiting the context
        run_id = run.info.run_id

    return acc, ll, run_id

def main():
    # Use local 'mlruns' directory
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Iris_Classification")

    X_train, X_test, y_train, y_test = load_data()

    experiments = {
        "LogisticRegression": {
            "model": LogisticRegression,
            "params": {"C": 1.0, "solver": "liblinear", "max_iter": 100}
        },
        "RandomForest": {
            "model": RandomForestClassifier,
            "params": {"n_estimators": 100, "max_depth": 4, "random_state": 42}
        }
    }

    best_acc = -1
    best_run_id = None
    best_model_name = None

    for name, spec in experiments.items():
        acc, ll, run_id = train_and_log(
            spec["model"](),
            spec["params"],
            X_train, X_test, y_train, y_test,
            run_name=name
        )
        print(f"{name} → accuracy={acc:.4f}, log_loss={ll:.4f}")
        if acc > best_acc:
            best_acc, best_run_id, best_model_name = acc, run_id, name

    # Register best model in the model registry
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, name="Best_Iris_Model")
    print(f"Registered best model ({best_model_name}) with accuracy={best_acc:.4f}")


    best_model = mlflow.sklearn.load_model(model_uri)
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/best_iris_model.pkl")
    print("Exported model to model/best_iris_model.pkl")


if __name__ == "__main__":
    main()
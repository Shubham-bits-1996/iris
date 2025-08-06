import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, train_test_split


def train_model():
    # Set an experiment name to group the runs
    mlflow.set_experiment("Iris Classification Experiment")

    # Load data and split into training and testing sets for evaluation
    iris_bunch = load_iris()
    X = pd.DataFrame(iris_bunch.data, columns=iris_bunch.feature_names)
    y = pd.Series(iris_bunch.target, name='species')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start a parent run to encapsulate the entire training process
    with mlflow.start_run(run_name="Model Comparison Run") as parent_run:
        mlflow.log_param("data_split_random_state", 42)
        mlflow.log_param("test_size", 0.2)
        print(f"Parent Run ID: {parent_run.info.run_id}")

        # --- Grid Search for Logistic Regression ---
        best_lr_accuracy = -1
        best_lr_model_info = None
        print("\n--- Starting Logistic Regression Trials ---")
        with mlflow.start_run(run_name="LogisticRegression_Trials", nested=True):
            lr_param_grid = {
                "max_iter": [1000],  # Increased for 'saga' solver
                "solver": ["saga"],
                "penalty": ["l1", "l2"],
                "C": [0.5, 1.0, 1.5],
                "random_state": [42],
            }

            for i, params in enumerate(ParameterGrid(lr_param_grid)):
                with mlflow.start_run(run_name=f"LR_run_{i}", nested=True) as lr_run:
                    mlflow.log_params(params)

                    lr_model = LogisticRegression(**params)
                    lr_model.fit(X_train, y_train)

                    y_pred_lr = lr_model.predict(X_test)
                    lr_accuracy = accuracy_score(y_test, y_pred_lr)
                    mlflow.log_metric("accuracy", lr_accuracy)
                    print(f"LR Trial {i} Accuracy: {lr_accuracy:.2f} with params {params}")

                    lr_model_info = mlflow.sklearn.log_model(
                        sk_model=lr_model,
                        name="logistic_regression_model"
                    )

                    if lr_accuracy > best_lr_accuracy:
                        best_lr_accuracy = lr_accuracy
                        best_lr_model_info = lr_model_info.model_uri  # Only store the URI (string)

            mlflow.log_metric("best_accuracy", best_lr_accuracy)
            print(f"Best Logistic Regression Accuracy: {best_lr_accuracy:.2f}")

        # --- Grid Search for Random Forest ---
        best_rf_accuracy = -1
        best_rf_model_info = None
        print("\n--- Starting Random Forest Trials ---")
        with mlflow.start_run(run_name="RandomForest_Trials", nested=True):
            rf_param_grid = {
                "n_estimators": [100, 150],
                "max_depth": [5, 10],
                "min_samples_leaf": [2, 4],
                "criterion": ["gini", "entropy"],
                "random_state": [42],
            }

            for i, params in enumerate(ParameterGrid(rf_param_grid)):
                with mlflow.start_run(run_name=f"RF_run_{i}", nested=True):
                    mlflow.log_params(params)

                    rf_model = RandomForestClassifier(**params)
                    rf_model.fit(X_train, y_train)

                    y_pred_rf = rf_model.predict(X_test)
                    rf_accuracy = accuracy_score(y_test, y_pred_rf)
                    mlflow.log_metric("accuracy", rf_accuracy)
                    print(f"RF Trial {i} Accuracy: {rf_accuracy:.2f} with params {params}")

                    rf_model_info = mlflow.sklearn.log_model(
                        sk_model=rf_model,
                        name="random_forest_model"
                    )

                    if rf_accuracy > best_rf_accuracy:
                        best_rf_accuracy = rf_accuracy
                        best_rf_model_info = rf_model_info.model_uri  # Only store the URI (string)

            mlflow.log_metric("best_accuracy", best_rf_accuracy)
            print(f"Best Random Forest Accuracy: {best_rf_accuracy:.2f}")

        

        # --- Select and Register the Best Model from all trials ---
        print("\n--- Selecting and Registering Best Model ---")
        if best_lr_accuracy > best_rf_accuracy:
            best_model_uri = best_lr_model_info  # Already a string
            best_model_name = "Logistic Regression"
            best_model_accuracy = best_lr_accuracy
        else:
            best_model_uri = best_rf_model_info  # Already a string
            best_model_name = "Random Forest"
            best_model_accuracy = best_rf_accuracy

        print(f"Overall Best Model: '{best_model_name}' with accuracy: {best_model_accuracy:.2f}")
        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_metric("best_model_accuracy", best_model_accuracy)

        # Register the best model
        registered_model_name = "IrisClassifier"
        print(f"Registering the best model as '{registered_model_name}'")
        registered_model = mlflow.register_model(
            model_uri=best_model_uri,
            name=registered_model_name
        )
        print(f"Model registered: {registered_model.name}, version: {registered_model.version}")

        print(type(registered_model_name), type(registered_model.version), type(best_model_name), type(parent_run.info.run_id))
        print(type(f"The best model ({best_model_name}) from run {parent_run.info.run_id} after grid search."))

        # Add a description to the registered model version for clarity
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name=registered_model_name,
            version=registered_model.version,
            # description=str(f"The best model ({best_model_name}) from run {parent_run.info.run_id} after grid search.")
            description="Best model"
        )
        
        

if __name__ == "__main__":
    train_model()
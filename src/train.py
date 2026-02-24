import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from data_preprocessing import load_data, preprocess_data


def build_models():
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="logloss",
            random_state=42
        )
    }
    return models


def train_and_evaluate(X_train, y_train, preprocessor):
    models = build_models()
    results = {}

    for name, model in models.items():

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc"
        )

        mean_score = np.mean(scores)
        results[name] = mean_score

        print(f"{name} ROC-AUC: {mean_score:.4f}")

    return results


def train_best_model(X_train, y_train, preprocessor, results):
    best_model_name = max(results, key=results.get)
    print(f"\nBest model: {best_model_name}")

    models = build_models()
    best_model = models[best_model_name]

    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])

    final_pipeline.fit(X_train, y_train)

    joblib.dump(final_pipeline, f"telco-customer-churn/models/{best_model_name}.pkl")

    print(f"Saved best model to models/{best_model_name}.pkl")


def main(data_path):

    print("Loading data...")
    df = load_data(data_path)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("Training models with cross-validation...")
    results = train_and_evaluate(X_train, y_train, preprocessor)

    print("Training final best model...")
    train_best_model(X_train, y_train, preprocessor, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="telco-customer-churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to raw dataset"
    )

    args = parser.parse_args()
    main(args.data_path)
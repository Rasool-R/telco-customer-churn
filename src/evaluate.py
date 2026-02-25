import argparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from data_preprocessing import load_data, preprocess_data


def evaluate_model(model_path, data_path):

    print("Loading data...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    print("Loading trained model...")
    model = joblib.load(model_path)

    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)


def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(y_true, y_prob):

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        # required=True,
        default="telco-customer-churn/models/xgboost.pkl",
        help="Path to saved model (.pkl file)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="telco-customer-churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to raw dataset"
    )

    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_path)
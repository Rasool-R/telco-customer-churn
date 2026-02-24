import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from given path.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning:
    - Convert TotalCharges to numeric
    - Drop missing values
    - Drop customerID (not predictive)
    """

    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert SeniorCitizen to categorical
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Drop missing values
    df.dropna(inplace=True)

    # Drop non-informative column
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate features and target.
    """

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create preprocessing pipeline:
    - Scale numeric features
    - One-hot encode categorical features
    """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2):

    df = clean_data(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor
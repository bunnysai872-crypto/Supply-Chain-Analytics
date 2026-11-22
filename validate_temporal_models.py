
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


print("=" * 65)
print("SUPPLY CHAIN - TIME-BASED MODEL VALIDATION")
print("=" * 65)

# 1. Load datasets
cleaned_path = "data/cleaned_supply_chain.csv"
ml_path = "data/ml_training_data.csv"

cleaned = pd.read_csv(cleaned_path)
ml_data = pd.read_csv(ml_path)

print("\nDatasets loaded successfully!")
print("Cleaned dataset:", cleaned.shape)
print("ML dataset:", ml_data.shape)

# 2. Prepare order dates and correct order-level target
cleaned["Order Id"] = pd.to_numeric(
    cleaned["Order Id"], errors="coerce"
)

cleaned["Delivery Delay Days"] = pd.to_numeric(
    cleaned["Delivery Delay Days"], errors="coerce"
)

cleaned["order date (DateOrders)"] = pd.to_datetime(
    cleaned["order date (DateOrders)"], errors="coerce"
)

order_info = cleaned.groupby("Order Id").agg(
    Order_Date=("order date (DateOrders)", "min"),
    Median_Delay=("Delivery Delay Days", "median")
).reset_index()

order_info["Target"] = (
    order_info["Median_Delay"] > 0
).astype(int)

ml_data["Order Id"] = pd.to_numeric(
    ml_data["Order Id"], errors="coerce"
)

# Remove the old target and use the order-level target
ml_data = ml_data.drop(columns=["Is_Delayed"], errors="ignore")

data = ml_data.merge(
    order_info[["Order Id", "Order_Date", "Target"]],
    on="Order Id",
    how="inner"
)

data = data.dropna(subset=["Order_Date", "Target"])
data = data.sort_values("Order_Date").reset_index(drop=True)

print("\nOrder-level dataset:", data.shape)
print("Not delayed:", (data["Target"] == 0).sum())
print("Delayed:", (data["Target"] == 1).sum())

# 3. Remove fields that should not be used as predictors
# Benefit per order is excluded because it can depend on
# the completed transaction outcome.
X = data.drop(
    columns=[
        "Order Id",
        "Order_Date",
        "Target",
        "Benefit per order"
    ],
    errors="ignore"
)

y = data["Target"]

# 4. Split chronologically: first 80% for training,
# final 20% for testing
split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Training period:",
      data["Order_Date"].iloc[0],
      "to",
      data["Order_Date"].iloc[split_index - 1])
print("Testing period:",
      data["Order_Date"].iloc[split_index],
      "to",
      data["Order_Date"].iloc[-1])

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Training or testing data contains only one target class. "
        "Check the date split and target distribution."
    )

# 5. Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# 6. Preprocessing
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10,
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10,
        sparse=True
    )

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 7. Train Random Forest
print("\nTraining Random Forest...")

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_predictions = rf_pipeline.predict(X_test)
rf_probabilities = rf_pipeline.predict_proba(X_test)[:, 1]

# 8. Train XGBoost
print("\nTraining XGBoost...")

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
class_weight_ratio = negative_count / positive_count

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weight_ratio,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    ))
])

xgb_pipeline.fit(X_train, y_train)

xgb_predictions = xgb_pipeline.predict(X_test)
xgb_probabilities = xgb_pipeline.predict_proba(X_test)[:, 1]

# 9. Evaluate both models
def evaluate_model(name, actual, predictions, probabilities):
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predictions),
        "Precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "Recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            actual, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(actual, probabilities)
    }


results = [
    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions,
        rf_probabilities
    ),
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_predictions,
        xgb_probabilities
    )
]

comparison = pd.DataFrame(results)

print("\n" + "=" * 65)
print("TIME-BASED MODEL RESULTS")
print("=" * 65)

print(comparison.to_string(index=False))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_predictions))

# 10. Save results and models
os.makedirs("data", exist_ok=True)

comparison.to_csv(
    "data/temporal_model_comparison.csv",
    index=False
)

rf_output = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": rf_predictions,
    "Delay_Probability": rf_probabilities
})

xgb_output = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": xgb_predictions,
    "Delay_Probability": xgb_probabilities
})

rf_output.to_csv(
    "data/temporal_predictions_random_forest.csv",
    index=False
)

xgb_output.to_csv(
    "data/temporal_predictions_xgboost.csv",
    index=False
)

joblib.dump(
    rf_pipeline,
    "data/random_forest_temporal.pkl"
)

joblib.dump(
    xgb_pipeline,
    "data/xgboost_temporal.pkl"
)

print("\nAll results and models saved successfully!")
print("Time-based validation completed!")

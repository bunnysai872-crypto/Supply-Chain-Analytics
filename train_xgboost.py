
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

print("=" * 65)
print("SUPPLY CHAIN - XGBOOST CLASSIFICATION")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/ml_training_data.csv")

print("\nDataset loaded:", df.shape)

# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["Is_Delayed", "Order Id"])
y = df["Is_Delayed"].astype(int)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# --------------------------------------------------
# 3. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. DATA PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 5. XGBOOST CLASSIFIER
# --------------------------------------------------

xgb_classifier = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_classifier)
])

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

print("XGBoost training completed!")

# --------------------------------------------------
# 7. PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 65)
print("XGBOOST MODEL RESULTS")
print("=" * 65)

print(f"\nAccuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Delayed", "Delayed"],
    zero_division=0
))

# --------------------------------------------------
# 9. SAVE XGBOOST MODEL
# --------------------------------------------------

Path("data").mkdir(exist_ok=True)

joblib.dump(
    model,
    "data/xgboost_model.pkl"
)

print("\nXGBoost model saved successfully!")

# --------------------------------------------------
# 10. SAVE PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "Actual_Delivery_Status": y_test.values,
    "Predicted_Delivery_Status": y_pred,
    "Delay_Probability": y_probability
})

results["Actual_Delivery_Status"] = (
    results["Actual_Delivery_Status"]
    .map({0: "Not Delayed", 1: "Delayed"})
)

results["Predicted_Delivery_Status"] = (
    results["Predicted_Delivery_Status"]
    .map({0: "Not Delayed", 1: "Delayed"})
)

results.to_csv(
    "data/xgboost_predictions.csv",
    index=False
)

print("XGBoost predictions saved successfully!")

# --------------------------------------------------
# 11. COMPARE WITH RANDOM FOREST
# --------------------------------------------------

rf_path = Path("data/random_forest_predictions.csv")

if rf_path.exists():

    rf_results = pd.read_csv(rf_path)

    actual_rf = rf_results[
        "Actual_Delivery_Status"
    ].map({
        "Not Delayed": 0,
        "Delayed": 1
    })

    predicted_rf = rf_results[
        "Predicted_Delivery_Status"
    ].map({
        "Not Delayed": 0,
        "Delayed": 1
    })

    rf_probability = rf_results["Delay_Probability"]

    comparison = pd.DataFrame([
        {
            "Model": "Random Forest",
            "Accuracy": accuracy_score(actual_rf, predicted_rf),
            "Precision": precision_score(
                actual_rf, predicted_rf, zero_division=0
            ),
            "Recall": recall_score(
                actual_rf, predicted_rf, zero_division=0
            ),
            "F1 Score": f1_score(
                actual_rf, predicted_rf, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                actual_rf, rf_probability
            )
        },
        {
            "Model": "XGBoost",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
    ])

    comparison.to_csv(
        "data/model_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print(comparison.round(4).to_string(index=False))

    print("\nComparison saved: data/model_comparison.csv")

else:
    print("\nRandom Forest predictions not found.")
    print("Run train_model.py to generate the comparison.")

print("\n" + "=" * 65)
print("XGBOOST TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)

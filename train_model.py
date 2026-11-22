
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

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
    confusion_matrix,
    classification_report
)

print("=" * 65)
print("SUPPLY CHAIN - RANDOM FOREST CLASSIFICATION")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

data_path = Path("data/ml_training_data.csv")

if not data_path.exists():
    raise FileNotFoundError(
        "ML dataset not found. Run prepare_ml_data.py first."
    )

df = pd.read_csv(data_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# --------------------------------------------------

target = "Is_Delayed"

# Remove target and identifier from model inputs
X = df.drop(columns=["Is_Delayed", "Order Id"])
y = df[target].astype(int)

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

print("\nTarget distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
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
# 4. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

# --------------------------------------------------
# 5. RANDOM FOREST MODEL
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ]
)

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# 7. PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 8. EVALUATION
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
print("RANDOM FOREST MODEL RESULTS")
print("=" * 65)

print(f"\nAccuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Delayed", "Delayed"],
        zero_division=0
    )
)

# --------------------------------------------------
# 9. SAVE MODEL
# --------------------------------------------------

model_path = Path("data/random_forest_model.pkl")

joblib.dump(model, model_path)

print("\nModel saved successfully:", model_path)

# --------------------------------------------------
# 10. SAVE TEST PREDICTIONS
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
    "data/random_forest_predictions.csv",
    index=False
)

print("Predictions saved successfully!")

print("\n" + "=" * 65)
print("RANDOM FOREST TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)


import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

print("=" * 60)
print("SUPPLY CHAIN - FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Load trained Random Forest model
model = joblib.load("data/random_forest_temporal.pkl")

# Get preprocessing and trained classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Get feature importance values
importance_values = classifier.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Save complete feature importance
os.makedirs("data", exist_ok=True)

importance_df.to_csv(
    "data/feature_importance.csv",
    index=False
)

# Display top 15 features
top_features = importance_df.head(15)

print("\nTOP 15 IMPORTANT FEATURES")
print(top_features.to_string(index=False))

# Create visualization
plt.figure(figsize=(12, 7))

plt.barh(
    top_features["Feature"].iloc[::-1],
    top_features["Importance"].iloc[::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 15 Factors Influencing Delivery Delay Predictions")
plt.tight_layout()

plt.savefig(
    "data/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance analysis completed!")
print("Saved: data/feature_importance.csv")
print("Saved: data/feature_importance.png")

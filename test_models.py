
import joblib
from pathlib import Path

model_files = [
    "data/random_forest_temporal.pkl",
    "data/xgboost_temporal.pkl"
]

print("=" * 55)
print("SUPPLY CHAIN - MODEL LOADING TEST")
print("=" * 55)

for file in model_files:
    path = Path(file)

    try:
        model = joblib.load(path)
        print(f"\nSUCCESS: {path.name}")
        print(f"Model type: {type(model).__name__}")
    except Exception as e:
        print(f"\nFAILED: {path.name}")
        print(f"Error: {e}")

print("\nModel loading test completed.")

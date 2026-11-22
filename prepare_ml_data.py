
import pandas as pd
from pathlib import Path

print("=" * 60)
print("SUPPLY CHAIN - ML DATA PREPARATION")
print("=" * 60)

# 1. Load cleaned dataset
file_path = Path("data/cleaned_supply_chain.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "cleaned_supply_chain.csv not found inside data folder."
    )

df = pd.read_csv(file_path)

print("\nOriginal dataset shape:", df.shape)

# 2. Check required columns
required_columns = [
    "Order Id",
    "Delivery Delay Days",
    "order date (DateOrders)"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column missing: {column}")

# 3. Create actual delivery outcome
df["Delivery Delay Days"] = pd.to_numeric(
    df["Delivery Delay Days"],
    errors="coerce"
)

df = df.dropna(
    subset=["Order Id", "Delivery Delay Days"]
).copy()

# Target: 1 = Delayed, 0 = Not delayed
df["Is_Delayed"] = (
    df["Delivery Delay Days"] > 0
).astype(int)

# 4. Select features available before delivery
candidate_features = [
    "Type",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Category Id",
    "Category Name",
    "Customer Segment",
    "Customer State",
    "Customer Country",
    "Market",
    "Order City",
    "Order Country",
    "Order Region",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Quantity",
    "Product Price",
    "Shipping Mode"
]

features = [
    col for col in candidate_features
    if col in df.columns
]

if len(features) < 3:
    raise ValueError(
        "Not enough valid feature columns were found."
    )

print("\nSelected features:")
for feature in features:
    print("-", feature)

# 5. Aggregate item records into one order record
aggregation = {}

for col in features:

    if pd.api.types.is_numeric_dtype(df[col]):
        aggregation[col] = "mean"
    else:
        aggregation[col] = "first"

# Quantity represents the total quantity in an order
if "Order Item Quantity" in features:
    aggregation["Order Item Quantity"] = "sum"

# Aggregate target and order date
aggregation["Is_Delayed"] = "max"
aggregation["order date (DateOrders)"] = "first"

ml_df = (
    df.groupby("Order Id")
    .agg(aggregation)
    .reset_index()
)

# 6. Create date-based features
ml_df["order date (DateOrders)"] = pd.to_datetime(
    ml_df["order date (DateOrders)"],
    errors="coerce"
)

ml_df["Order Month"] = (
    ml_df["order date (DateOrders)"].dt.month
)

ml_df["Order DayOfWeek"] = (
    ml_df["order date (DateOrders)"].dt.dayofweek
)

ml_df["Order Year"] = (
    ml_df["order date (DateOrders)"].dt.year
)

# Remove raw date column
ml_df = ml_df.drop(
    columns=["order date (DateOrders)"]
)

# 7. Save prepared dataset
output_path = Path("data/ml_training_data.csv")

ml_df.to_csv(output_path, index=False)

# 8. Display results
print("\n" + "=" * 60)
print("ML DATASET PREPARATION COMPLETED")
print("=" * 60)

print("\nPrepared dataset shape:", ml_df.shape)

print("\nTarget distribution:")
print(
    ml_df["Is_Delayed"]
    .value_counts()
    .rename(index={
        0: "Not Delayed",
        1: "Delayed"
    })
)

print("\nTarget percentages:")
print(
    (ml_df["Is_Delayed"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nMissing values:")
print(ml_df.isnull().sum())

print("\nSaved successfully:", output_path)
print("\nFirst five records:")
print(ml_df.head())

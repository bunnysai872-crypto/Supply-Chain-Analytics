
import pandas as pd
import os

print("=" * 60)
print("SUPPLY CHAIN DATA CLEANING")
print("=" * 60)

# 1. Load original dataset
input_path = "data/DataCoSupplyChainDataset.csv"

df = pd.read_csv(input_path, encoding="latin-1")

print("\nOriginal Dataset Shape:", df.shape)

# 2. Remove completely empty columns
df = df.dropna(axis=1, how="all")

# 3. Remove unnecessary personal information
unnecessary_columns = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Product Image",
    "Product Description"
]

df = df.drop(
    columns=unnecessary_columns,
    errors="ignore"
)

# 4. Handle remaining missing values
for column in df.select_dtypes(
    include=["object", "str"]
).columns:
    df[column] = df[column].fillna("Unknown")

# 5. Remove exact duplicate rows
duplicates = df.duplicated().sum()

print("\nDuplicate Rows Found:", duplicates)

df = df.drop_duplicates()

# 6. Convert date columns
date_columns = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

# 7. Create delivery delay column
df["Delivery Delay Days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

# 8. Create delivery performance category
df["Delivery Performance"] = df[
    "Delivery Delay Days"
].apply(
    lambda x: "Delayed" if x > 0
    else "On Time" if x == 0
    else "Early"
)

# 9. Create order month column
df["Order Month"] = df[
    "order date (DateOrders)"
].dt.month

df["Order Year"] = df[
    "order date (DateOrders)"
].dt.year

# 10. Save cleaned dataset
os.makedirs("data", exist_ok=True)

output_path = "data/cleaned_supply_chain.csv"

df.to_csv(output_path, index=False)

# 11. Display cleaning results
print("\n========== CLEANING COMPLETED ==========")

print("\nCleaned Dataset Shape:", df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

print("\nDelivery Performance:")
print(df["Delivery Performance"].value_counts())

print("\nCleaned Dataset Preview:")
print(df.head())

print("\nCleaned dataset saved at:", output_path)

print("\n========================================")

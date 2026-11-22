
import pandas as pd
import plotly.express as px

print("=" * 60)
print("SUPPLY CHAIN EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_supply_chain.csv")

# Convert dates
df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

# ==========================================
# 1. GENERAL BUSINESS PERFORMANCE
# ==========================================

print("\n1. GENERAL BUSINESS PERFORMANCE")

total_orders = df["Order Id"].nunique()
total_sales = df["Sales"].sum()
total_profit = df["Order Profit Per Order"].sum()
total_quantity = df["Order Item Quantity"].sum()

print("Total Orders:", total_orders)
print("Total Sales:", round(total_sales, 2))
print("Total Profit:", round(total_profit, 2))
print("Total Items Sold:", total_quantity)

# ==========================================
# 2. DELIVERY PERFORMANCE
# ==========================================

print("\n2. DELIVERY PERFORMANCE")

delivery_counts = df[
    "Delivery Performance"
].value_counts()

print(delivery_counts)

delivery_percentage = (
    delivery_counts / len(df) * 100
)

print("\nDelivery Performance Percentage:")
print(delivery_percentage.round(2))

# ==========================================
# 3. SHIPPING MODE ANALYSIS
# ==========================================

print("\n3. SHIPPING MODE ANALYSIS")

shipping_analysis = df.groupby(
    "Shipping Mode"
).agg(
    Total_Order_Items=("Order Id", "count"),
    Average_Delivery_Days=(
        "Days for shipping (real)", "mean"
    ),
    Average_Delay=(
        "Delivery Delay Days", "mean"
    )
).round(2)

print(shipping_analysis)

# ==========================================
# 4. REGIONAL ANALYSIS
# ==========================================

print("\n4. REGIONAL DELIVERY ANALYSIS")

region_analysis = df.groupby(
    "Order Region"
).agg(
    Total_Order_Items=("Order Id", "count"),
    Average_Delay=("Delivery Delay Days", "mean"),
    Total_Sales=("Sales", "sum")
).round(2)

print(region_analysis.sort_values(
    by="Average_Delay",
    ascending=False
))

# ==========================================
# 5. MONTHLY SALES ANALYSIS
# ==========================================

print("\n5. MONTHLY SALES ANALYSIS")

monthly_sales = df.groupby(
    ["Order Year", "Order Month"]
)["Sales"].sum().reset_index()

print(monthly_sales.head(15))

# ==========================================
# 6. PRODUCT CATEGORY ANALYSIS
# ==========================================

print("\n6. PRODUCT CATEGORY ANALYSIS")

category_analysis = df.groupby(
    "Category Name"
).agg(
    Total_Sales=("Sales", "sum"),
    Total_Quantity=("Order Item Quantity", "sum"),
    Average_Profit=("Order Profit Per Order", "mean")
).round(2)

print(category_analysis.sort_values(
    by="Total_Sales",
    ascending=False
).head(10))

# ==========================================
# 7. SHIPPING COST ANALYSIS
# ==========================================

print("\n7. SHIPPING COST ANALYSIS")

cost_analysis = df.groupby(
    "Shipping Mode"
).agg(
    Total_Sales=("Sales", "sum"),
    Total_Discount=("Order Item Discount", "sum"),
    Average_Product_Price=(
        "Order Item Product Price", "mean"
    )
).round(2)

print(cost_analysis)

# ==========================================
# 8. SAVE ANALYTICAL OUTPUTS
# ==========================================

monthly_sales.to_csv(
    "data/monthly_sales_analysis.csv",
    index=False
)

region_analysis.to_csv(
    "data/regional_analysis.csv"
)

category_analysis.to_csv(
    "data/category_analysis.csv"
)

shipping_analysis.to_csv(
    "data/shipping_analysis.csv"
)

print("\nAll analytical reports saved successfully!")

print("\nEDA COMPLETED!")


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL UI DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F6FA;
}

[data-testid="stSidebar"] {
    background: #101C35;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* Sidebar dropdown styling */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #263650 !important;
    border: 1px solid #465875 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] input {
    color: #FFFFFF !important;
}

/* Dropdown options */
[data-baseweb="popover"] [role="option"] {
    color: #17233B !important;
    background-color: #FFFFFF !important;
}

.main-title {
    font-size: 29px;
    font-weight: 800;
    color: #17233B;
    letter-spacing: -1px;
}

.subtitle {
    color: #7B8497;
    font-size: 13px;
    margin-top: -8px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #17233B;
    margin-top: 15px;
    margin-bottom: 12px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8ECF3;
    border-radius: 12px;
    padding: 19px;
    min-height: 125px;
    box-shadow: 0 2px 8px rgba(20,40,80,0.03);
}

.metric-label {
    color: #7C879B;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    color: #17233B;
    font-size: 27px;
    font-weight: 800;
    margin-top: 12px;
}

.metric-note {
    color: #8792A5;
    font-size: 11px;
    margin-top: 5px;
}

div[data-testid="stPlotlyChart"] {
    background: white;
    border-radius: 12px;
    padding: 8px;
    border: 1px solid #E8ECF3;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E8ECF3;
    border-radius: 10px;
}

hr {
    border-color: #E5EAF2;
}

/* Improve main dashboard text contrast */
.main .block-container h1,
.main .block-container h2,
.main .block-container h3 {
    color: #17233B !important;
}

.main .block-container p,
.main .block-container .stCaption {
    color: #667085;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data

def load_data(uploaded_file=None):

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    else:
        file_path = Path("data/cleaned_supply_chain.csv")

        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)

    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"],
        errors="coerce"
    )

    df["Delivery Delay Days"] = pd.to_numeric(
        df["Delivery Delay Days"], errors="coerce"
    )

    return df


# Dataset upload option
uploaded_file = st.sidebar.file_uploader(
    "Upload Supply Chain Dataset",
    type=["csv"]
)

df = load_data(uploaded_file)

if df is None:
    st.info("Please upload your cleaned_supply_chain.csv file using the sidebar.")
    st.stop()

if df.empty:
    st.error("Your cleaned dataset is empty.")
    st.stop()



# --------------------------------------------------
# CREATE ORDER-LEVEL DATA
# --------------------------------------------------

# The original dataset contains order-item records.
# This creates one delivery record per unique Order Id.

order_df = (
    df.groupby("Order Id", as_index=False)
    .agg({
        "Days for shipping (real)": "median",
        "Days for shipment (scheduled)": "median",
        "Delivery Delay Days": "median",
        "Order Region": "first",
        "Market": "first",
        "Shipping Mode": "first",
        "order date (DateOrders)": "first"
    })
)

order_df["Order Delivery Status"] = order_df[
    "Delivery Delay Days"
].apply(
    lambda x: "Delayed" if x > 0
    else ("Early" if x < 0 else "On Time")
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## ◈ SUPPLY CHAIN")
    st.caption("INTELLIGENCE PLATFORM")

    st.markdown("---")

    page = st.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Delivery Intelligence",
        "Delivery Risk Analysis",
        "Regional Analytics",
        "Revenue Analytics",
        "Product Analytics"
    ]
)

    st.markdown("---")

    st.markdown("### DATA FILTERS")

    markets = ["All"] + sorted(
        df["Market"].dropna().astype(str).unique().tolist()
    )

    selected_market = st.selectbox("Market", markets)

    shipping_modes = ["All"] + sorted(
        df["Shipping Mode"].dropna().astype(str).unique().tolist()
    )

    selected_shipping = st.selectbox(
        "Shipping Mode", shipping_modes
    )

    regions = ["All"] + sorted(
        df["Order Region"].dropna().astype(str).unique().tolist()
    )

    selected_region = st.selectbox("Region", regions)

    min_date = df["order date (DateOrders)"].min()
    max_date = df["order date (DateOrders)"].max()

    if pd.isna(min_date) or pd.isna(max_date):
        start_date, end_date = None, None
    else:
        selected_dates = st.date_input(
            "Order Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date, end_date = min_date.date(), max_date.date()

    st.markdown("---")
    st.caption("SUPPLY CHAIN ANALYTICS | 2026")


# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

filtered_df = df.copy()
filtered_orders = order_df.copy()

if selected_market != "All":
    filtered_df = filtered_df[
        filtered_df["Market"].astype(str) == selected_market
    ]
    filtered_orders = filtered_orders[
        filtered_orders["Market"].astype(str) == selected_market
    ]

if selected_shipping != "All":
    filtered_df = filtered_df[
        filtered_df["Shipping Mode"].astype(str) == selected_shipping
    ]
    filtered_orders = filtered_orders[
        filtered_orders["Shipping Mode"].astype(str) == selected_shipping
    ]

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["Order Region"].astype(str) == selected_region
    ]
    filtered_orders = filtered_orders[
        filtered_orders["Order Region"].astype(str) == selected_region
    ]

if start_date and end_date:
    filtered_df = filtered_df[
        filtered_df["order date (DateOrders)"].dt.date.between(
            start_date, end_date
        )
    ]

    filtered_orders = filtered_orders[
        filtered_orders["order date (DateOrders)"].dt.date.between(
            start_date, end_date
        )
    ]


# --------------------------------------------------
# COMMON METRICS
# --------------------------------------------------

total_orders = filtered_orders["Order Id"].nunique()
total_sales = filtered_df["Sales"].sum()
total_items = filtered_df["Order Item Quantity"].sum()

avg_delivery = filtered_orders[
    "Days for shipping (real)"
].mean()

on_time_rate = (
    filtered_orders["Order Delivery Status"].eq("On Time").mean() * 100
    if len(filtered_orders) else 0
)

delayed_rate = (
    filtered_orders["Order Delivery Status"].eq("Delayed").mean() * 100
    if len(filtered_orders) else 0
)


# --------------------------------------------------
# REUSABLE FUNCTIONS
# --------------------------------------------------

def metric_card(label, value, note, icon):

    st.markdown(f"""
    <div class="metric-card">
        <div style="display:flex;justify-content:space-between;">
            <span class="metric-label">{label}</span>
            <span style="font-size:21px;">{icon}</span>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, description=""):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if description:
        st.caption(description)


def chart_style(fig, height=350):

    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=35, r=25, t=45, b=45),

        font=dict(
            family="Inter",
            size=12,
            color="#344054"
        ),

        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        legend=dict(
            font=dict(color="#344054", size=12),
            bgcolor="rgba(255,255,255,0)"
        ),

        xaxis=dict(
            title_font=dict(color="#344054"),
            tickfont=dict(color="#667085"),
            showgrid=False,
            zeroline=False
        ),

        yaxis=dict(
            title_font=dict(color="#344054"),
            tickfont=dict(color="#667085"),
            gridcolor="#E9EDF4",
            zeroline=False
        ),

        coloraxis_colorbar=dict(
            tickfont=dict(color="#344054"),
            title_font=dict(color="#344054")
        )
    )

    return fig


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">SUPPLY CHAIN INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enterprise Operations & Performance Analytics Platform</div>',
    unsafe_allow_html=True
)

st.write("")

st.caption(
    f"Showing {total_orders:,} unique orders | "
    f"{len(filtered_df):,} order-item records"
)

if filtered_df.empty or filtered_orders.empty:
    st.warning("No records match the selected filters. Change your filters.")
    st.stop()


# ==================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==================================================

if page == "Executive Overview":

    st.markdown("## Executive Overview")
    st.caption("High-level business performance and operational intelligence.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Unique Orders",
            f"{total_orders:,}",
            "Distinct order IDs",
            "📦"
        )

    with c2:
        metric_card(
            "Total Line-Item Sales",
            f"{total_sales:,.0f}",
            "Sum of sales across item records",
            "💰"
        )

    with c3:
        metric_card(
            "On-Time Delivery",
            f"{on_time_rate:.1f}%",
            "Share of unique orders",
            "🚚"
        )

    with c4:
        metric_card(
            "Average Delivery",
            f"{avg_delivery:.2f} days",
            "Actual shipping duration",
            "⏱️"
        )

    st.write("")

    left, right = st.columns([1.6, 1])

    with left:

        section_header("Monthly Sales Trend")

        monthly = (
            filtered_df.dropna(subset=["order date (DateOrders)"])
            .set_index("order date (DateOrders)")
            .resample("ME")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.area(
            monthly,
            x="order date (DateOrders)",
            y="Sales",
            markers=True,
            color_discrete_sequence=["#3979D6"]
        )

        fig.update_traces(
            fill="tozeroy",
            line=dict(width=3)
        )

        st.plotly_chart(
            chart_style(fig, 370),
            use_container_width=True
        )

    with right:

        section_header("Delivery Status Distribution")

        delivery_counts = (
            filtered_orders["Order Delivery Status"]
            .value_counts()
            .reset_index()
        )

        delivery_counts.columns = ["Status", "Orders"]

        fig = px.pie(
            delivery_counts,
            names="Status",
            values="Orders",
            hole=0.65,
            color="Status",
            color_discrete_map={
                "On Time": "#24A47A",
                "Delayed": "#E45C68",
                "Early": "#4387D7"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent"
        )

        st.plotly_chart(
            chart_style(fig, 370),
            use_container_width=True
        )

    left, right = st.columns(2)

    with left:

        section_header("Shipping Mode Performance")

        shipping = (
            filtered_orders.groupby("Shipping Mode", as_index=False)
            ["Days for shipping (real)"].mean()
            .sort_values("Days for shipping (real)")
        )

        fig = px.bar(
            shipping,
            x="Shipping Mode",
            y="Days for shipping (real)",
            color="Days for shipping (real)",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with right:

        section_header("Regional Sales Contribution")

        region_sales = (
            filtered_df.groupby("Order Region", as_index=False)
            ["Sales"].sum()
            .sort_values("Sales", ascending=False)
            .head(10)
        )

        fig = px.bar(
            region_sales,
            x="Sales",
            y="Order Region",
            orientation="h",
            color="Sales",
            color_continuous_scale="Blues"
        )

        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )



# ==================================================
# BUSINESS INSIGHTS
# ==================================================

    st.markdown("---")

    st.markdown("## Business Intelligence & Insights")

    st.caption(
        "Automatically generated observations based on the selected filters."
    )

    # Best performing region by sales
    region_performance = (
        filtered_df.groupby("Order Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    # Shipping mode with highest average delay
    shipping_performance = (
        filtered_orders.groupby("Shipping Mode")["Delivery Delay Days"]
        .mean()
        .sort_values(ascending=False)
    )

    # Delivery status counts
    delayed_orders = (
        filtered_orders["Order Delivery Status"] == "Delayed"
    ).sum()

    total_filtered_orders = len(filtered_orders)

    delayed_percentage = (
        delayed_orders / total_filtered_orders * 100
        if total_filtered_orders > 0 else 0
    )

    # Generate insights
    insight_columns = st.columns(3)

    # Insight 1: Best region
    with insight_columns[0]:

        if not region_performance.empty:

            best_region = region_performance.index[0]
            best_sales = region_performance.iloc[0]

            st.markdown(f"""
            <div class="metric-card" style="min-height:180px;">
                <div class="metric-label">🏆 TOP SALES REGION</div>
                <div class="metric-value" style="font-size:21px;">
                    {best_region}
                </div>
                <div class="metric-note">
                    Sales contribution: {best_sales:,.2f}
                </div>
                <p style="color:#24A47A;font-size:12px;">
                    Highest sales among selected regions.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Insight 2: Shipping delay
    with insight_columns[1]:

        if not shipping_performance.empty:

            worst_shipping = shipping_performance.index[0]
            highest_delay = shipping_performance.iloc[0]

            st.markdown(f"""
            <div class="metric-card" style="min-height:180px;">
                <div class="metric-label">⚠️ SHIPPING DELAY ALERT</div>
                <div class="metric-value" style="font-size:21px;">
                    {worst_shipping}
                </div>
                <div class="metric-note">
                    Average delay: {highest_delay:.2f} days
                </div>
                <p style="color:#E45C68;font-size:12px;">
                    Highest average delivery delay among selected modes.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Insight 3: Delivery performance
    with insight_columns[2]:

        if delayed_percentage >= 30:
            status_color = "#E45C68"
            status_text = "High delivery delay"
        elif delayed_percentage >= 15:
            status_color = "#E5A13A"
            status_text = "Moderate delivery delay"
        else:
            status_color = "#24A47A"
            status_text = "Lower delivery delay"

        st.markdown(f"""
        <div class="metric-card" style="min-height:180px;">
            <div class="metric-label">📦 DELIVERY HEALTH</div>
            <div class="metric-value" style="font-size:27px;">
                {delayed_percentage:.1f}%
            </div>
            <div class="metric-note">
                Delayed orders: {delayed_orders:,} / {total_filtered_orders:,}
            </div>
            <p style="color:{status_color};font-size:12px;">
                {status_text}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Business recommendation
    st.markdown("### Recommended Business Action")

    if delayed_percentage >= 30:

        st.warning(
            "Delivery delays are high in the selected data. "
            "Review shipping schedules and investigate the shipping "
            "modes with the highest average delays."
        )

    elif delayed_percentage >= 15:

        st.info(
            "Delivery performance needs attention. "
            "Monitor delayed orders and review regional shipping operations."
        )

    else:

        st.success(
            "Delivery performance is relatively stable in the selected data. "
            "Continue monitoring shipping efficiency."
        )

# ==================================================
# PAGE 2: DELIVERY INTELLIGENCE
# ==================================================

elif page == "Delivery Intelligence":

    st.markdown("## Delivery Intelligence")
    st.caption("Analyze delivery speed, delay patterns, and shipping performance.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "On-Time Orders",
            f"{(filtered_orders['Order Delivery Status'] == 'On Time').sum():,}",
            "Unique orders",
            "✅"
        )

    with c2:
        metric_card(
            "Delayed Orders",
            f"{(filtered_orders['Order Delivery Status'] == 'Delayed').sum():,}",
            "Unique orders",
            "⚠️"
        )

    with c3:
        metric_card(
            "Delayed Rate",
            f"{delayed_rate:.1f}%",
            "Share of unique orders",
            "📉"
        )

    with c4:
        metric_card(
            "Average Delay",
            f"{filtered_orders['Delivery Delay Days'].mean():.2f} days",
            "Mean delay across orders",
            "🕒"
        )

    left, right = st.columns(2)

    with left:

        section_header("Actual vs Scheduled Delivery")

        delivery_comparison = (
            filtered_orders.groupby("Shipping Mode", as_index=False)
            .agg({
                "Days for shipping (real)": "mean",
                "Days for shipment (scheduled)": "mean"
            })
        )

        comparison_long = delivery_comparison.melt(
            id_vars="Shipping Mode",
            var_name="Delivery Type",
            value_name="Days"
        )

        comparison_long["Delivery Type"] = comparison_long[
            "Delivery Type"
        ].replace({
            "Days for shipping (real)": "Actual",
            "Days for shipment (scheduled)": "Scheduled"
        })

        fig = px.bar(
            comparison_long,
            x="Shipping Mode",
            y="Days",
            color="Delivery Type",
            barmode="group",
            color_discrete_map={
                "Actual": "#3979D6",
                "Scheduled": "#F0A64A"
            }
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with right:

        section_header("Delivery Delay Distribution")

        fig = px.histogram(
            filtered_orders,
            x="Delivery Delay Days",
            nbins=35,
            color_discrete_sequence=["#3979D6"]
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#E45C68"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    section_header("Shipping Mode Detailed Performance")

    shipping_table = (
        filtered_orders.groupby("Shipping Mode")
        .agg(
            Total_Orders=("Order Id", "nunique"),
            Average_Actual_Days=("Days for shipping (real)", "mean"),
            Average_Scheduled_Days=("Days for shipment (scheduled)", "mean"),
            Average_Delay=("Delivery Delay Days", "mean")
        )
        .reset_index()
        .round(2)
    )

    st.dataframe(
        shipping_table,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# DELIVERY RISK ANALYSIS
# ==================================================

elif page == "Delivery Risk Analysis":

    st.markdown("## Delivery Risk Intelligence")
    st.caption(
        "Analyze recorded late-delivery risk indicators "
        "across shipping operations."
    )

    if "Late_delivery_risk" not in filtered_df.columns:

        st.error(
            "The Late_delivery_risk column was not found in the dataset."
        )
        st.stop()

    # Prepare one risk record per unique order
    risk_source = filtered_df.copy()

    risk_source["Late_delivery_risk"] = pd.to_numeric(
        risk_source["Late_delivery_risk"],
        errors="coerce"
    )

    risk_orders = (
        risk_source.groupby("Order Id", as_index=False)
        .agg({
            "Late_delivery_risk": "max",
            "Shipping Mode": "first",
            "Order Region": "first",
            "Delivery Delay Days": "median"
        })
    )

    risk_orders = risk_orders.dropna(
        subset=["Late_delivery_risk"]
    )

    risk_orders["Risk Status"] = risk_orders[
        "Late_delivery_risk"
    ].map({
        0: "Low Risk",
        1: "High Risk"
    }).fillna("Unknown")

    if risk_orders.empty:
        st.warning("No valid delivery-risk records are available.")
        st.stop()

    # Risk KPIs
    total_risk_orders = len(risk_orders)

    high_risk_orders = (
        risk_orders["Risk Status"] == "High Risk"
    ).sum()

    low_risk_orders = (
        risk_orders["Risk Status"] == "Low Risk"
    ).sum()

    high_risk_rate = (
        high_risk_orders / total_risk_orders * 100
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total Risk Records",
            f"{total_risk_orders:,}",
            "Unique orders with valid risk data",
            "📦"
        )

    with c2:
        metric_card(
            "High-Risk Orders",
            f"{high_risk_orders:,}",
            "Dataset risk indicator = 1",
            "⚠️"
        )

    with c3:
        metric_card(
            "Low-Risk Orders",
            f"{low_risk_orders:,}",
            "Dataset risk indicator = 0",
            "✅"
        )

    with c4:
        metric_card(
            "High-Risk Rate",
            f"{high_risk_rate:.1f}%",
            "Share of valid risk records",
            "📊"
        )

    st.write("")

    # Risk distribution
    left, right = st.columns(2)

    with left:

        section_header("Delivery Risk Distribution")

        risk_distribution = (
            risk_orders["Risk Status"]
            .value_counts()
            .reset_index()
        )

        risk_distribution.columns = ["Risk Status", "Orders"]

        fig = px.pie(
            risk_distribution,
            names="Risk Status",
            values="Orders",
            hole=0.6,
            color="Risk Status",
            color_discrete_map={
                "High Risk": "#E45C68",
                "Low Risk": "#24A47A",
                "Unknown": "#98A2B3"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            chart_style(fig, 380),
            use_container_width=True
        )

    with right:

        section_header("Risk by Shipping Mode")

        mode_risk = (
            risk_orders.groupby("Shipping Mode")
            .agg(
                Total_Orders=("Order Id", "nunique"),
                High_Risk_Orders=(
                    "Risk Status",
                    lambda x: (x == "High Risk").sum()
                )
            )
            .reset_index()
        )

        mode_risk["High Risk Rate"] = (
            mode_risk["High_Risk_Orders"]
            / mode_risk["Total_Orders"] * 100
        )

        fig = px.bar(
            mode_risk.sort_values("High Risk Rate"),
            x="High Risk Rate",
            y="Shipping Mode",
            orientation="h",
            color="High Risk Rate",
            color_continuous_scale="Reds",
            hover_data=["Total_Orders", "High_Risk_Orders"]
        )

        fig.update_layout(
            xaxis_title="High-Risk Orders (%)",
            yaxis_title="Shipping Mode"
        )

        st.plotly_chart(
            chart_style(fig, 380),
            use_container_width=True
        )

    # Regional risk analysis
    section_header("Regional Delivery Risk Comparison")

    region_risk = (
        risk_orders.groupby("Order Region")
        .agg(
            Total_Orders=("Order Id", "nunique"),
            High_Risk_Orders=(
                "Risk Status",
                lambda x: (x == "High Risk").sum()
            )
        )
        .reset_index()
    )

    region_risk["High Risk Rate"] = (
        region_risk["High_Risk_Orders"]
        / region_risk["Total_Orders"] * 100
    )

    fig = px.bar(
        region_risk.sort_values("High Risk Rate", ascending=False),
        x="Order Region",
        y="High Risk Rate",
        color="High Risk Rate",
        color_continuous_scale="Reds",
        hover_data=["Total_Orders", "High_Risk_Orders"]
    )

    fig.update_layout(
        xaxis_title="Order Region",
        yaxis_title="High-Risk Orders (%)"
    )

    st.plotly_chart(
        chart_style(fig, 420),
        use_container_width=True
    )

    # Compare risk with actual delivery outcomes
    section_header("Recorded Risk vs Actual Delivery Outcome")

    risk_outcome = (
        risk_orders.groupby("Risk Status")
        .agg(
            Total_Orders=("Order Id", "nunique"),
            Average_Delay=("Delivery Delay Days", "mean"),
            Delayed_Orders=(
                "Delivery Delay Days",
                lambda x: (x > 0).sum()
            )
        )
        .reset_index()
    )

    risk_outcome["Actual Delayed Rate"] = (
        risk_outcome["Delayed_Orders"]
        / risk_outcome["Total_Orders"] * 100
    )

    st.dataframe(
        risk_outcome.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Interpretation: Late_delivery_risk is a risk indicator "
        "provided by the original dataset. Compare it with actual "
        "delivery delays to understand how the recorded risk relates "
        "to delivery outcomes. This analysis does not represent a "
        "new machine-learning prediction."
    )

    # Download risk analysis
    csv_data = risk_orders.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Delivery Risk Report",
        data=csv_data,
        file_name="delivery_risk_analysis.csv",
        mime="text/csv"
    )

# ==================================================
# PAGE 3: REGIONAL ANALYTICS
# ==================================================

elif page == "Regional Analytics":

    st.markdown("## Regional Analytics")
    st.caption("Explore regional sales contribution and delivery efficiency.")

    regional = (
        filtered_orders.groupby("Order Region")
        .agg(
            Total_Orders=("Order Id", "nunique"),
            Average_Delay=("Delivery Delay Days", "mean"),
            Average_Delivery=("Days for shipping (real)", "mean")
        )
        .reset_index()
    )

    regional_sales = (
        filtered_df.groupby("Order Region")["Sales"]
        .sum()
        .reset_index()
    )

    regional = regional.merge(
        regional_sales,
        on="Order Region",
        how="left"
    ).round(2)

    left, right = st.columns(2)

    with left:

        section_header("Regional Sales Performance")

        fig = px.bar(
            regional.sort_values("Sales", ascending=False),
            x="Order Region",
            y="Sales",
            color="Sales",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            chart_style(fig, 400),
            use_container_width=True
        )

    with right:

        section_header("Average Delivery Delay by Region")

        fig = px.bar(
            regional.sort_values("Average_Delay"),
            x="Average_Delay",
            y="Order Region",
            orientation="h",
            color="Average_Delay",
            color_continuous_scale="RdYlGn_r"
        )

        st.plotly_chart(
            chart_style(fig, 400),
            use_container_width=True
        )

    section_header("Regional Performance Matrix")

    fig = px.scatter(
        regional,
        x="Average_Delay",
        y="Sales",
        size="Total_Orders",
        color="Order Region",
        hover_name="Order Region",
        size_max=45
    )

    st.plotly_chart(
        chart_style(fig, 450),
        use_container_width=True
    )

    st.dataframe(
        regional.sort_values("Sales", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# PAGE 4: REVENUE ANALYTICS
# ==================================================

elif page == "Revenue Analytics":

    st.markdown("## Revenue Analytics")
    st.caption("Analyze sales trends, markets, and business contribution.")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Total Sales",
            f"{total_sales:,.0f}",
            "Line-item sales sum",
            "💰"
        )

    with c2:
        metric_card(
            "Total Item Quantity",
            f"{total_items:,.0f}",
            "Units represented by item records",
            "📦"
        )

    with c3:
        metric_card(
            "Average Sales per Item Record",
            f"{filtered_df['Sales'].mean():,.2f}",
            "Average row-level sales",
            "📈"
        )

    left, right = st.columns(2)

    with left:

        section_header("Monthly Revenue Trend")

        monthly = (
            filtered_df.dropna(subset=["order date (DateOrders)"])
            .set_index("order date (DateOrders)")
            .resample("ME")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="order date (DateOrders)",
            y="Sales",
            markers=True,
            color_discrete_sequence=["#3979D6"]
        )

        fig.update_traces(line_width=3)

        st.plotly_chart(
            chart_style(fig, 400),
            use_container_width=True
        )

    with right:

        section_header("Sales by Market")

        market_sales = (
            filtered_df.groupby("Market")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        fig = px.pie(
            market_sales,
            names="Market",
            values="Sales",
            hole=0.55
        )

        st.plotly_chart(
            chart_style(fig, 400),
            use_container_width=True
        )

    section_header("Market Sales Comparison")

    fig = px.bar(
        market_sales,
        x="Market",
        y="Sales",
        color="Sales",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        chart_style(fig),
        use_container_width=True
    )


# ==================================================
# PAGE 5: PRODUCT ANALYTICS
# ==================================================

elif page == "Product Analytics":

    st.markdown("## Product Analytics")
    st.caption("Understand product category performance and demand patterns.")

    category = (
        filtered_df.groupby("Category Name")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Quantity=("Order Item Quantity", "sum"),
            Average_Product_Price=("Product Price", "mean")
        )
        .reset_index()
        .sort_values("Total_Sales", ascending=False)
    )

    left, right = st.columns(2)

    with left:

        section_header("Top 15 Product Categories by Sales")

        top_categories = category.head(15)

        fig = px.bar(
            top_categories.sort_values("Total_Sales"),
            x="Total_Sales",
            y="Category Name",
            orientation="h",
            color="Total_Sales",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            chart_style(fig, 480),
            use_container_width=True
        )

    with right:

        section_header("Category Demand Distribution")

        fig = px.treemap(
            category.head(20),
            path=["Category Name"],
            values="Total_Quantity",
            color="Total_Sales",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            chart_style(fig, 480),
            use_container_width=True
        )

    section_header("Complete Category Performance")

    st.dataframe(
        category.round(2),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "<div style='text-align:center;color:#8792A5;font-size:11px;'>"
    "SUPPLY CHAIN INTELLIGENCE PLATFORM | DATA ANALYTICS PROJECT"
    "</div>",
    unsafe_allow_html=True
)

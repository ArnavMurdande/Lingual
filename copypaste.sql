# Import Python packages
import os
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Coffee Sales Dashboard",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Coffee Sales Dashboard")
st.caption("Sales performance and customer insights")

# ---------------------------------------------------------
# CONNECT TO SNOWFLAKE
# ---------------------------------------------------------
conn = st.connection(
    "snowflake",
    ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL")
)

session = conn.session()

# Fully qualified table name
SALES_TABLE = "COFFEE_LAB.DATA.SALES"

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Dashboard Filters")

# Get date range from the SALES table
date_range = session.sql(f"""
    SELECT
        MIN(SALE_DATE) AS MIN_DATE,
        MAX(SALE_DATE) AS MAX_DATE
    FROM {SALES_TABLE}
""").to_pandas()

min_date = date_range.loc[0, "MIN_DATE"]
max_date = date_range.loc[0, "MAX_DATE"]

selected_dates = st.sidebar.date_input(
    "Select sales date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Handle single-date and date-range selections
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
    start_date = selected_dates[0]
    end_date = selected_dates[0]
else:
    start_date = selected_dates
    end_date = selected_dates

# Get available stores
stores_df = session.sql(f"""
    SELECT DISTINCT STORE
    FROM {SALES_TABLE}
    WHERE STORE IS NOT NULL
    ORDER BY STORE
""").to_pandas()

store_options = stores_df["STORE"].tolist()

selected_stores = st.sidebar.multiselect(
    "Select stores",
    options=store_options,
    default=store_options
)

# Get available products
products_df = session.sql(f"""
    SELECT DISTINCT PRODUCT
    FROM {SALES_TABLE}
    WHERE PRODUCT IS NOT NULL
    ORDER BY PRODUCT
""").to_pandas()

product_options = products_df["PRODUCT"].tolist()

selected_products = st.sidebar.multiselect(
    "Select products",
    options=product_options,
    default=product_options
)

# Stop if no store or product is selected
if not selected_stores:
    st.warning("Please select at least one store.")
    st.stop()

if not selected_products:
    st.warning("Please select at least one product.")
    st.stop()

# ---------------------------------------------------------
# CREATE FILTER CONDITIONS
# ---------------------------------------------------------
# Escape single quotes in values before using them in SQL
safe_stores = [
    store.replace("'", "''")
    for store in selected_stores
]

safe_products = [
    product.replace("'", "''")
    for product in selected_products
]

store_filter = ", ".join(
    [f"'{store}'" for store in safe_stores]
)

product_filter = ", ".join(
    [f"'{product}'" for product in safe_products]
)

where_clause = f"""
    WHERE SALE_DATE BETWEEN '{start_date}' AND '{end_date}'
      AND STORE IN ({store_filter})
      AND PRODUCT IN ({product_filter})
"""

# ---------------------------------------------------------
# QUERY 1: KPI INFORMATION
# ---------------------------------------------------------
kpi_df = session.sql(f"""
    SELECT
        COALESCE(SUM(QUANTITY * PRICE), 0) AS TOTAL_REVENUE,
        COALESCE(ROUND(AVG(CUSTOMER_AGE), 1), 0) AS AVG_CUSTOMER_AGE,
        COUNT(*) AS TOTAL_TRANSACTIONS,
        COALESCE(SUM(QUANTITY), 0) AS UNITS_SOLD
    FROM {SALES_TABLE}
    {where_clause}
""").to_pandas()

total_revenue = float(kpi_df.loc[0, "TOTAL_REVENUE"])
average_age = float(kpi_df.loc[0, "AVG_CUSTOMER_AGE"])
total_transactions = int(kpi_df.loc[0, "TOTAL_TRANSACTIONS"])
units_sold = int(kpi_df.loc[0, "UNITS_SOLD"])

# ---------------------------------------------------------
# DISPLAY KPI TILES
# ---------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric(
        label="💰 Total Revenue",
        value=f"${total_revenue:,.2f}",
        help="Total revenue calculated as quantity multiplied by price."
    )

with kpi_col2:
    st.metric(
        label="👤 Average Customer Age",
        value=f"{average_age:.1f} years",
        help="Average age of customers in the selected data."
    )

with kpi_col3:
    st.metric(
        label="🧾 Total Transactions",
        value=f"{total_transactions:,}",
        help="Total number of sales records."
    )

with kpi_col4:
    st.metric(
        label="☕ Units Sold",
        value=f"{units_sold:,}",
        help="Total quantity of products sold."
    )

st.divider()

# ---------------------------------------------------------
# QUERY 2: REVENUE BY PRODUCT
# ---------------------------------------------------------
revenue_by_product = session.sql(f"""
    SELECT
        PRODUCT,
        ROUND(SUM(QUANTITY * PRICE), 2) AS REVENUE
    FROM {SALES_TABLE}
    {where_clause}
    GROUP BY PRODUCT
    ORDER BY REVENUE DESC
""").to_pandas()

# ---------------------------------------------------------
# QUERY 3: REVENUE BY STORE
# ---------------------------------------------------------
revenue_by_store = session.sql(f"""
    SELECT
        STORE,
        ROUND(SUM(QUANTITY * PRICE), 2) AS REVENUE
    FROM {SALES_TABLE}
    {where_clause}
    GROUP BY STORE
    ORDER BY REVENUE DESC
""").to_pandas()

# ---------------------------------------------------------
# FIRST CHART ROW
# ---------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Revenue by Product")

    if revenue_by_product.empty:
        st.info("No product revenue data is available.")
    else:
        st.bar_chart(
            data=revenue_by_product,
            x="PRODUCT",
            y="REVENUE",
            x_label="Product",
            y_label="Revenue ($)",
            color="#6F4E37",
            use_container_width=True
        )

with chart_col2:
    st.subheader("Revenue by Store")

    if revenue_by_store.empty:
        st.info("No store revenue data is available.")
    else:
        st.bar_chart(
            data=revenue_by_store,
            x="STORE",
            y="REVENUE",
            x_label="Store",
            y_label="Revenue ($)",
            color="#D2691E",
            use_container_width=True
        )

# ---------------------------------------------------------
# QUERY 4: TRANSACTIONS BY STORE
# ---------------------------------------------------------
transactions_by_store = session.sql(f"""
    SELECT
        STORE,
        COUNT(*) AS TRANSACTIONS
    FROM {SALES_TABLE}
    {where_clause}
    GROUP BY STORE
    ORDER BY TRANSACTIONS DESC
""").to_pandas()

# ---------------------------------------------------------
# QUERY 5: DAILY SALES
# ---------------------------------------------------------
daily_sales = session.sql(f"""
    SELECT
        SALE_DATE,
        ROUND(SUM(QUANTITY * PRICE), 2) AS DAILY_SALES
    FROM {SALES_TABLE}
    {where_clause}
    GROUP BY SALE_DATE
    ORDER BY SALE_DATE
""").to_pandas()

# Convert date for correct chart display
if not daily_sales.empty:
    daily_sales["SALE_DATE"] = daily_sales["SALE_DATE"].astype(str)

# ---------------------------------------------------------
# SECOND CHART ROW
# ---------------------------------------------------------
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("Transactions by Store")

    if transactions_by_store.empty:
        st.info("No transaction data is available.")
    else:
        st.bar_chart(
            data=transactions_by_store,
            x="STORE",
            y="TRANSACTIONS",
            x_label="Store",
            y_label="Number of Transactions",
            color="#4682B4",
            use_container_width=True
        )

with chart_col4:
    st.subheader("Daily Sales")

    if daily_sales.empty:
        st.info("No daily sales data is available.")
    else:
        st.line_chart(
            data=daily_sales,
            x="SALE_DATE",
            y="DAILY_SALES",
            x_label="Sale Date",
            y_label="Daily Sales ($)",
            color="#2E8B57",
            use_container_width=True
        )

# ---------------------------------------------------------
# BEST-PERFORMING DAY
# ---------------------------------------------------------
st.divider()
st.subheader("🏆 Highest-Performing Sales Day")

if not daily_sales.empty:
    best_day_row = daily_sales.loc[daily_sales["DAILY_SALES"].idxmax()]

    best_day = best_day_row["SALE_DATE"]
    best_day_sales = float(best_day_row["DAILY_SALES"])

    best_col1, best_col2 = st.columns(2)

    with best_col1:
        st.metric(
            label="Best Sales Date",
            value=str(best_day)
        )

    with best_col2:
        st.metric(
            label="Revenue on Best Day",
            value=f"${best_day_sales:,.2f}"
        )
else:
    st.info("No sales records were found for the selected filters.")

# ---------------------------------------------------------
# UNDERLYING SALES DATA
# ---------------------------------------------------------
with st.expander("View Underlying Sales Data"):
    sales_data = session.sql(f"""
        SELECT
            SALE_DATE,
            STORE,
            PRODUCT,
            QUANTITY,
            PRICE,
            CUSTOMER_AGE,
            ROUND(QUANTITY * PRICE, 2) AS REVENUE
        FROM {SALES_TABLE}
        {where_clause}
        ORDER BY SALE_DATE, STORE, PRODUCT
    """).to_pandas()

    st.dataframe(
        sales_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "SALE_DATE": st.column_config.DateColumn(
                "Sale Date"
            ),
            "STORE": st.column_config.TextColumn(
                "Store"
            ),
            "PRODUCT": st.column_config.TextColumn(
                "Product"
            ),
            "QUANTITY": st.column_config.NumberColumn(
                "Quantity",
                format="%d"
            ),
            "PRICE": st.column_config.NumberColumn(
                "Unit Price",
                format="$%.2f"
            ),
            "CUSTOMER_AGE": st.column_config.NumberColumn(
                "Customer Age",
                format="%d"
            ),
            "REVENUE": st.column_config.NumberColumn(
                "Revenue",
                format="$%.2f"
            )
        }
    )

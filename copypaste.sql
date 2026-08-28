import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

st.set_page_config(
    page_title="Logistics Performance Dashboard",
    layout="wide"
)

session = get_active_session()

st.title("🚚 Logistics Performance Dashboard")

# Load Data
df = session.sql("""
    SELECT *
    FROM DELIVERIES
""").to_pandas()

# -----------------------------
# KPI Metrics
# -----------------------------

total_orders = len(df)

ontime_pct = round(
    (df["IS_ONTIME"] == "Yes").mean() * 100,
    1
)

avg_delivery_time = round(
    df["DELIVERY_TIME_MIN"].mean(),
    1
)

avg_rating = round(
    df["DRIVER_RATING"].mean(),
    2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("On-Time %", f"{ontime_pct}%")
col3.metric("Avg Delivery Time", f"{avg_delivery_time} min")
col4.metric("Avg Driver Rating", avg_rating)

st.divider()

# -----------------------------
# Deliveries by City
# -----------------------------

city_orders = (
    df.groupby("CITY")
      .size()
      .reset_index(name="TOTAL_DELIVERIES")
)

st.subheader("Deliveries by City")

st.bar_chart(
    city_orders.set_index("CITY")
)

# -----------------------------
# On-Time Performance by City
# -----------------------------

city_ontime = (
    df.groupby("CITY")
      .apply(
          lambda x:
          ((x["IS_ONTIME"]=="Yes").mean()*100)
      )
      .reset_index(name="ONTIME_PERCENT")
)

st.subheader("On-Time % by City")

st.bar_chart(
    city_ontime.set_index("CITY")
)

# -----------------------------
# Vehicle Type Performance
# -----------------------------

vehicle_perf = (
    df.groupby("VEHICLE_TYPE")
    .agg({
        "DELIVERY_TIME_MIN":"mean",
        "DISTANCE_KM":"mean"
    })
)

st.subheader("Vehicle Performance")

st.bar_chart(vehicle_perf)

# -----------------------------
# Daily Delivery Volume
# -----------------------------

daily = (
    df.groupby("ORDER_DATE")
    .size()
    .reset_index(name="DELIVERIES")
)

daily["ORDER_DATE"] = pd.to_datetime(
    daily["ORDER_DATE"]
)

daily = daily.sort_values("ORDER_DATE")

st.subheader("Daily Delivery Volume")

st.line_chart(
    daily.set_index("ORDER_DATE")
)

# -----------------------------
# Top Rated Deliveries
# -----------------------------

st.subheader("Top Rated Deliveries")

top10 = df.sort_values(
    by="DRIVER_RATING",
    ascending=False
).head(10)

st.dataframe(
    top10[
        [
            "ORDER_ID",
            "CITY",
            "VEHICLE_TYPE",
            "DRIVER_RATING",
            "IS_ONTIME"
        ]
    ],
    use_container_width=True
)

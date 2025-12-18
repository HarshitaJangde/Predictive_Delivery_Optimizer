# ============================================
# Predictive Delivery Optimizer with Cost Intelligence
# Author: Harshita
# Tech Stack: Python, Streamlit, Pandas, Scikit-learn
# Python version -> 3.10.8
# ============================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px
import plotly.io as pio

# ============================================
# Page Config
# ============================================

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

st.title("🚚 Predictive Delivery Optimizer with Cost Intelligence")
st.caption("Predict delivery delays, estimate financial impact, and recommend actions")
st.write("Kaleido loaded successfully")


# ============================================
# Load Data
# ============================================

@st.cache_data
def load_data():
    orders = pd.read_csv("Data/orders.csv")
    delivery = pd.read_csv("Data/delivery_performance.csv")
    routes = pd.read_csv("Data/routes_distance.csv")
    costs = pd.read_csv("Data/cost_breakdown.csv")
    return orders, delivery, routes, costs

orders_df, delivery_df, routes_df, costs_df = load_data()

# ============================================
# Preprocess & Merge
# ============================================

def preprocess_data():
    df = orders_df.merge(delivery_df, on="Order_ID", how="left")
    df = df.merge(routes_df, on="Order_ID", how="left")
    df = df.merge(costs_df, on="Order_ID", how="left")

    # Handle missing values
    df.fillna({
        "Traffic_Delay_Minutes": 0,
        "Weather_Impact": 0,
        "Toll_Charges_INR": 0,
        "Fuel_Cost": df["Fuel_Cost"].median(),
        "Labor_Cost": df["Labor_Cost"].median(),
        "Vehicle_Maintenance": df["Vehicle_Maintenance"].median(),
    }, inplace=True)
    
    # Encode Weather Impact (business logic)
    weather_map = {
        "Clear": 0,
        "Light_Rain": 1,
        "Heavy_Rain": 2,
        "Storm": 3,
        "Fog": 2
    }

    df["Weather_Impact"] = df["Weather_Impact"].map(weather_map).fillna(0)

    # Target variable
    df["Delayed"] = (df["Actual_Delivery_Days"] > df["Promised_Delivery_Days"]).astype(int)

    # Total cost metric
    df["Total_Cost"] = (
        df["Fuel_Cost"]
        + df["Labor_Cost"]
        + df["Vehicle_Maintenance"]
        + df["Insurance"]
        + df["Packaging_Cost"]
        + df["Technology_Platform_Fee"]
        + df["Other_Overhead"]
        + df["Toll_Charges_INR"]
    )

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    return df

df = preprocess_data()

# ============================================
# Sidebar Filters
# ============================================

st.sidebar.header("🔎 Filters")

priority_filter = st.sidebar.multiselect(
    "Select Priority",
    options=df["Priority"].unique(),
    default=df["Priority"].unique()
)

df_filtered = df[df["Priority"].isin(priority_filter)]

# ============================================
# Encode Categorical Features
# ============================================

encoded_df = df_filtered.copy()
encoder = LabelEncoder()

for col in ["Priority", "Product_Category", "Carrier"]:
    encoded_df[col] = encoder.fit_transform(encoded_df[col].astype(str))

# ============================================
# Feature Selection
# ============================================

features = [
    "Priority",
    "Order_Value_INR",
    "Distance_KM",
    "Traffic_Delay_Minutes",
    "Weather_Impact",
    "Fuel_Cost",
    "Labor_Cost"
]

X = encoded_df[features]
y = encoded_df["Delayed"]

# Replace any remaining NaNs with median (numeric-safe)
X = X.fillna(X.median())

# ============================================
# Train Model
# ============================================

@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# ============================================
# KPIs
# ============================================

st.subheader("📊 Executive Overview")

col1, col2, col3 = st.columns(3)

col1.metric("🚨 Delay Rate", f"{y.mean() * 100:.1f}%")
col2.metric("💰 Avg Delivery Cost", f"₹{df_filtered['Total_Cost'].mean():,.0f}")
col3.metric("📦 Total Orders", len(df_filtered))

# ============================================
# Visualizations (4 Types)
# ============================================

st.subheader("📈 Operational Insights")


# 1️⃣ Bar Chart
fig1 = px.bar(df_filtered, x="Priority", y="Delayed", title="Delays by Priority")
st.plotly_chart(fig1, use_container_width=True)


# 2️⃣ Pie Chart
status_counts = df_filtered["Delivery_Status"].value_counts()
fig2 = px.pie(values=status_counts.values, names=status_counts.index, title="Delivery Status")
st.plotly_chart(fig2, use_container_width=True)


# 3️⃣ Scatter Plot
fig3 = px.scatter(
    df_filtered,
    x="Distance_KM",
    y="Traffic_Delay_Minutes",
    color="Delayed",
    title="Distance vs Traffic Delay"
)
st.plotly_chart(fig3, use_container_width=True)


# 4️⃣ Line Chart
cost_trend = df_filtered.groupby("Order_Date")["Total_Cost"].mean().reset_index()
fig4 = px.line(cost_trend, x="Order_Date", y="Total_Cost", title="Avg Delivery Cost Over Time")
st.plotly_chart(fig4, use_container_width=True)



# ============================================
# Order-Level Prediction
# ============================================

st.subheader("🧠 Predict Delay for a Specific Order")

selected_order = st.selectbox("Select Order ID", df_filtered["Order_ID"].unique())

order_row = encoded_df[encoded_df["Order_ID"] == selected_order][features]

if not order_row.empty:
    prob = model.predict_proba(order_row)[0][1]

    if prob > 0.7:
        risk = "🔴 High Risk"
        action = "Reroute or assign fastest carrier"
    elif prob > 0.4:
        risk = "🟠 Medium Risk"
        action = "Monitor closely with buffer"
    else:
        risk = "🟢 Low Risk"
        action = "Proceed as planned"

    st.metric("Delay Probability", f"{prob:.2%}")
    st.info(f"**Risk Level:** {risk}\n\n**Recommended Action:** {action}")

# ============================================
# Export
# ============================================

st.subheader("⬇️ Export Insights")

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    convert_df(df_filtered),
    "predictive_delivery_insights.csv",
    "text/csv"
)

st.markdown("---")
st.caption("Predictive analytics for proactive logistics decision-making")

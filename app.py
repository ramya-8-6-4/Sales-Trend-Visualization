import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from analysis import *

# PAGE CONFIG
st.set_page_config(
    page_title="Sales Dashboard",
    layout="centered"
)

# TITLE
st.title(" Sales Trend Visualization Dashboard")

# LOAD DATA
df = load_data()

df = clean_data(df)

# DATASET PREVIEW
st.subheader(" Dataset Preview")

st.dataframe(df.head())

# MONTHLY SALES TREND
st.subheader(" Monthly Sales Trend")

monthly = monthly_sales(df)

fig1, ax1 = plt.subplots(figsize=(6, 3))

# Line chart
ax1.plot(
    monthly["Month"],
    monthly["Sales"],
    marker='o',
    linewidth=2
)

#  Show fewer x-axis labels (prevents collision)
step = max(1, len(monthly) // 6)
ax1.set_xticks(range(0, len(monthly), step))
ax1.set_xticklabels(
    monthly["Month"][::step],
    rotation=45,
    ha="right",
    fontsize=8
)

# Grid for clarity
ax1.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

st.pyplot(fig1)

# SALES BY REGION
st.subheader(" Sales by Region")

region = sales_by_region(df)

fig2, ax2 = plt.subplots(figsize=(3,2))

sns.barplot(
    data=region,
    x="Region",
    y="Sales",
    ax=ax2
)

plt.xticks(fontsize=6)

plt.yticks(fontsize=6)

st.pyplot(fig2)

# SALES BY CATEGORY
st.subheader(" Sales by Category")

category = sales_by_category(df)

fig3, ax3 = plt.subplots(figsize=(2.5,2.5))

ax3.pie(
    category["Sales"],
    labels=category["Category"],
    autopct="%1.1f%%",
    textprops={'fontsize': 6}
)

st.pyplot(fig3)

# SALES PREDICTION
st.subheader(" Future Sales Prediction")

monthly_data, future_df = predict_sales(df)

fig4, ax4 = plt.subplots(figsize=(3,2))

# ACTUAL SALES
ax4.plot(
    monthly_data["Month_Num"],
    monthly_data["Sales"],
    marker='o',
    label="Actual"
)

# PREDICTED SALES
ax4.plot(
    range(len(monthly_data), len(monthly_data)+6),
    future_df["Predicted_Sales"],
    marker='o',
    linestyle='dashed',
    label="Predicted"
)

ax4.legend(fontsize=6)

plt.xticks(fontsize=6)

plt.yticks(fontsize=6)

st.pyplot(fig4)

# PREDICTION TABLE
st.subheader(" Predicted Sales Data")

st.dataframe(future_df)

# SUCCESS MESSAGE
st.success("Dashboard Created Successfully ")

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# LOAD DATA
def load_data():

    df = pd.read_csv(
        r"C:\Users\ramya\OneDrive\Documents\Desktop\Task 1\Sample - Superstore.csv",
        encoding='latin1'
    )

    return df


# CLEAN DATA
def clean_data(df):

    df = df.dropna()

    return df


# SALES BY REGION
def sales_by_region(df):

    return df.groupby("Region")["Sales"].sum().reset_index()


# SALES BY CATEGORY
def sales_by_category(df):

    return df.groupby("Category")["Sales"].sum().reset_index()


# MONTHLY SALES
def monthly_sales(df):

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    df["Month"] = df["Order Date"].dt.to_period("M")

    monthly = df.groupby("Month")["Sales"].sum().reset_index()

    monthly["Month"] = monthly["Month"].astype(str)

    return monthly


# SALES PREDICTION
def predict_sales(df):

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    monthly = df.groupby(
        df["Order Date"].dt.to_period("M")
    )["Sales"].sum().reset_index()

    monthly["Month_Num"] = np.arange(len(monthly))

    X = monthly[["Month_Num"]]

    y = monthly["Sales"]

    # MODEL
    model = LinearRegression()

    model.fit(X, y)

    # FUTURE MONTHS
    future_months = np.arange(
        len(monthly),
        len(monthly) + 6
    ).reshape(-1, 1)

    future_months_df = pd.DataFrame(
        future_months,
        columns=["Month_Num"]
    )

    predictions = model.predict(future_months_df)

    future_df = pd.DataFrame({
        "Future_Month": range(1, 7),
        "Predicted_Sales": predictions
    })

    return monthly, future_df

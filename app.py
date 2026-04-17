# pip install streamlit pandas numpy plotly scikit-learn openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(page_title="Kenya Loan Analytics Dashboard", layout="wide")

st.title("Kenya Loan Portfolio & Credit Risk Dashboard - Jamii Bora Microfinance Ltd")

# -------------------
# GENERATE REALISTIC DATA
# -------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 3000

    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        "Date": dates,
        "Loan Amount": np.random.randint(5000, 150000, n),
        "Interest Rate": np.random.uniform(8, 24, n),
        "Credit Score": np.random.randint(300, 850, n),
        "Income": np.random.randint(15000, 300000, n),
        "Loan Term": np.random.choice([6, 12, 24, 36, 48, 60, 72], n),
        "Region": np.random.choice(["Nairobi", "Mombasa", "Kisumu", "Eldoret", "Nakuru", "Kakamega", "Kitale"], n)
    })

    df["Default"] = np.where(
        (df["Credit Score"] < 550) & (df["Interest Rate"] > 18), 1, 0
    )

    return df

df = generate_data()

# -------------------
# SIDEBAR FILTERS
# -------------------
st.sidebar.header("🔍 Filters")

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Date"].min(), df["Date"].max()]
)

df = df[
    (df["Region"].isin(region)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# -------------------
# KPI DASHBOARD
# -------------------
st.subheader("Portfolio KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Loans", len(df))
col2.metric("Portfolio Value (KES)", f"{df['Loan Amount'].sum():,.0f}")
col3.metric("Avg Interest Rate", f"{df['Interest Rate'].mean():.2f}%")
col4.metric("Default Rate", f"{df['Default'].mean()*100:.2f}%")

# -------------------
# TIME SERIES ANALYSIS
# -------------------
st.subheader("Loan Trends Over Time")

time_df = df.groupby("Date")["Loan Amount"].sum().reset_index()

fig_time = px.line(time_df, x="Date", y="Loan Amount", title="Loan Disbursement Trend")
st.plotly_chart(fig_time, use_container_width=True)

# -------------------
# REGIONAL PERFORMANCE
# -------------------
st.subheader("Regional Loan Distribution")

region_df = df.groupby("Region")["Loan Amount"].sum().reset_index()

fig_region = px.bar(region_df, x="Region", y="Loan Amount", color="Region")
st.plotly_chart(fig_region, use_container_width=True)

# -------------------
# RISK ANALYSIS
# -------------------
st.subheader("Credit Risk Analysis")

fig_scatter = px.scatter(
    df,
    x="Credit Score",
    y="Loan Amount",
    color=df["Default"].astype(str),
    size="Income",
    title="Risk Segmentation"
)

st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------
# MACHINE LEARNING MODEL
# -------------------
st.subheader("Default Prediction Model")

features = ["Loan Amount", "Interest Rate", "Credit Score", "Income", "Loan Term"]
X = df[features]
y = df["Default"]

model = RandomForestClassifier()
model.fit(X, y)

# -------------------
# PREDICTION TOOL
# -------------------
st.subheader("🔮 Predict Loan Risk")

col1, col2 = st.columns(2)

loan_amt = col1.number_input("Loan Amount", 5000, 100000, 20000)
interest = col1.number_input("Interest Rate", 1.0, 30.0, 12.0)
credit = col2.number_input("Credit Score", 300, 850, 600)
income = col2.number_input("Income", 10000, 300000, 50000)
term = st.selectbox("Loan Term", [6, 12, 24, 36])

if st.button("Predict Risk"):
    input_data = np.array([[loan_amt, interest, credit, income, term]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")

# -------------------
# SMART INSIGHTS (AUTO REPORT)
# -------------------
st.subheader("📄 Automated Insights")

st.write(f"""
- Total portfolio value is **KES {df['Loan Amount'].sum():,.0f}**
- Default rate is **{df['Default'].mean()*100:.2f}%**
- Highest lending region: **{region_df.sort_values('Loan Amount', ascending=False).iloc[0]['Region']}**
- Average borrower income: **KES {df['Income'].mean():,.0f}**
""")

# -------------------
# DOWNLOAD DATA
# -------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Dataset", csv, "kenya_loans.csv", "text/csv")

# -------------------
# FOOTER
# -------------------
st.markdown("---")
st.markdown("Kenya FinTech Analytics")

#In the terminal, run
#streamlit run app.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === Load the dataset ===
# Adjust this file path as needed
df = pd.read_excel("C:/Users/knigh/Downloads/EIA data T.xls", sheet_name="Data 1")

# === Step 1: Inspect and Clean Column Names ===
df.columns = df.columns.str.strip()

# === Step 2: Convert Date column ===
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Sort by date and reset index
df = df.sort_values('Date').reset_index(drop=True)

# === Step 3: Check for duplicates and missing values ===
print("\nMissing Values per Column:")
print(df.isna().sum())

print("\nDuplicate Dates:", df['Date'].duplicated().sum())

# === Step 4: Check date frequency (should be weekly) ===
date_diffs = df['Date'].diff().dt.days.dropna()
print("\nAverage days between observations:", date_diffs.mean())
print("Unique gaps found:", date_diffs.value_counts())

# === Step 5: Detect numeric columns and convert ===
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# If some numeric columns were read as objects, convert them
for col in df.columns:
    if df[col].dtype == 'object' and col != 'Date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# === Step 6: Summary statistics ===
print("\nSummary statistics:")
print(df.describe())

# === Step 7: Correlation matrix ===
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Petroleum Products")
plt.show()

# === Step 8: Check multicollinearity using VIF ===
# Drop any NaNs before computing VIF
df_vif = df[numeric_cols].dropna()
vif_data = pd.DataFrame()
vif_data["Feature"] = df_vif.columns
vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i)
                   for i in range(df_vif.shape[1])]

print("\nVariance Inflation Factors:")
print(vif_data.sort_values(by="VIF", ascending=False))

# === Step 9: Optional — visualize missing data pattern ===
plt.figure(figsize=(10, 4))
sns.heatmap(df[numeric_cols].isna(), cbar=False, yticklabels=False)
plt.title("Missing Data Pattern")
plt.show()

# Assuming df is already cleaned and loaded
numeric_df = df.select_dtypes(include=[np.number]).dropna()  # keep only numeric columns for correlation

# --- 1️⃣ Correlation Matrix ---
corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix:\n", corr_matrix)

# Heatmap visualization
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Weekly Product Variables")
plt.show()

# --- 2️⃣ Variance Inflation Factor (VIF) ---
# VIF detects multicollinearity — high values (>10) mean strong correlation with other predictors

# Prepare numeric data for VIF
X = numeric_df.copy()
X = X.loc[:, X.columns.notnull()]  # ensure no NaN column names
X = X.fillna(0)  # avoid dropping columns due to missing values

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factors:")
print(vif_data.sort_values(by="VIF", ascending=False))

from scipy.signal import detrend

X_detrended = numeric_df.apply(detrend)
vif_data["VIF_detrended"] = [variance_inflation_factor(X_detrended, i) for i in range(X_detrended.shape[1])]
print(vif_data[["Feature", "VIF_detrended"]])

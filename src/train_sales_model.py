# src/train_sales_model.py
"""
Sales Prediction Model - Portfolio Version
------------------------------------------
This script trains a Linear Regression model on the extended sales dataset,
evaluates its performance, and saves visualizations for portfolio use.

Author: Raphael Tankeu Buguieu
Date: 2026-03-10
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# -----------------------------
# Step 1: Setup paths
# -----------------------------
DATA_DIR = '../data'
os.makedirs(DATA_DIR, exist_ok=True)

# Load extended dataset
df = pd.read_csv(f'{DATA_DIR}/sales_prediction_data_extended.csv', parse_dates=['Date'])

# -----------------------------
# Step 2: Feature engineering
# -----------------------------
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday  # Monday=0, Sunday=6

# Features and target
X = df[['Year', 'Month', 'Day', 'Weekday']]
y = df['Sales']

# -----------------------------
# Step 3: Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Train Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Predictions & Evaluation
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------
# Step 6: Visualization
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot for portfolio
plot_path = f'{DATA_DIR}/predicted_vs_actual_sales.png'
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"Prediction chart saved as {plot_path}")
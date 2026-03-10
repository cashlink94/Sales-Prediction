import pandas as pd
import os

# Example: generate a sample DataFrame (replace with your actual code)
df = pd.DataFrame({
    "Date": pd.date_range(start="2026-01-01", periods=10, freq="D"),
    "Sales": [100, 150, 200, 130, 170, 220, 190, 210, 230, 250]
})

# Ensure the data folder exists
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

# Save the CSV
file_path = os.path.join(data_dir, "sales_prediction_data.csv")
df.to_csv(file_path, index=False)

print(f"CSV successfully saved at {file_path}")
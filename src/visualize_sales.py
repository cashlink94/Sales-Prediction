import matplotlib
matplotlib.use('TkAgg')  # Windows-friendly backend

import pandas as pd
import matplotlib.pyplot as plt

# Load extended dataset
df = pd.read_csv('data/sales_prediction_data_extended.csv', parse_dates=['Date'])

# Sort by date
df = df.sort_values('Date')

# Plot sales over time
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Sales'], color='blue', linewidth=2)

plt.title("Sales Trend Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.grid(True)

plt.tight_layout()

# Save the chart as PNG
plt.savefig('data/sales_trend.png')
print("Chart saved as sales_trend.png in the data folder")

# Show the chart (optional)
plt.show()
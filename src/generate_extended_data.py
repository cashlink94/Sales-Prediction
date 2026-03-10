import pandas as pd
import numpy as np

# Step 1: Load the existing dataset
df = pd.read_csv('data/sales_prediction_data.csv', parse_dates=['Date'])

# Step 2: Number of extra days to generate
extra_days = 365  # 1 year of extra data

# Step 3: Get the last date from the dataset
last_date = df['Date'].max()

# Step 4: Create new future dates
new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=extra_days)

# Step 5: Analyze the existing sales pattern
mean_sales = df['Sales'].mean()
std_sales = df['Sales'].std()

# Weekly pattern (sales may change depending on weekday)
weekday_avg = df.groupby(df['Date'].dt.dayofweek)['Sales'].mean()

# Step 6: Generate new sales values
new_sales = []

for date in new_dates:
    base = weekday_avg[date.weekday()]
    simulated = np.random.normal(loc=base, scale=std_sales)
    simulated = max(0, simulated)  # prevent negative sales
    new_sales.append(simulated)

# Step 7: Create dataframe for new data
df_new = pd.DataFrame({
    'Date': new_dates,
    'Sales': new_sales
})

# Step 8: Combine old and new data
df_extended = pd.concat([df, df_new], ignore_index=True)

# Step 9: Save extended dataset
df_extended.to_csv('data/sales_prediction_data_extended.csv', index=False)

print("Extended dataset saved successfully!")
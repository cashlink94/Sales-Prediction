import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Load new data
new_data = pd.read_csv("../data/sales_prediction_data.csv")
X_new = new_data[['units_sold', 'price']]

# Predict
predictions = model.predict(X_new)
print("Predicted revenue:", predictions)
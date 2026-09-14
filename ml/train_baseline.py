
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv('data/historical_sales_sample.csv')
data['day'] = pd.to_datetime(data['date']).dt.dayofyear

X = data[['day']]
y = data['quantity']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'ml/models/footfall_model.joblib')
print("Model trained and saved successfully!")

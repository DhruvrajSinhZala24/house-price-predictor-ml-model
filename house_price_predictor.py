# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Load the dataset
data = pd.read_csv('data.csv')  # Replace with the correct path if needed

# ✅ Select only the existing numeric columns
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
          'floors', 'waterfront', 'view', 'condition', 
          'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']]

y = data['price']

# ✅ Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Initialize models
lin_reg = LinearRegression()
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# ✅ Fit models
lin_reg.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)

# ✅ Make predictions
lin_pred = lin_reg.predict(X_test)
rf_pred = rf_reg.predict(X_test)

# ✅ Evaluate models
lin_mse = mean_squared_error(y_test, lin_pred)
lin_r2 = r2_score(y_test, lin_pred)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# ✅ Display evaluation metrics
print("\n--- Model Performance ---")
print(f"Linear Regression - MSE: {lin_mse:.2f}, R²: {lin_r2:.4f}")
print(f"Random Forest - MSE: {rf_mse:.2f}, R²: {rf_r2:.4f}")

# ✅ Plot actual vs predicted prices for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, color='blue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Ideal Fit Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()
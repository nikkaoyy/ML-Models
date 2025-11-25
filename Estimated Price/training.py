# Step 1: Import libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 2: Load dataset
data = pd.read_csv('data/housing_data.csv')  # data path

# Step 3: Preprocess data
x = data.drop('price', axis=1)  # Input features
y = data['price']  # Output variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression() # Initialize Linear Regression model
model.fit(x_train, y_train) # Train the model with training data


# Step 5: Make predictions and evaluate

y_pred = model.predict(x_test) # Predict prices on test set
print('Predicted Prices:', y_pred) # Print predicted prices

# Accuracy evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}') # Print Mean Squared Error

# Step 6: Visualize results
# Graphical representation of Actual vs Predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.legend(['Ideal Prediction', 'Predicted Prices'])
plt.show()

# Step 7: Save the model
import joblib
joblib.dump(model, 'model/linear_regression_model.pkl') # Save the trained model in pickle format

#Step 8: Save predictions in csv file

# Guardar como CSV con pandas (MÁS LIMPIO)
predictions_df = pd.DataFrame({
    'Area_m2': x_test['area_m2'].values,
    'Actual_Price': y_test.values,
    'Predicted_Price': y_pred,
    'Error': y_test.values - y_pred,
    'Absolute_Error': np.abs(y_test.values - y_pred),  # ✅ Siempre positivo
    'Error_Percent': ((y_test.values - y_pred) / y_test.values * 100)  # ✅ Porcentaje
})

predictions_df.to_csv('predictions/full_predictions.csv', index=False)
print("Predictions saved with the context")
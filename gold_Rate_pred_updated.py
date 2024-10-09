import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

import matplotlib.pyplot as plt
dataset=pd.read_csv(r'/workspaces/Gold_Rate_prediction_App/Gold_Rate.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
y_pred


plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Gold rate vs Year (Training set)')
plt.xlabel('Year of Gold rate')
plt.ylabel('Gold rate')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Gold rate vs Year (Training set)')
plt.xlabel('Year of Gold rate')
plt.ylabel('Gold rate')
plt.show()


y_2021 = regressor.predict([[2021]])
y_2025 = regressor.predict([[2025]])
print(f"Predicted Gold rate  for 2021 year : {y_2021[0]:,.2f}")
print(f"Predicted Gold rate for 2025 year : {y_2025[0]:,.2f}")

# Check model performance
bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

print(dataset)


# Save the trained model to disk
filename = 'gold_rate_pred_updated.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as gold_rate_pred_updated.pkl")

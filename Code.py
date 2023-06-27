import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Perform Random Forest Regression
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X, y)
predicted_y = regressor.predict(X)

# Visualize the results
plt.scatter(range(100), y, color='blue', label='Actual')  #100: number of samples
plt.scatter(range(100), predicted_y, color='red', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Crop Yield')
plt.title('Random Forest Regression')
plt.legend()
plt.show()

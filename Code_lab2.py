import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\Arav\\Desktop\\3rd Sem\\ML\\DS1.csv')

X = df['X'].values
Y = df['Y'].values

mean_x = np.mean(X)
mean_y = np.mean(Y)

numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x) ** 2)

slope = numerator / denominator
intercept = mean_y - slope * mean_x

def predict(x):
    return slope * x + intercept

Y_pred = np.array([predict(x) for x in X])
mean_squared_error = np.mean((Y - Y_pred) ** 2)
total_sum_of_squares = np.sum((Y - mean_y) ** 2)
r2 = 1 - (mean_squared_error / total_sum_of_squares)

print(f"Slope (Coefficient): {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

new_data = np.array([6, 7, 8])
new_predictions = np.array([predict(x) for x in new_data])

print("Predictions for new data:")
for i in range(len(new_data)):
    print(f"X = {new_data[i]}, Predicted Y = {new_predictions[i]:.2f}")

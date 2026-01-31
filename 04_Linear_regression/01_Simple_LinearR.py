import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv('height_weight_linear_data.csv')

X = df[['Weight']]
y = df['Height']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=51)

#Standardization 

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test) # Do not fit to prevent data leakage

# Initialzing the model 

regression = LinearRegression(n_jobs=-1)

regression.fit(X_train, y_train)
print(regression.coef_)
print(regression.intercept_)

#Prediction of Test data

y_pred = regression.predict(X_test)

#Checkng performance
mse = mean_squared_error(y_test, y_pred)
print(mse)

mae = mean_absolute_error(y_test, y_pred)
print(mae)

print(np.sqrt(mse))

print(regression.predict(scalar.transform([[84]])))

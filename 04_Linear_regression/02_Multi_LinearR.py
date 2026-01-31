import pandas as pd

df = pd.read_csv('economic_index.csv')
print(df.head(2))

df.drop(['Unnamed: 0', 'year', 'month'], axis=1, inplace=True)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(X.head(2))
print(y.head(2))

# Segregate Test and Train data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=11)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

regression = LinearRegression(n_jobs=-1)
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(mse)

mae = mean_absolute_error(y_test, y_pred)
print(mae)

import numpy as np

rmse = np.sqrt(mse)
print(rmse)
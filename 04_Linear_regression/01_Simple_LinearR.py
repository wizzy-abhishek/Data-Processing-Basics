import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('height_weight_linear_data.csv')

X = df[['Weight']]
y = df['Height']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=51)


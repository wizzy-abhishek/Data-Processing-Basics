import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('height_weight_linear_data.csv')

X = df[['Weight']]
y = df['Height']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=51)

#Standardization 

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test) # Do not fit to prevent data leakage
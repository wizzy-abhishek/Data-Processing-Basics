from sklearn.preprocessing import LabelEncoder
import pandas as pd

encoder = LabelEncoder()

df = pd.DataFrame({
    'colors':['Red', 'Green', 'Gold', 'Grey', 'Yellow', 'Baige', 'Black', 'Pink']
})

encoded = encoder.fit_transform(df[['colors']])

print(encoded)

encoder.fit(["Purple"])
encoder.transform(["Purple"])
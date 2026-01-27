from sklearn.preprocessing import OneHotEncoder
import pandas as pd

encoder = OneHotEncoder()

df = pd.DataFrame({
    'color':['Red', 'Blue', 'Green', 'Yellow', 'Red', 'Grey']
})

print(df[['color']])

encoded = encoder.fit_transform(df[['color']]).toarray()
print(encoded)

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
print(encoded_df)

print(encoder.transform([['Yellow']]).toarray())

df1 = pd.concat([df, encoded_df], axis=1)
print(df1)
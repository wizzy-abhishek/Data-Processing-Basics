from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

df = pd.DataFrame({
    'size':['medium','small', 'small', 'large']
})

encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])

encoded = encoder.fit_transform(df[['size']])
print(encoded)

encoder.fit([['extra large', 3]])
print(encoder.transform([['extra large']]))
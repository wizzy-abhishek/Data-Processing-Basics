import pandas as pd
import numpy as np
from sklearn.utils import resample

np.random.seed(121)

n_sample = 1000
class_0_ratio = 0.9

n_class_0 = int(n_sample * class_0_ratio)
n_class_1 = n_sample - n_class_0

class_0 = pd.DataFrame({
    'feature_1':np.random.normal(loc=0, scale=1, size=n_class_0), # Creates a normal/gaussian distribution, loc => mean, scale => standard deviation 
    'feature_2':np.random.normal(loc=0, scale=1, size=n_class_0),
    'target':[0]*n_class_0
})

class_1 = pd.DataFrame({
    'feature_1':np.random.normal(loc=1, scale=1, size=n_class_1), 
    'feature_2':np.random.normal(loc=1, scale=1, size=n_class_1),
    'target':[1]*n_class_1
})

df = pd.concat([class_1, class_0]).reset_index(drop=True)

df_minority = df[df['target'] == 1]
df_majority = df[df['target'] == 0]

df_majority_downsampled = resample(
    df_majority, replace=False, n_samples=len(df_minority), random_state=42
)

df_downsampled = pd.concat([df_majority_downsampled, df_minority]).reset_index(drop=True)
print(df_downsampled)
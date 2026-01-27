import numpy as np

list_ages = [1,2,4,33,12,143,24,156,2,10,14,25,15,44,2,104,21,45,27,36,20,18,56,72]

min, q1, median, q3, max= np.quantile(list_ages, [0,0.25,0.5,0.75,1])

print(f"Min: {min}\
    Q1: {q1}\
    Median: {median}\
    Q3: {q3}\
    Max: {max}")

IQR = q3 - q1
print(IQR)

lower_fence = q1 - 1.5 * IQR
higer_fence = q3 + 1.5 * IQR

print(f"Lower Fence: {lower_fence}")
print(f"Higer fence: {higer_fence}")
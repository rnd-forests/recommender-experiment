import os
import numpy as np
import pandas as pd
from rs import parse_config
import matplotlib.pyplot as plt


def load_data_into_dataframe(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise RuntimeError('Cannot find the dataset.')
    df = pd.read_csv(path, sep=',', names=['UserID', 'ItemID', 'Rating'])
    return df


path = parse_config(section='Path', key='data') + '/views.csv'
df = load_data_into_dataframe(path)

print(df.shape)
print()
print(df.info())
print()
print(df.describe())

plt.figure(figsize=(15, 9))
bins = np.sort(df['Rating'].unique())
plt.hist(df['Rating'], bins=bins, facecolor='blue', log=True,
         alpha=0.85, edgecolor="white", label="Post Views Histogram")
plt.xlabel("View Count")
plt.ylabel("Log Count")
plt.legend()
plt.grid(True, alpha=0.25, linestyle="dashed")
plt.show()

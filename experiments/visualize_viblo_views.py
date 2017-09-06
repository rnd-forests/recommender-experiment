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


plt.figure(figsize=(15, 9))

plt.subplot(211)
path = parse_config(section='Path', key='data') + '/views.csv'
original_data = load_data_into_dataframe(path)

print("Original Data (With IP addresses)")
print(original_data.shape)
print(original_data.info(), "\n")
print(original_data.describe(), "\n")

bins = np.sort(original_data['Rating'].unique())
plt.hist(original_data['Rating'], bins=bins, facecolor='blue', log=True,
         alpha=0.85, edgecolor="white", label="Post Views (With IP Addresses)")
plt.xlabel("View Count")
plt.ylabel("Log Count")
plt.legend()
plt.grid(True, alpha=0.25, linestyle="dashed")


plt.subplot(212)
path = parse_config(section='Path', key='data') + '/views_without_ips.csv'
ip_free_data = load_data_into_dataframe(path)

print("Modified data (Without IP addresses)")
print(ip_free_data.shape)
print(ip_free_data.info(), "\n")
print(ip_free_data.describe())

bins = np.sort(ip_free_data['Rating'].unique())
plt.hist(ip_free_data['Rating'], bins=bins, facecolor='blue', log=True,
         alpha=0.85, edgecolor="white", label="Post Views (Without IP Addresses)")
plt.xlabel("View Count")
plt.ylabel("Log Count")
plt.legend()
plt.grid(True, alpha=0.25, linestyle="dashed")

plt.show()

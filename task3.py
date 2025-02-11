#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd

df = pd.read_csv("C:\\Users\\Admin\\Downloads\\Task 3 dataset\\Mall_Customers.csv")
print(df.head())
print("\nNull values\n",df.isnull().sum())
print("\nNumber of duplicates:",df.duplicated().sum())
df = df.drop_duplicates()

import statistics as st
print("\nSummary Statistics","\nMean of annual income:",st.mean(df['Annual Income']))
print("Mean of Spending score:",st.mean(df['Spending Score']))
print("Range of numeric columns:\n",df.agg(["min","max"]))

print("\nClustering")

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss1 = []
wcss2 = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(df[["Annual Income"]])
    wcss1.append(kmeans.inertia_)

print("\nElbow method")
plt.plot(range(1,11),wcss1)
plt.title("Within clusters sum of squares for annual income")
plt.ylabel("Within clusters sum of squares")
plt.xlabel("Clusters")
plt.grid(True)
plt.show()

optimal_k=3
km = KMeans(n_clusters=optimal_k, random_state=1)
cluster_labels = km.fit_predict(df[["Annual Income"]])
df["Cluster"]= cluster_labels
df.head(10)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data = df[["Annual Income","Spending Score","Cluster"]]
pca_df = pca.fit_transform(data)
scatter = plt.scatter(pca_df[:,0],pca_df[:,1],c=cluster_labels)
plt.title("Scatter plot after PCA")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.colorbar(scatter,label="Cluster label")
plt.show()

import seaborn as sns

sns.pairplot(data, hue="Cluster", diag_kind="hist")
plt.show()

print("\nAs the annual income increases, there is a definite pattern of increased spending, however \nfor even low income there are high spending scores indicating there might not be a complete \nrelationship")


# In[ ]:





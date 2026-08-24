# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the customer dataset and select Annual Income (k$) and Spending Score (1-100) as features.

2.Standardize the selected features and apply K-Means clustering with 5 clusters.

3.Assign each customer to a cluster and find the cluster centers.

4.Display the customer segments and visualize them using a scatter plot.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: DHAYAL ABISEK R
RegisterNumber:  212225060061  
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("/content/Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
model = KMeans(n_clusters=5, random_state=42, n_init=10)

# Fit the model
df["Cluster"] = model.fit_predict(X_scaled)

# Display clusters
print(df[["CustomerID", "Annual Income (k$)",
          "Spending Score (1-100)", "Cluster"]])

# Display cluster centers
print("\nCluster Centers:")
print(scaler.inverse_transform(model.cluster_centers_))

# Plot clusters
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()
```
## Output:
```
     CustomerID  Annual Income (k$)  Spending Score (1-100)  Cluster
0             1                  15                      39        4
1             2                  15                      81        2
2             3                  16                       6        4
3             4                  16                      77        2
4             5                  17                      40        4
..          ...                 ...                     ...      ...
195         196                 120                      79        1
196         197                 126                      28        3
197         198                 126                      74        1
198         199                 137                      18        3
199         200                 137                      83        1

[200 rows x 4 columns]

Cluster Centers:
[[55.2962963  49.51851852]
 [86.53846154 82.12820513]
 [25.72727273 79.36363636]
 [88.2        17.11428571]
 [26.30434783 20.91304348]]
```
<img width="573" height="455" alt="download (1)" src="https://github.com/user-attachments/assets/e83a5bbb-db40-48c5-96f6-a8023304d8a3" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

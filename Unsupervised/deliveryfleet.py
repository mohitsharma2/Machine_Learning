"""
Q1. (Create a program that fulfills the following specification.)
deliveryfleet.csv


Import deliveryfleet.csv file

Here we need Two driver features: mean distance driven per day (Distance_feature) 
and the mean percentage of time a driver was >5 mph over the speed limit (speeding_feature).

Perform K-means clustering to distinguish urban drivers and rural drivers.
Perform K-means clustering again to further distinguish speeding drivers 
from those who follow speed limits, in addition to the rural vs. urban division.
Label accordingly for the 4 groups.

"""
# 1) Perform K-means clustering to distinguish urban drivers and rural drivers.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("deliveryfleet.csv")

data.isnull().any(axis=0)

data.info()

features=data.iloc[:,[1,2]].values

plt.scatter(features[:,0], features[:,1])
plt.show()


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features) 
    wcss.append(kmeans.inertia_)
    
print(wcss)

#Now plot it        
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset using 2 clusters


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)

pred_cluster = kmeans.fit_predict(features) 

print(pred_cluster) # Its the cluster id with 0 and 1 

#There we consider 0 rural for  drivers and 1 for Urban drivers.

print(features[pred_cluster == 0])

print(len(features[pred_cluster == 0]))

print(features[pred_cluster == 1])

print(len(features[pred_cluster == 1]))

print(features[pred_cluster == 0, 0])

print(features[pred_cluster == 0, 1])

# Visualising the clusters

plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Rural')

plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Urban')

print(kmeans.cluster_centers_)  # There are two points for 2 centroid

print(kmeans.cluster_centers_[:, 0]) 

print(kmeans.cluster_centers_[:, 1])

# Central Location of the Cluster == Centroid

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')


plt.title('Clusters of datapoints')
plt.xlabel('Distance')
plt.ylabel('Speed')
plt.legend()
plt.show()



#2) Perform K-means clustering again to further distinguish speeding drivers 
#from those who follow speed limits, in addition to the rural vs. urban division.
#Label accordingly for the 4 groups.


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)

pred_cluster = kmeans.fit_predict(features) 

print(pred_cluster) # Its the cluster id with 0,1,2,3 

print(features[pred_cluster == 0])

print(features[pred_cluster == 1])

print(features[pred_cluster == 2])

print(features[pred_cluster == 3])


# Visualising the clusters
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Follow speed in Rural')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Not follow speed in Urban')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Follow speed in Urban')
plt.scatter(features[pred_cluster == 3, 0], features[pred_cluster == 3, 1], c = 'olive', label = 'Not follow speed in Rural')


print(kmeans.cluster_centers_)  # There are two points for 2 centroid   Rural follow speed

print(kmeans.cluster_centers_[:, 0]) 
print(kmeans.cluster_centers_[:, 1])
# Central Location of the Cluster == Centroid

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')


plt.title('Clusters of datapoints')
plt.xlabel('Distance')
plt.ylabel('Speed')
plt.legend()
plt.show()





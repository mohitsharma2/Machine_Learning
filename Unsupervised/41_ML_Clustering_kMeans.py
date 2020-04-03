# K-Means Clustering
"""
Now the focus has changed to  UnSupervised Machine Learning 
That means you are training your model with features only
It might happen that your dataset does have label in it, 
but you are not using it
Also it might happen that you dont want to predict something 
but group your data into groups
And these groups are known as clusters

There are two main algo for clustering
1. k-Means
2. DbSCAN

Use Cases
K-Means is widely used for many applications.

Image Segmentation or Grouping
Gene Segementation Data
News Article Segmentation
Clustering Languages
Species Segmentation 
Fraud/Anomally Detection

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset (Bivariate Data Set with 3 Clusters)
dataset = pd.read_csv('data/xclara.csv')

features = dataset.iloc[:, [0, 1]].values


# There is no Categorial data
# There is no Missing data 
# There is no different scale data


#Scatter all these data points on the matplotlib
# It seems as a human that it will have 3 clusters or groups
plt.scatter(features[:,0], features[:,1])
plt.show()


# Fitting K-Means to the dataset using 3 clusters
from sklearn.cluster import KMeans
# Since we have seen the visual, we have told the algo to make 3 cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

pred_cluster = kmeans.fit_predict(features) # We have only passed features 

print(pred_cluster) # Its the cluster id with 0 1 and 2 



# How to find which points from features falls with cluster id 0
print(features)
print(pred_cluster)
print(pred_cluster == 0  )
# will gives a True False Array, we will use boolean indexing concept
print(features[pred_cluster == 0])
print(len(features[pred_cluster == 0]))


# How to find which points from features falls with cluster id 1
print(features)
print(pred_cluster == 1  )
# will gives a True False Array, we will use boolean indexing concept
print(features[pred_cluster == 1])
print(len(features[pred_cluster == 1]))
print(len(features[pred_cluster == 0]))
print(len(features[pred_cluster == 2]))


# How to find which points from features falls with cluster id 2
print(features)
print(pred_cluster == 2 )
# will gives a True False Array, we will use boolean indexing concept
print(features[pred_cluster == 2])
print(len(features[pred_cluster == 2]))


# Will print V1
print(features[pred_cluster == 0, 0])  # 0 is for V1

# Will print V2
print(features[pred_cluster == 0, 1])  # 1 is for V2


# Visualising the clusters
#plt.scatter(features[:,0][y_kmeans == 0], features[:,1][y_kmeans == 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Cluster 1')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Cluster 2')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Cluster 3')


# Centroid of the clusters 
"""
Explain the concept of Center of Mass = Centroid 

                     Sum of x coordinates 
x of centroid   =   -----------------------
                    Total number of values 
                    

                     Sum of y coordinates 
y of centroid   =   -----------------------
                    Total number of values                     

"""
print(kmeans.cluster_centers_)  # There are three points for 3 centroid

print(kmeans.cluster_centers_[:, 0]) # x 
print(kmeans.cluster_centers_[:, 1]) # y 

# Central Location of the Cluster == Centroid
# Centroid = Avg of x coordinate and Avg of y coordinate for lets assume Cluster 1
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Cluster 1')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Cluster 2')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')



plt.title('Clusters of datapoints')
plt.xlabel('X Cordindates')
plt.ylabel('Y Cordinates')
plt.legend()
plt.show()


# How does the k-Means Algorithm Works and the math behind it ?

# How to decide programatically that there would be 3 clusters ?

"""
https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
How to pick the initial centroids?  == Randomly
What kind of data would you like?   == Uniform Points
Display data in 2D Plane 
Add Centroid - Randomly places a point as a centroid
Since we need to have a centroid, but this is placed randomly 
which is not the right position
Add another Centroid, which now divides the data points into 2 parts or divisions
It uses a logic wherein it draws a straight line between the two centroid 
and bisects centrally at 90 degree
Add Centroid again, this will divide into 3 parts, but all the 3 centroid 
are not at the right place, 
since they are placed randomly

Click GO to fix the data points and starting points of the centroid.
Now Click Update Centroids again and again, so that it would change the boundary division
Since it would recalculate the centroid
Now because of recalulation, some of the points will cross boundary of the older divisions
Now Reassign Points based on new division , this should change the centroid again 
Again and Again Reassign and Update Centroid
This loops goes on, you will come to a point that will not change the centroid, 
that is the actual centroid
That would be the final clusters 
voronoi diagram is the technical name of the image drawn
"""

# Show voronoi.png

"""
What does this centroid infer, that this points has average distance 
minimum from all its points in its boundary

Jaipur Population Density - can be visualised in this way to setup 3 health centers 
So that the avarge customers needs to travel the least to reach the health center 

London Example of Ambulance to be placed at which location
"""


# How to find the optimal number of clusters ??
# Understand a concept WCSS ( Within Cluster Sum of Squares )
"""
open WCSS.jpg
To calculate the WCSS of first cluster 
First calculate the Sum of square of each distance from the centroid

Now calculate for the other centroid

Then Add all the three
So we need to have nested loops to do that
Outer loop for the number of centroid and the inner loop for every point of that cluster

Map this logic with the image

When the Clusters are less then WCSS would be more since distance from the centroid 
would be more
If we increase the number of clusters then the WCSS would start becoming less
One level will come, when you increase the number of centroid, but WCSS remains same

Lets see the code now
"""

# Outer Loop to iterate through the number of clusters
for i in range(1, 11):
    print(i)
    # Within each iteration we need to now find the sum
    
    
# Using the elbow method to find the optimal number of clusters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/xclara.csv')

features = dataset.iloc[:, [0, 1]].values
   

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)  # we have not used the fit_predict
    #print("Cluster " + str(i) +  "  =  " + str(kmeans.inertia_))
    wcss.append(kmeans.inertia_)     # ( calculates wcss for a cluster )
    
print(wcss)

#Now plot it        
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

"""
Now plot it and try to find Elbow, The elbow points is the right number of cluster 
Another way to detect the optimal number of cluster is Silhouette method
Run the Silhouette Code, when the Silhouette score is maximum, 
it is the optimal number of clusters 
There is another way to find the right number of clusters
Known as Silhouette score
Run the Silhouette Code, when the Silhouette score is maximum, it is the optimal number of clusters 
We need at least 2 clusters to Silhouette calculate score 
"""



#Silhouette Score calculation to find optimal number of clusters
"""
The Silhouette Coefficient is defined for each sample and is composed of two scores:

a: The mean distance between a sample and all other points in the same class.
b: The mean distance between a sample and all other points in the next nearest cluster.
The Silhouette Coefficient s for a single sample is then given as:

score = (b - a)/max(a, b)
"""

from sklearn.metrics import silhouette_score

for n_clusters in range(2, 11):
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(features)
    centers = clusterer.cluster_centers_

    score = silhouette_score (features, preds, metric='euclidean')
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", score)
    
#With above values, we can clearly see that the optimal cluster values are 3




#Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1
#silhouette_score requires more than 1 cluster labels
#So if you call the silhouette score method for jsut one cluster, it gives error
"""
Advantages of Silhouette Score
The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
Scores around zero indicate overlapping clusters.
The score is higher when clusters are dense and well separated, 
which relates to a standard concept of a cluster.
Drawbacks
The Silhouette Coefficient is generally higher for convex clusters than
other concepts of clusters, such as density based clusters like those 
obtained through DBSCAN.
"""




"""
Measuring Performance of K-means
Homogeneity and Completeness – 
If you have pre-existing class labels that you’re trying to duplicate 
with k-means clustering, 
you can use two measures: homogeneity and completeness.

Homogeneity means all of the observations with the same class label are 
in the same cluster.
Completeness means all members of the same class are in the same cluster.
Scikit-Learn (Python) has an excellent write-up on these two measures.

"""
#We can turn those concept as scores homogeneity_score and completeness_score. Both are bounded below by 0.0 and above by 1.0 (higher is better):

from sklearn import metrics

labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

metrics.homogeneity_score(labels_true, labels_pred)  


metrics.completeness_score(labels_true, labels_pred) 

#Their harmonic mean called V-measure is computed by v_measure_score
metrics.v_measure_score(labels_true, labels_pred) 

#All calculated together
metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)



#https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

#http://www.learnbymarketing.com/methods/k-means-clustering/


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




Q2. (Create a program that fulfills the following specification.)
tshirts.csv


T-Shirt Factory:

You own a clothing factory. You know how to make a T-shirt given the height 
and weight of a customer.

You want to standardize the production on three sizes: small, medium, and large. 
How would you figure out the actual size of these 3 types of shirt to better 
fit your customers?

Import the tshirts.csv file and perform Clustering on it to make sense out of 
the data as stated above.



Q.3. Code Challenge - 
 This is a pre-crawled dataset, taken as subset of a bigger dataset 
 (more than 4.7 million job listings) that was created by extracting data 
 from Monster.com, a leading job board.
 
 monster_com-job_sample.csv
 
 Remove location from Organization column?
 Remove organization from Location column?
 
 In Location column, instead of city name, zip code is given, deal with it?
 
 Seperate the salary column on hourly and yearly basis and after modification
 salary should not be in range form , handle the ranges with their average
 
 Which organization has highest, lowest, and average salary?
 
 which Sector has how many jobs?
 Which organization has how many jobs
 Which Location has how many jobs?
"""


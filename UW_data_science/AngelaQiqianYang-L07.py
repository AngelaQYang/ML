# -*- coding: utf-8 -*-
"""
L07 assignment: unsupervised learning with k-means 
Created on Fri Nov 16 15:09:21 2018
Question: Is it an add? 
@author: Angela Qiqian Yang df.info()
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data'
data = pd.read_csv(url, header=None)
print (data.info())

# pay attention that missing data '?' has a few blank string in it
FlagMissing = (data.loc[:, 0] == '   ?')
## 1558 is the last column name, AKA the ad observed value
data[FlagMissing].loc[:, 1558].value_counts()

##1.  count the missing value 
test = data.replace(to_replace="   ?", value=float("NaN")) # the first and second columns
test = test.replace(to_replace="     ?", value=float("NaN")) #the third column
test = test.replace(to_replace="?", value=float("NaN")) #the fourth column
test.isnull().sum()

##2.  delete all null values, since they are unknown data and not so much 27%
test_dropna = test.dropna(axis=0)
test_dropna.shape
test = test_dropna 

##3.  EDA
test.dtypes
test.loc[:, 0] = pd.to_numeric(test.loc[:, 0], errors='coerce')
test.loc[:, 1] = pd.to_numeric(test.loc[:, 1], errors='coerce')
test.loc[:, 2] = pd.to_numeric(test.loc[:, 2], errors='coerce')
test.loc[:, 3] = pd.to_numeric(test.loc[:, 3], errors='coerce')

plt.hist(test.loc[:, 0]) 
plt.xlabel('height')
plt.ylabel('frequency')

plt.hist(test.loc[:, 1]) 
plt.xlabel('width')
plt.ylabel('frequency')

plt.hist(test.loc[:, 2]) 
plt.xlabel('aspect of ratio')
plt.ylabel('frequency')

plt.hist(test.loc[:, 3]) 

my_data = test

## 3. encode categorical column 
my_data.loc[ my_data.loc[:, 1558] == 'nonad.', 1558] = 0
my_data.loc[ my_data.loc[:, 1558] == 'ad.', 1558] = 1

## 4. Clustering preparatipon. Compare distributions of the three dimensions
Points = my_data[[0, 1, 1558]]
Points = Points.rename(columns={1558 : 2}) # rename for the covenience in following codes
## use column 0,1,1558 (height, width, non-ad/ad) to do clustering 
plt.rcParams["figure.figsize"] = [8.0, 8.0] # Standard
plt.hist(Points.loc[:,0], bins = 20, color=[0, 0, 1, 0.5])
plt.hist(Points.loc[:,1], bins = 20, color=[1, 1, 0, 0.5])
plt.hist(Points.loc[:,2], bins = 20, color=[1, 1, 0, 1])
plt.title("Compare Distributions")
plt.show()

### 5. K-means 
# Create initial cluster centroids
ClusterCentroidGuesses = pd.DataFrame()
ClusterCentroidGuesses.loc[:,0] = [50, 200, 0]
ClusterCentroidGuesses.loc[:,1] = [100, 100, 0]
ClusterCentroidGuesses.loc[:,2] = [0, 1, 0]

def Plot2DKMeans(Points, Labels, ClusterCentroids, Title):
    for LabelNumber in range(max(Labels)+1):
        LabelFlag = Labels == LabelNumber
        color =  ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y'][LabelNumber]
        marker = ['s', 'o', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'][LabelNumber]
        plt.scatter(Points.loc[LabelFlag,0], Points.loc[LabelFlag,1],
                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
        plt.scatter(ClusterCentroids.loc[LabelNumber,0], 
                    ClusterCentroids.loc[LabelNumber,1], 
                    s=200, c="black", marker=marker)
    plt.title(Title)
    plt.show()

def KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2, NormD3):
    PointsNorm = Points.copy()
    ClusterCentroids = ClusterCentroidGuesses.copy()
    if NormD1:
        # Determine mean of 1st dimension
        mu1 = np.mean(Points.loc[:,0])
        # Determine standard deviation of 1st dimension
        sd1 = np.std(Points.loc[:,0])
        # Normalize 1st dimension of Points
        PointsNorm.loc[:, 0] = (Points.loc[:, 0] - mu1)/sd1
        # Normalize 1st dimension of ClusterCentroids
        ClusterCentroids.loc[:, 0] = (ClusterCentroids.loc[:, 0] - mu1)/sd1
    if NormD2:
        # Determine mean of 2nd dimension
        mu2 = np.mean(Points.loc[:,1])
        # Determine standard deviation of 2nd dimension
        sd2 = np.std(Points.loc[:,1])
        # Normalize 2nd dimension of Points
        PointsNorm.loc[:, 1] = (Points.loc[:, 1] - mu2)/sd2
        # Normalize 2nd dimension of ClusterCentroids
        ClusterCentroids.loc[:, 1] = (ClusterCentroids.loc[:, 1] - mu2)/sd2
    if NormD3:
        # Determine mean of 2nd dimension
        mu3 = np.mean(Points.loc[:,2])
        # Determine standard deviation of 2nd dimension
        sd3 = np.std(Points.loc[:,2])
        # Normalize 2nd dimension of Points
        PointsNorm.loc[:, 2] = (Points.loc[:, 2] - mu3)/sd3
        # Normalize 2nd dimension of ClusterCentroids
        ClusterCentroids.loc[:, 2] = (ClusterCentroids.loc[:, 2] - mu3)/sd3
    # Do actual clustering
    kmeans = KMeans(n_clusters=3, init=ClusterCentroids, n_init=1).fit(PointsNorm)
    Labels = kmeans.labels_
    ClusterCentroids = pd.DataFrame(kmeans.cluster_centers_)
    if NormD1:
        # Denormalize 1st dimension
        ClusterCentroids.loc[:, 0] = ClusterCentroids.loc[:, 0]*sd1 + mu1
    if NormD2:
        # Denormalize 2nd dimension
        ClusterCentroids.loc[:, 1] = ClusterCentroids.loc[:, 1]*sd2 + mu2
    if NormD3:
        # Denormalize 2nd dimension
        ClusterCentroids.loc[:, 2] = ClusterCentroids.loc[:, 2]*sd3 + mu3
    return Labels, ClusterCentroids



## 6. K-means 
NormD1=False
NormD2=False
NormD3=False
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2, NormD3)
Title = 'No Normalization'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)

NormD1=True
NormD2=False
NormD3=False
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2, NormD3)
Title = 'Normalization in first dimension'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)


NormD1=False
NormD2=True
NormD3=False
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2, NormD3)
Title = 'Normalization in second dimension'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)

NormD1=False
NormD2=False
NormD3=True
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2, NormD3)
Title = 'Normalization in third dimension'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)


NormD1=True
NormD2=True
NormD3=False
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2, NormD3)
Title = 'Normalization in the first and second dimension'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)


NormD1=True
NormD2=True
NormD3=True
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2, NormD3)
Title = 'Normalization in all three dimensions'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)



## 7. output the labeled data points 
labeled_points = pd.DataFrame()
labeled_points['Label'] = Labels
labeled_points['height'] = np.array(Points[0])
labeled_points['width'] = np.array(Points[1])
labeled_points['nonad.ORad.'] = np.array(Points[2])


## 8. Better visulization in 3D plot ---- not finihsed yet
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
fig = pyplot.figure()
ax = Axes3D(fig)
x = np.array(Points[0])
y = np.array(Points[1])
z = np.array(Points[2])
ax.scatter(x, y, z)


c_x = np.array(ClusterCentroids[0])
c_y = np.array(ClusterCentroids[1])
c_z = np.array(ClusterCentroids[2])
ax.scatter(c_x, c_y, c_z, s = 300, c = 'r', marker='*', label = 'Centroid')

ax.set_xlabel('height')
ax.set_ylabel('width')
ax.set_zlabel('non-ad/ad')

pyplot.show()
fig





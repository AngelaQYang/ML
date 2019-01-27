# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:25:47 2018
Assigment 08: Predictive Models 
@author: Angela Qiqian Yang 

"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier

# read the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data'
data = pd.read_csv(url, header=None)
print (data.info())

## 1. Replace missing numeric data
# look up missing values, and get to know data types of them
FlagMissing = (data.loc[:, 0] == '   ?')
## 1558 is the last column name, AKA the ad observed value
data[FlagMissing].loc[:, 1558].value_counts()
test = data.replace(to_replace="   ?", value=float("NaN")) # the first and second columns
test = test.replace(to_replace="     ?", value=float("NaN")) #the third column
test = test.replace(to_replace="?", value=float("NaN")) #the fourth column
test.isnull().sum()
#delete all null values, since they are unknown data and not so much 27%
test_dropna = test.dropna(axis=0)
test_dropna.shape
test = test_dropna 


## 2. Account for outlier values in numeric columns 
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
x = np.array(my_data[0])
LimitHi = np.mean(x) + 2*np.std(x)
LimitLo = np.mean(x) - 2*np.std(x)
FlagOutlier = (x < LimitLo) | (x > LimitHi)
print ('the histogram plot for the hours-per-week column is below:')
plt.hist(my_data.loc[~FlagOutlier, 0], bins = 10, color=[1, 1, 0, 0.5])
plt.hist(my_data.loc[FlagOutlier, 0], bins = 10, color=[0, 0, 1, 0.5])
plt.xlabel('length of the bridge')
plt.title("Compare Distributions between inliers and outliers")
plt.show()
print ('below are the value counts for outliers in LENGTH column')
print (my_data.loc[FlagOutlier, 0].value_counts())
# I decided to simply remove the outliers
my_data = my_data.loc[~FlagOutlier]


## 3. Normalize height values 
height =np.array(my_data[0])
# use min-max normalized as an example
height_mm = (height - np.min(height))/(np.max(height) - np.min(height)) 
# look at original data distributrions 
plt.hist(height, bins = 20, color=[0, 0, 0, 1])
plt.title("Original picture Height Distribution")
plt.xlabel('height of the bridge')
plt.show()
plt.hist(height_mm, bins = 20, color=[1, 1, 0, 1])
plt.title("MinMax-normalization of Original height")
plt.xlabel('height of the picture')
plt.show()


## 4. One-hot encode with categorical data 
my_data.loc[ my_data.loc[:, 1558] == 'nonad.', 1558] = 0
my_data.loc[ my_data.loc[:, 1558] == 'ad.', 1558] = 1



## 5. Specify an appropriate column as your expert label for a classification.
# I want to know the picture is advertisement or not, 
# so I specify the last column: advertisement or not as my expert label



## 6. Clustering preparatipon. Compare distributions of the three dimensions
Points = my_data[[0, 1, 1558]]
Points = Points.rename(columns={1558 : 2}) # rename for the covenience in following codes
## use column 0,1,1558 (height, width, non-ad/ad) to do clustering 
plt.rcParams["figure.figsize"] = [8.0, 8.0] # Standard
plt.hist(Points.loc[:,0], bins = 20, color=[0, 0, 1, 0.5])
plt.hist(Points.loc[:,1], bins = 20, color=[1, 1, 0, 0.5])
plt.hist(Points.loc[:,2], bins = 20, color=[1, 1, 0, 1])
plt.title("Compare Distributions")
plt.show()

### 7. K-means 
# Create initial cluster centroids
ClusterCentroidGuesses = pd.DataFrame()
ClusterCentroidGuesses.loc[:,0] = [50, 200,]
ClusterCentroidGuesses.loc[:,1] = [100, 100]
ClusterCentroidGuesses.loc[:,2] = [100, 100]

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
    kmeans = KMeans(n_clusters=2, init=ClusterCentroids, n_init=1).fit(PointsNorm)
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


NormD1=True
NormD2=True
NormD3=True
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2, NormD3)
Title = 'Normalization in all three dimensions'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)


## 8. add cluster label into my dataset 
my_data['Label'] = Labels


## 9. Split the data set into training and testing sets 
# normalization is very important in this dataset, 
# it is becuase height, width and ratio of aspect could be very different
def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        mX = min(X[:,i])
        Y[:,i] = (X[:,i] - mX) / (max(X[:,i]) - mX)
    
    return Y

def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data[ind_,:-1] # training features
	XX = data[ind,:-1] # testing features
	Y = data[ind_,-1] # training targets
	YY = data[ind,-1] # testing targests
	return X, XX, Y, YY

# split dataset into - 70% as training dataset, 30% as test dataset
r = 0.3 
dataset = np.array(my_data)
all_inputs = normalize(dataset[:,:3]) # inputs (features)
normalized_data = deepcopy(dataset)
normalized_data[:,:3] = all_inputs
X, XX, Y, YY = split_dataset(normalized_data, r)


## 10. Create a classification model for the expert label
# Decision Tree classifier
print ('\n\nDecision Tree classifier\n')
clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X, Y)


## 11. write out an csv to show the predicted value 
predicted_data = pd.DataFrame()
predicted_data['actual_ad'] = YY
predicted_data['predicted_ad'] = clf.predict(XX)
predicted_data.to_csv('AngelaQYang-L08-DataModelBuild-output.csv')


## 12. accuracy rate 
print ('accuracy rate is: ')
print (np.corrcoef(clf.predict(XX), YY))


## 13. comment 
'''
I applied internet advertisment picture dataset. 
The task is to detect whether the picture is an online advertisement. 
So the expert label would be the last column of the dataset: non-ad/ad. 
I deleted the 27% missing data, and outlier data. 
I applied the first three columns: height, width, and aspect of ratio to do clustering and modeling.
Because all three values I used are in very different range, so I did normalization for all three features. 
The K-means  cluster number would be only 2, because the final task is a yes/no classification question. 
I split the original dataset into 70% training and 30% test dataset.
I applied decision tree to build the predictive model, because it is a classifier problem. 
The final result is pretty accurate is because decision tree performance very well in this question. 
'''











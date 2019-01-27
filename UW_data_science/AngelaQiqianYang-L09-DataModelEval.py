# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:45:44 2018
L09 Assignment: Accuracy Measures
Data Source: 
@author: Anela Qiqian Yang
"""

import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB

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


my_data = test
x = np.array(my_data[0])
LimitHi = np.mean(x) + 2*np.std(x)
LimitLo = np.mean(x) - 2*np.std(x)
FlagOutlier = (x < LimitLo) | (x > LimitHi)
# I decided to simply remove the outliers
my_data = my_data.loc[~FlagOutlier]

## 3. Normalize height values 
# normalization is very important in this dataset, 
# it is becuase height, width and ratio of aspect could be very different
height =np.array(my_data[0])
width = np.array(my_data[1])
ratio = np.array(my_data[2])
# use min-max normalized as an example
height_mm = (height - np.min(height))/(np.max(height) - np.min(height)) 
width_mm = (width - np.min(width))/(np.max(width) - np.min(width)) 
ratio_mm = (ratio - np.min(ratio))/(np.max(ratio) - np.min(ratio)) 

my_data[0] = height_mm
my_data[1] = width_mm
my_data[2] = ratio_mm


## 4. One-hot encode with categorical data 
my_data.loc[ my_data.loc[:, 1558] == 'nonad.', 1558] = 0
my_data.loc[ my_data.loc[:, 1558] == 'ad.', 1558] = 1


## 5. Specify an appropriate column as your expert label for a classification.
# I want to know the picture is advertisement or not, 
# so I specify the last column: advertisement or not as my expert label


## 6. Split the data set into training and testing sets 
TestFraction = 0.3 
print ("Test fraction is chosen to be:", TestFraction)

print ('\nSimple approximate split:')
isTest = np.random.rand(len(my_data)) < TestFraction 

TrainSet = my_data[~isTest]
TestSet = my_data[isTest] # should be 249 but usually is not
print ('Test size should have been ', 
       TestFraction*len(my_data), "; and is: ", len(TestSet)) 

Input = my_data.columns[:3] # the first 1557 columns are inputs
Target = my_data.columns[1558] # the last column is the target: is advertisment or not 

X = TrainSet.loc[:, Input]
Y = TrainSet.loc[:, Target]
XX = TestSet.loc[:, Input]
YY = TestSet.loc[:, Target]


## 7. Create different classification models for the expert label

# Naive Bayes classifier
print ('\n\nNaive Bayes classifier\n')
nbc = GaussianNB() # default parameters are fine
nbc.fit(X, Y)
print (np.corrcoef(nbc.predict(XX), YY))

# K nearest neighbors classifier
print ('\n\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X, Y)
print (np.corrcoef(knn.predict(XX), YY))

# Decision Tree classifier
print ('\n\nDecision Tree classifier\n')
clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X, Y)
print (np.corrcoef(clf.predict(XX), YY))


## 8. Accuracy measures for your predicted values，
# decide to use KNN model as an example

BothProbabilities = knn.predict_proba(XX)
#print (BothProbabilities)

## Confusion Matrix (specify the probability threshold)
# actual probability
probabilities = BothProbabilities[:,1] #the second value is the "positive"-probabilty - P of being 1. 
# threshold 
print ('\nConfusion Matrix and Metrics')
Threshold = 1 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", Threshold)
# the actual prediction
predictions = (probabilities > Threshold).astype(int)
# confusion matrix 
CM = confusion_matrix(YY, predictions)
print (CM) 
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)


## ROC Analysis with AUC score
fpr, tpr, th = roc_curve(YY, probabilities)
AUC = auc(fpr, tpr)
print ('\nAUC is')
print (AUC)

# plotting
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()



## 9. Comments explaining the code blocks
'''
I applied online advertisement dataset in this assignment. 
The goal is to detect if the online picture is an advertisement or not. 
I prepared the data for model construction. 
The script explored three models: Native Bayes classifer, K Nearest Neighbors Classifier and Decision Tree Classifier 
I decided to use KNN model for the following CM analysis, because it is easier to understand than decision tree, and also has better performance than Native Bayes classifier. 
I set threshold as 1.0, and print out the CM. The result is as expected: no True Positive and False Positive. 
It means the threshold are so high that we can't have any prediction as positive value.
AUC for KNN model is 0.94. 
It is suspciously high. The reason could be within model inputs: picture height, width, and ascpect ratio, there could be any proxy value. 
The ROC curve was plotted in this scripts. 

'''







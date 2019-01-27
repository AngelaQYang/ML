# -*- coding: utf-8 -*-
"""
Created on Thu Nov. 8 23:43:59 2018
data: Pittsburgh Bridges Data Set with Version 1
@author: Angela (Qiqian) Yang
"""
# import statement 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/bridges/bridges.data.version1"
my_data = pd.read_csv(url, header=None)
my_data.columns = ['IDENTIF', 'RIVER', 'LOCATION', 'ERECTED', 'PURPOSE',
                   'LENGTH', 'LANES', 'CLEAR-G', 'T-OR-D', 'MATERIAL',
                   'SPAN', 'REL-L', 'TYPE']

my_data.dtypes

## 1. Replace missing numeric data
# look up missing values, and get to know data types of them
my_data = my_data.replace(to_replace="?", value=float("NaN"))
my_data.isnull().sum()
# impute missing value in LENGTH to its median value 
my_data.loc[:, "LENGTH"] = pd.to_numeric(my_data.loc[:, "LENGTH"], errors='coerce')
HasNan = np.isnan(my_data.loc[:,"LENGTH"])
my_data.loc[HasNan, "LENGTH"] = np.nanmedian(my_data.loc[:,"LENGTH"])
# drop other columns, which can't be imputed by numeric data 
my_data = my_data.dropna(axis=0)


## 2. Account for outlier values in numeric columns 
x = np.array(my_data['LENGTH'])
LimitHi = np.mean(x) + 2*np.std(x)
LimitLo = np.mean(x) - 2*np.std(x)
FlagOutlier = (x < LimitLo) | (x > LimitHi)
print ('the histogram plot for the hours-per-week column is below:')
plt.hist(my_data.loc[~FlagOutlier, 'LENGTH'], bins = 10, color=[1, 1, 0, 0.5])
plt.hist(my_data.loc[FlagOutlier, 'LENGTH'], bins = 10, color=[0, 0, 1, 0.5])
plt.xlabel('length of the bridge')
plt.title("Compare Distributions between inliers and outliers")
plt.show()
print ('below are the value counts for outliers in LENGTH column')
print (my_data.loc[FlagOutlier, 'LENGTH'].value_counts())
# because the outliers are only three rows, approcimitely 4% of the whole data
# so I decided to simply remove the outliers
my_data = my_data.loc[~FlagOutlier]


## 3. Normalize numeric values 
length =np.array(my_data['LENGTH'])
# use min-max normalized as an example
length_mm = (length - np.min(length))/(np.max(length) - np.min(length)) 
# look at original data distributrions 
plt.hist(length, bins = 20, color=[0, 0, 0, 1])
plt.title("Original Bridge Length Distribution")
plt.xlabel('length of the bridge')
plt.show()
plt.hist(length_mm, bins = 20, color=[1, 1, 0, 1])
plt.title("MinMax-normalization of Original Length")
plt.xlabel('length of the bridge')
plt.show()


## 4. Bin numeric variables
# Decide to have 3 bins
NumberOfBins = 3
x = np.array(my_data['LENGTH'])
# equal frequency binning
ApproxBinCount = len(x)/NumberOfBins
print("\n########\n\n Each bin should contain approximately", ApproxBinCount, "elements.")
print(np.sort(x))
# Bins with 26, 26, and 25 elements
# Calculate the bin edge value
MinBin = float('-inf')  
MaxBin1 = 1092.5
MaxBin2 = 1500.5
MaxBin = float('inf')
print("\n########\n\n Bin 1 is from ", MinBin, " to ", MaxBin1)
print(" Bin 2 is greater than ", MaxBin1, " up to ", MaxBin2)
print(" Bin 3 is greater than ", MaxBin2, " up to ", MaxBin)

# Empty starting point for equal-frequency-binned array
Binned_EF = np.array([" "]*len(x)) 
Binned_EF[(MinBin < x) & (x <= MaxBin1)] = "L" # Low
Binned_EF[(MaxBin1 < x) & (x <= MaxBin2)] = "M" # Med
Binned_EF[(MaxBin2 < x) & (x  < MaxBin)] = "H" # High
print(" x binned into 3 equal-freq1uency bins: ")
print(Binned_EF)


## 5. Consolidate categorical data
# for bridge type, I want to consolidate Cont-T with SIMPLE-T 
my_data.loc[my_data.loc[:, "TYPE"] == "CONT-T", "TYPE"] = "SIMPLE-T"


## 6. One-hot encode categorical data with at least 3 categories
# encode the river 
print (my_data['RIVER'].value_counts())
my_data.loc[:, "A_RIVER"] = (my_data.loc[:, "RIVER"] == "A").astype(int)
my_data.loc[:, "M_RIVER"] = (my_data.loc[:, "RIVER"] == "M").astype(int)
my_data.loc[:, "O_RIVER"] = (my_data.loc[:, "RIVER"] == "O").astype(int)


## 7. Remove obsolete columns
# Remove 'RIVER', because it already presented by above three columns
my_data = my_data.drop("RIVER", axis=1)

## 8. Write out data 
my_data.to_csv('AngelaQiqianYang-M02-Dataset.csv', index=False)

## 8. Final Comment
# With the Pittsburgh bridges dataset, I mostly focus on bridge length. 
# Because this is the only column that could transfer to numeric value and won't mislead the model. 
# there are three extreme long bridge. It is the reason that three outliers showed up in the database. 
# I use min-max normalization is because the length values has no outliers already, and min-max is easy to understand and explain 
# I binned length in equal frequency binned array is because I want to see where the length distribution divided. 
# I chose to consolidate the bridge types, because there are two types really similar to each other. 
# I chose to encode the RIVER, because river represent really different geographic location, so it worth to seperate them and treat them as three different valuables. 
 























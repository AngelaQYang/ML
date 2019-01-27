# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 00:17:49 2018

@author: Angela Q. Yang
"""

import numpy as np 
# Create three array with the data
x1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,100, 10000])
x2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,100, 10000, 20000])
x3 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,100, ''])

#Write function(s) that remove outliers in the first array
mean_x1 = x1.mean()
std_x1 = x1.std()
x1_good = (x1<(mean_x1 + 2*std_x1)) & (x1>(mean_x1-2*std_x1))
x1 = x1[x1_good]

#Write function(s) that replace outliers in the second array
x2_bad= (x2>(x2.mean() + 2*x2.std())) | (x2<(x2.mean()-2*x2.std()))
x2[x2_bad] = np.median(x2)

# Write function(s) that fill in missing values in the third array

x3_bad = (x3=='')
x3 = x3[~x3_bad]

# summary comment: my data has been cleaned up by detect the outlier data and missing data, which will be repalced by median data point later.
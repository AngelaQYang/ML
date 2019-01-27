# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:49:23 2018
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

### 1. Normalize variable: LENGTH
length =np.array(my_data['LENGTH'])
FlagGood = length != "?"
length = length[FlagGood]
length = length.astype(int)
length_z = (length - np.mean(length))/np.std(length) # use z normalized as an example
# look at original data distributrions 
plt.hist(length, bins = 20, color=[0, 0, 0, 1])
plt.title("Original Length Distribution")
plt.show()

plt.hist(length_z, bins = 20, color=[1, 1, 0, 1])
plt.title("Z-normalization")
plt.show()


### 2. Bin variable: bridge length
# Decide to have 5 bins
NumberOfBins = 5
BinWidth = (max(length) - min(length))/NumberOfBins
# Calculate the bin edge value
MinBin = float('-inf')  
MaxBin1 = min(length) + 1 * BinWidth
MaxBin2 = min(length) + 2 * BinWidth
MaxBin3 = min(length) + 3 * BinWidth
MaxBin4 = min(length) + 4 * BinWidth
MaxBin = float('inf')
print("\n########\n\n Bin 1 is from ", MinBin, " to ", MaxBin1)
print(" Bin 2 is greater than ", MaxBin1, " up to ", MaxBin2)
print(" Bin 3 is greater than ", MaxBin2, " up to ", MaxBin3)
print(" Bin 4 is greater than ", MaxBin3, " up to ", MaxBin4)
print(" Bin 5 is greater than ", MaxBin4, " up to ", MaxBin)

#create an empty array of strings
Binned_EqW = np.array([" "]*len(length)) 
Binned_EqW[(MinBin < length) & (length <= MaxBin1)] = "XS" # Extra short bridge
Binned_EqW[(MaxBin1 < length) & (length <= MaxBin2)] = "S" # Short bridge
Binned_EqW[(MaxBin2 < length) & (length  < MaxBin3)] = "M" # Medium long bridge
Binned_EqW[(MaxBin3 < length) & (length  < MaxBin4)] = "L" # Long bridge
Binned_EqW[(MaxBin4 < length) & (length  < MaxBin)] = "XL" # Extram long bridge
print(" bridge length binned into 5 equal-width bins: ")
print(Binned_EqW)


### 3. Decoding categorical data: bridge lanes
# overview values in "LANES" variable
my_data.loc[:,"LANES"].value_counts()
# remove missing data
my_data = my_data[~(my_data.loc[:,"LANES"] == '?')]
my_data.loc[ my_data.loc[:, "LANES"] == "1", "LANES"] = 1
my_data.loc[ my_data.loc[:, "LANES"] == "2", "LANES"] = 2
my_data.loc[ my_data.loc[:, "LANES"] == "4", "LANES"] = 4
my_data.loc[ my_data.loc[:, "LANES"] == "6", "LANES"] = 6

### 3. Imputing missing values
# look up missing values, and get to know data types of them
my_data = my_data.replace(to_replace="?", value=float("NaN"))
my_data.isnull().sum()

# impute missing value to median value 
my_data.loc[:, "LENGTH"] = pd.to_numeric(my_data.loc[:, "LENGTH"], errors='coerce')
HasNan = np.isnan(my_data.loc[:,"LENGTH"])
my_data.loc[HasNan, "LENGTH"] = np.nanmedian(my_data.loc[:,"LENGTH"])
# drop other columns, which can't be transfered to numeric data 
my_data = my_data.dropna(axis=0)

### 4. Consolidating categories if applicable
# for material, consolidating iron and steel 
my_data.loc[my_data.loc[:, "MATERIAL"] == "IRON", "MATERIAL"] = "STEEL"

### 5. One-hot encoding (dummy variables) for a categorical column 
# I am interested in code 'PURPOSE' variable
print (my_data['PURPOSE'].value_counts())
my_data.loc[:, "HIGHWAY"] = (my_data.loc[:, "PURPOSE"] == "HIGHWAY").astype(int)
my_data.loc[:, "RR"] = (my_data.loc[:, "PURPOSE"] == "RR").astype(int)
my_data.loc[:, "AQUEDUCT"] = (my_data.loc[:, "PURPOSE"] == "AQUEDUCT").astype(int)
# Remove 'PURPOSE', because it already presented by above three columns
my_data = my_data.drop("PURPOSE", axis=1)

### 6. plot one or more categories 
plt.hist(my_data.loc[:, "LANES"])
plt.xlabel('lane numbers')
plt.ylabel('frequency')


### 7. summery comment 
'''
I encode column ["LANES"] from string value: '1', '2', '4', '6' to numeric value: 1,2,4,6
It is because more lane numbers does impact bridge design, so I encoded categorical data to numerical to reflect this.

Before imputing missing values, I found where all missing value are. Some columns are categorical data and not suitable to transfer them to numerical. 
For these columns, they are unable to impute missing value by using median values. 
For the rest of columns (['LENGTH']) I transfered data type to numerical, and used median to impute the missing value. 

The column ["MATERIAL"], the value "iron" is close to 'steel'. So I consolidate 'iron' to become'steel'.

I coded dummie variables on column ['PURPOSE']. The reason is different purpose of bridge may impact the design in dramatic way. 
And the purpose are exclusive to each other, so dummie valuable is perfect for this column. 
After created the dummie variables, I deleted obsolete. 
 
'''



















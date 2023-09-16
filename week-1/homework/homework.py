import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Question 1
# What's the version of Pandas that you installed?
pd.__version__ # '2.0.1'

df = pd.read_csv("week-1/homework/raw.githubusercontent.com_alexeygrigorev_datasets_master_housing.csv")
# Question 2
# How many columns are in the dataset?
df.shape # There are 10 columns in the dataset.

# Question 3
# Which columns in the dataset have missing values?
df.isnull().sum() # Only 'total_bedrooms' have missing values.

# Question 4
# How many unique values does the ocean_proximity column have?
df["ocean_proximity"].nunique() # "ocean_proximity" has 5 unique values.


# Question 5
# What's the average value of the median_house_value for the houses located near the bay?
df[df["ocean_proximity"] == "NEAR BAY"]["median_house_value"].mean() # 259212.31179039303


# Question 6
# 1- Calculate the average of total_bedrooms column in the dataset.
# 2- Use the fillna method to fill the missing values in total_bedrooms with the mean value from the previous step.
# 3- Now, calculate the average of total_bedrooms again.
# 4- Has it changed?
fill_mean = df['total_bedrooms'].mean()
df.fillna(fill_mean, inplace=True)
after_fill_mean = df['total_bedrooms'].mean() # mean didn't change

# Question 7
# 1- Select all the options located on islands.
# 2- Select only columns housing_median_age, total_rooms, total_bedrooms.
# 3- Get the underlying NumPy array. Let's call it X.
# 4- Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# 5- Compute the inverse of XTX.
# 6- Create an array y with values [950, 1300, 800, 1000, 1300].
# 7- Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# 8- What's the value of the last element of w?
island_group = df.loc[df["ocean_proximity"] == "ISLAND"]
column_group = df.loc[df["ocean_proximity"] == "ISLAND",["housing_median_age", "total_rooms", "total_bedrooms"]]
X = column_group.values
XT = X.T
XTX = np.dot(XT , X)
inverse_XTX = np.linalg.inv(XTX)
y = np.array([950, 1300, 800, 1000, 1300])
w = np.dot(np.dot(inverse_XTX, XT), y)
last_element_of_w = w[-1] # 5.699229455065594





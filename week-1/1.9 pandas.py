import pandas as pd
import numpy as np

data = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
    ['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'Sedan', 27150],
    ['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
    ['GMC', 'Acadia',  2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340],
]

columns = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle_Style', 'MSRP'
]
# DataFrames
df = pd.DataFrame(data, columns=columns)
df.head(2)

# Series
df["Make"]
df[["Make", "Model", "MSRP"]]
df["id"] = [1,2,3,4,5]
del df["id"]
# Index
df.index
df.Make.index

# Accessing elements
df.index = ["a", "b", "c", "d", "e"]
df.loc[[1, 2]]
df.loc[["b", "c"]]
df = df.reset_index(drop=True)

# Element-wise operations
df["Engine HP"] / 100
df["Engine HP"] * 2
df["Year"] >= 2015

# Filtering
df[
    df["Year"] >= 2015
]

df[
    df["Make"] == "Nissan"
]
df[
    (df["Make"] == "Nissan") & (df["Year"] >= 2015)
]

# String operations
df["Vehicle_Style"].str.lower()
df["Vehicle_Style"].str.replace(" ", "_")

df["Vehicle_Style"] = df["Vehicle_Style"].str.replace(" ", "_").str.lower()

# Summarizing operations
df["MSRP"].describe()
df.describe().round()

df["Make"].nunique()
df["Make"].unique()

# Missing values
df.isnull().sum()

# Grouping
df.groupby("Transmission Type").MSRP.min()

# Getting the NumPy arrays
df.MSRP.values
# Script to describe the dataset. 
# i want this to be like the R's summary() function.


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
from ucimlrepo import fetch_ucirepo

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# R style summary function
def summary(data):
    # Number of rows and columns
    print("Number of rows: ", data.shape[0])
    print("Number of columns: ", data.shape[1])
    print("\n")
    
    # Column names
    print("Column names: ", data.columns)
    print("\n")
    
    # Data types
    print("Data types: ")
    print(data.dtypes)
    print("\n")
    
    # Missing values
    print("Missing values: ", data.isnull().sum())
    print("\n")
    
    # Summary statistics
    print("Summary statistics: ")
    print(data.describe())
    print("\n")
    
    # Unique values
    print("Unique values: ")
    for col in data.columns:
        print(col, ":", data[col].nunique())
    print("\n")
    
    # Frequency of unique values
    print("Frequency of unique values: ")
    for col in data.columns:
        print(data[col].value_counts())
    print("\n")
    
    # Correlation
    print("Correlation: ")
    print(data.corr())
    print("\n")
    
    # Plotting histogram
    data.hist(figsize=(10, 10))
    plt.show()

# Call the function
summary(wine_quality.data.features)

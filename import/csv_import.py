import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
red_wine = pd.read_csv(r"C:\Users\Alex\Desktop\VSCode\prin_of_lang\Data-Analsys-Project\data\winequality-red.csv", sep=";")
white_wine = pd.read_csv(r"C:\Users\Alex\Desktop\VSCode\prin_of_lang\Data-Analsys-Project\data\winequality-white.csv", sep=";")

# --- Start of Analysis ---

# Basic statistics for both datasets
print("Red Wine Summary:")
print(red_wine.describe())
print("\nWhite Wine Summary:")
print(white_wine.describe())

# Check for missing values
print("\nMissing Values (Red Wine):")
print(red_wine.isnull().sum())
print("\nMissing Values (White Wine):")
print(white_wine.isnull().sum())

# Correlation matrix (example for red wine - do the same for white)
plt.figure(figsize=(12, 10))
plt.matshow(red_wine.corr()) # Use matplotlib's matshow
plt.xticks(range(len(red_wine.columns)), red_wine.columns, rotation=90)
plt.yticks(range(len(red_wine.columns)), red_wine.columns)
plt.colorbar()
plt.title("Red Wine Correlation Matrix")
plt.show()

# Example plot: Alcohol vs. Quality (for red wine)
plt.figure(figsize=(8, 6))
plt.scatter(red_wine["alcohol"], red_wine["quality"])
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Red Wine: Alcohol vs. Quality")
plt.show()

# Histograms of Quality (for both red and white)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(red_wine["quality"], bins=range(3, 11))
plt.xlabel("Quality")
plt.ylabel("Frequency")
plt.title("Distribution of Red Wine Quality")

plt.subplot(1, 2, 2)
plt.hist(white_wine["quality"], bins=range(3, 11))
plt.xlabel("Quality")
plt.ylabel("Frequency")
plt.title("Distribution of White Wine Quality")

plt.tight_layout()  # Adjusts subplot parameters for a tight layout
plt.show()

# --- End of Analysis ---
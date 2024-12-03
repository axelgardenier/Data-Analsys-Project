import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Load the datasets
red_wine = pd.read_csv(r"C:\Users\Alex\Desktop\VSCode\prin_of_lang\Data-Analsys-Project\data\winequality-red.csv", sep=";")
white_wine = pd.read_csv(r"C:\Users\Alex\Desktop\VSCode\prin_of_lang\Data-Analsys-Project\data\winequality-white.csv", sep=";")

def generate_report(df, wine_type):
    report = f"--- {wine_type.upper()} WINE QUALITY REPORT ---\n\n"

    # 1. Summary Statistics
    report += "1. Summary Statistics:\n"
    report += str(df.describe()) + "\n\n"

    # 2. Missing Values
    report += "2. Missing Values:\n"
    report += str(df.isnull().sum()) + "\n\n"

    # 3. Correlation Matrix
    report += "3. Correlation Matrix:\n"
    correlations = df.corr()
    report += str(correlations) + "\n\n"

    # 4. Quality Statistics by Feature
    report += "4. Quality Statistics by Feature:\n"
    for column in df.columns:
        if column != "quality":
            report += f"\n--- {column.upper()} ---\n"
            for quality_level in sorted(df["quality"].unique()):
                subset = df[df["quality"] == quality_level]
                mean = subset[column].mean()
                std = subset[column].std()
                report += f"Quality {quality_level}: Mean = {mean:.2f}, Std = {std:.2f}\n"
            # ANOVA test
            groups = [df[df["quality"] == q][column] for q in sorted(df["quality"].unique())]
            fvalue, pvalue = stats.f_oneway(*groups)
            report += f"ANOVA p-value: {pvalue:.3f}\n"
            if pvalue < 0.05:
                report += "Significant difference between quality levels.\n"
            else:
                report += "No significant difference between quality levels.\n"

    # 5. Outlier Analysis
    report += "\n5. Outlier Analysis:\n"
    for column in df.columns:
        if column != "quality":  # Exclude the 'quality' column
            z = np.abs(stats.zscore(df[column]))
            outliers = df[(z > 3)]  # Identify outliers based on z-score > 3
            num_outliers = len(outliers)
            report += f"Number of outliers in {column}: {num_outliers}\n"

    return report

red_wine_report = generate_report(red_wine, "red")
white_wine_report = generate_report(white_wine, "white")

with open("wine_quality_report.txt", "w") as f:
    f.write(red_wine_report)
    f.write("\n\n")  # Add some spacing between reports
    f.write(white_wine_report)

print("Report generated successfully: wine_quality_report.txt")

# Optional: Display plots (uncomment if needed)
# plt.show()

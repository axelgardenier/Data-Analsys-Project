Final Report Outline: Wine Quality Analysis
1. Title Page
Project Title: Wine Quality Analysis
Course: COMCS 230 - Principles of Programming Languages
Date: December 13, 2024
Authors: Alex Gardenier and Sebastian [Last Name]


2. Abstract
Brief summary of the project:
Objectives and significance.


Methodology and key findings.


Final conclusions and implications.




3. Introduction
Problem Statement:
The importance of wine quality in production and marketing.
The value of identifying physicochemical predictors of quality.
Research Question:
How do physicochemical properties affect the perceived quality of wine?
Hypotheses:
Null Hypothesis (H0): Physicochemical properties have no significant impact on wine quality.
Alternative Hypothesis (Ha): At least one physicochemical property significantly affects wine quality.


4. Dataset Overview
Source: UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/186/wine+quality 
Description:
6,497 samples (1,599 red and 4,898 white wines).
Features: 12 physicochemical properties and 1 target variable (quality).
Key Variables:
Predictors: alcohol, volatile acidity, sulphates, residual sugar, density.
Target: quality (score range: 3-9).


5. Methods
Dataset loading and Preprocessing 
We utilized a dataset with two sections for this project:

Red Wine Quality: Contains physicochemical characteristics of red variants of the Portuguese "Vinho Verde" wine.
White Wine Quality: Contains physicochemical characteristics of white variants of the Portuguese "Vinho Verde" wine.

Both datasets are linked above and can be found in the data directory (in .csv form) 

data/winequality-red.csv
data/winequality-white.csv


Combining the Datasets
The initial step involved loading the individual datasets and then merging them into a single, unified dataset for comprehensive analysis. Here's a breakdown of the process:

Loading the Data:

The datasets were loaded using the pandas library. The read_csv function was used with a semicolon (;) as the separator, as specified in the dataset format.

# Dataset Paths
red_wine_path = "data/winequality-red.csv"
white_wine_path = "data/winequality-white.csv"

# Load datasets
red_wine = pd.read_csv(red_wine_path, sep=';')
white_wine = pd.read_csv(white_wine_path, sep=';')

Adding a Wine Type Identifier:
To distinguish between red and white wines after combining, a new column named wine_type was added to each dataset. This column was populated with the string 'red' for the red wine dataset and 'white' for the white wine dataset.

# Add wine type

red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'


Concatenating the Datasets:
Finally, the two datasets were combined vertically using the pd.concat function. The ignore_index=True argument was used to reset the index of the resulting DataFrame, ensuring a continuous index across all rows.

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

The resulting wine_data DataFrame now contains all the data from both the red and white wine datasets, with an added wine_type column to differentiate the origin of each sample.


















Exploratory Data Analysis (EDA)
This section outlines the exploratory data analysis (EDA) performed on the combined wine quality dataset. The primary goal of this EDA was to understand the data's characteristics, identify patterns, and gain insights that would inform our subsequent modeling steps.
Initial Data Overview
The first step in our EDA involved getting a general overview of the dataset. We printed the head and tail of the dataset to get a glimpse of the data structure and values. We also generated descriptive statistics to understand the central tendency, dispersion, and shape of the data's distribution.

# Display head and tail of the dataset
print("Head of the dataset:")
print(data.head())
print("\n\nTail of the dataset:")
print(data.tail())

# Generate descriptive statistics
summary_stats = data.describe()
print("\n\nSummary Statistics:")
print(summary_stats)

Correlation Analysis
To understand the relationships between different variables, we calculated the correlation matrix and visualized it as a heatmap. This helped us identify which variables were strongly correlated with each other and with the target variable, quality.






# Correlation heatmap (numeric columns only)

numeric_data = data.select_dtypes(include=[np.number])  # Exclude non-numeric columns
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Wine Features")
plt.tight_layout()
save_plot("outputs/final_vis/correlation_heatmap.png")
plt.show()


Key Observations from the Correlation Heatmap:

Alcohol content showed a notable positive correlation with quality.
Volatile acidity exhibited a significant negative correlation with quality.
Density had a strong negative correlation with alcohol, which is expected as alcohol is less dense than water.
Distribution of Key Predictors
We further investigated the distributions of the top five predictors identified from the correlation analysis: alcohol, volatile acidity, sulphates, citric acid, and density. Histograms with kernel density estimates were generated to visualize the distribution of each predictor.

# Histogram of top 5 predictors

top_predictors = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'density']
plt.figure(figsize=(12, 10))
for i, predictor in enumerate(top_predictors):
    plt.subplot(3, 2, i + 1)
    sns.histplot(data[predictor], kde=True, color='skyblue', bins=30)
    plt.title(f"{predictor.capitalize()} Distribution")
    plt.xlabel(predictor.capitalize())
    plt.ylabel("Frequency")
plt.tight_layout()
save_plot("outputs/final_vis/top_predictor_histograms.png")
plt.show()



Key Observations from the Histograms:

Alcohol content showed a relatively normal distribution, slightly skewed to the right.
Volatile acidity exhibited a right-skewed distribution, with most wines having lower levels of volatile acidity.
Sulphates also showed a right-skewed distribution.
Citric acid and density distributions provided additional insights into the characteristics of the wines.
Stacked Bar Chart: Wine Quality Distribution by Wine Type
Finally, we created a stacked bar chart to visualize the distribution of wine quality for each wine type (red and white). This helped us understand if there were any differences in the quality distribution between the two types of wine.




def plot_stacked_bar_chart(data):
 """Generates a stacked bar chart showing wine type distribution by quality.""" 
quality_counts = data.groupby(['quality', 'wine_type']).size().unstack()

quality_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#E06666', '#6699CC'])
plt.title('Distribution of Wine Type by Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Wine Type')
save_plot("outputs/final_vis/stacked_bar_chart.png")  # Save the plot
plt.show()



Key Observations from the Stacked Bar Chart:

The distribution of wine quality was relatively similar between red and white wines.
The majority of wines were rated as quality 5, 6, or 7.
This was less insightful than the heatmap visual specifically, but it is still valuable insight. 
Conclusion
The EDA provided valuable insights into the wine quality dataset. We gained a better understanding of the data's characteristics, identified key predictors, and observed relationships between variables. These findings informed our subsequent modeling steps, where we aimed to build predictive models to estimate wine quality based on its physicochemical properties.

b. Exploratory Data Analysis (EDA)
Techniques:
Descriptive statistics for key variables.
Heatmap for correlations between predictors and quality.


Histograms for variable distributions.

Findings:
Alcohol shows the strongest positive correlation with quality.
Volatile acidity exhibits the strongest negative correlation.
c. Statistical Modeling
Linear Regression:
Analyze the impact of individual predictors on wine quality.
Metrics: (R^2), RMSE.
d. Visualization
Scatterplots with regression lines.
Heatmap of correlations.


6. Results
a. Linear Regression Results
Summary table of metrics for predictors (e.g., alcohol, volatile acidity).


b. Visualization Outputs
Heatmap for correlation overview.
Scatterplots for linear regression insights.


7. Discussion
Key Findings:
Alcohol is the most influential predictor of quality.
Volatile acidity negatively impacts wine quality.
Sulphates have a moderate positive effect.
Interpretation of Results:
Discuss how physicochemical properties impact wine quality.
Explain the significance of non-linear relationships and interactions.
Limitations:
Dataset imbalance (fewer red wines).
Potential overfitting in quadratic regression.
Lack of sensory data (e.g., taste, aroma).


8. Conclusion
Summary:
Alcohol, volatile acidity, and sulphates significantly influence wine quality.
Quadratic regression outperforms linear models in capturing complex relationships.
Practical Implications:
Recommendations for winemakers:
Optimize alcohol and sulphate levels.
Minimize volatile acidity.
Future Work:
Explore advanced machine learning models (e.g., Random Forests, Gradient Boosting).
Incorporate sensory data for richer analysis.


9. Software Configuration Management
GitHub Repository:
Repository link: https://github.com/axelgardenier/Data-Analsys-Project
Repository structure:
Data: Raw and processed datasets.
Code: Python scripts for data preprocessing, modeling, and visualization.
Documentation: README and analysis notes.
Version Control:
Git commands used (git status, git commit, etc.).
Collaboration strategies (e.g., resolving merge conflicts).


10. References
List sources, including the UCI Machine Learning Repository and any referenced statistical or programming resources.


11. Appendices (if needed)
Code snippets for key functions (e.g., regression models).
Detailed output logs (e.g., (R^2) and RMSE calculations).
Additional visualizations not included in the main report.



This outline should guide you in developing a well-structured final report. Let me know if you'd like assistance with specific sections or further refinement!


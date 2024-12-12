import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # --- Data Loading and Preprocessing ---
    print("Loading and preprocessing data...")
    red_wine_path = "data/winequality-red.csv"
    white_wine_path = "data/winequality-white.csv"

    # Load datasets
    red_wine = pd.read_csv(red_wine_path, sep=';')
    white_wine = pd.read_csv(white_wine_path, sep=';')
    print("Loading data.")

    # Add wine type
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    print("Adding wine type.")

    # Combine datasets
    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    print("Combining datasets.")

    # --- Exploratory Data Analysis ---
    print("Performing exploratory data analysis...")
    perform_eda(wine_data)

    # --- Simple Linear Regression ---
    print("Performing simple linear regression...")
    predictors = ['alcohol', 'volatile acidity', 'sulphates']
    perform_simple_regression(wine_data, predictors)


def perform_eda(data):
    # Print head and tail of the dataset
    print("Head of the dataset:")
    print(data.head())
    print("\n\nTail of the dataset:")
    print(data.tail())

    # Generate descriptive statistics
    summary_stats = data.describe()
    print("\n\nSummary Statistics:")
    print(summary_stats)

    # Correlation heatmap (numeric columns only)
    numeric_data = data.select_dtypes(include=[np.number])  # Exclude non-numeric columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Wine Features")
    plt.tight_layout()
    save_plot("outputs/final_vis/correlation_heatmap.png")
    plt.show()

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

    # Stacked Bar Chart: Wine Quality Distribution by Wine Type
    plot_stacked_bar_chart(data)
    print("Stacked bar chart of wine quality distribution by wine type created and saved.")

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

def perform_simple_regression(data, predictors):
    os.makedirs("outputs/plots", exist_ok=True)

    for predictor in predictors:
        # Prepare data
        X = data[[predictor]].values
        y = data['quality'].values

        # Train-test split (42 -> meaning of life -> random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Model for {predictor}: R^2 = {r2:.3f}, RMSE = {rmse:.3f}")

        # Scatter plot with regression line
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_test.ravel(), y=y_test, alpha=0.6, label='Observed')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        plt.title(f"{predictor.capitalize()} vs Quality")
        plt.xlabel(predictor.capitalize())
        plt.ylabel("Quality")
        plt.legend()
        save_plot(f"outputs/final_vis/{predictor}_quality_scatterplot.png")


def save_plot(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D


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

    # --- Quadratic Regression with 3D Visualization ---
    print("Performing quadratic regression and generating 3D visualization...")
    perform_quadratic_regression(wine_data)


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

    # Histogram of top 5 predictors\]
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


def perform_simple_regression(data, predictors):
    os.makedirs("outputs/plots", exist_ok=True)

    for predictor in predictors:
        # Prepare data
        X = data[[predictor]].values
        y = data['quality'].values

        # Train-test split
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


def perform_quadratic_regression(data):
    # Select key predictors
    predictors = ['alcohol', 'volatile acidity']
    X = data[predictors].values
    y = data['quality'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Polynomial regression (degree=2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test_poly)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Quadratic Model: R^2 = {r2:.3f}, RMSE = {rmse:.3f}")

    alcohol_range = np.linspace(data['alcohol'].min(), data['alcohol'].max(), 50)
    acidity_range = np.linspace(data['volatile acidity'].min(), data['volatile acidity'].max(), 50)
    alcohol_grid, acidity_grid = np.meshgrid(alcohol_range, acidity_range)

    # Predict quality for each combination of alcohol and acidity
    grid_data = np.c_[alcohol_grid.ravel(), acidity_grid.ravel()]
    grid_poly = poly.transform(grid_data)
    quality_grid = model.predict(grid_poly).reshape(alcohol_grid.shape)

    # Plot 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis', alpha=0.6, label='Observed Data')
    ax.plot_surface(alcohol_grid, acidity_grid, quality_grid, cmap='viridis', alpha=0.7, edgecolor='none')
    ax.set_title("3D Visualization: Alcohol & Volatile Acidity vs Quality")
    ax.set_xlabel("Alcohol")
    ax.set_ylabel("Volatile Acidity")
    ax.set_zlabel("Quality")
    ax.text2D(0.05, 0.95, f"R^2 = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes) # Add metrics to plot
    save_plot("outputs/final_vis/3D_quality_visualization.png")


def save_plot(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    main()

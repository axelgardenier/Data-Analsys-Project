#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Script to Analyze and Visualize Wine Quality Dataset

This script performs the following:
1. Loads and preprocesses red and white wine datasets.
2. Trains simple linear regression models using individual predictors.
3. Compares the performance of these models.
4. Builds an advanced polynomial regression model.
5. Visualizes the predictions in 3D.
6. Saves evaluation metrics and visualizations.

Author: AG
Date: 2024-12-04
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

def create_directories():
    """
    Create necessary directories for results and outputs.
    """
    directories = ['results', 'outputs/plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories checked/created: 'results', 'outputs/plots'.")

def load_data(red_wine_path='data/winequality-red.csv', white_wine_path='data/winequality-white.csv'):
    """
    Load red and white wine datasets from CSV files.

    Parameters:
        red_wine_path (str): Path to red wine CSV file.
        white_wine_path (str): Path to white wine CSV file.

    Returns:
        red (DataFrame): Red wine dataset.
        white (DataFrame): White wine dataset.
    """
    try:
        red = pd.read_csv(red_wine_path, sep=';')
        white = pd.read_csv(white_wine_path, sep=';')
        print("Datasets loaded successfully.")
        return red, white
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise

def add_wine_type(red, white):
    """
    Add a 'wine_type' column to distinguish between red and white wines.

    Parameters:
        red (DataFrame): Red wine dataset.
        white (DataFrame): White wine dataset.

    Returns:
        red (DataFrame): Red wine dataset with 'wine_type' column.
        white (DataFrame): White wine dataset with 'wine_type' column.
    """
    red['wine_type'] = 'red'
    white['wine_type'] = 'white'
    print("'wine_type' column added to both datasets.")
    return red, white

def combine_datasets(red, white):
    """
    Combine red and white wine datasets into a single DataFrame.

    Parameters:
        red (DataFrame): Red wine dataset.
        white (DataFrame): White wine dataset.

    Returns:
        combined (DataFrame): Combined wine dataset.
    """
    combined = pd.concat([red, white], axis=0, ignore_index=True)
    print("Datasets combined into a unified DataFrame.")
    return combined

def validate_data(df):
    """
    Validate the combined dataset by checking for missing values and ensuring data integrity.

    Parameters:
        df (DataFrame): Combined wine dataset.

    Returns:
        df (DataFrame): Validated and cleaned dataset.
    """
    # Check for missing values
    if df.isnull().sum().any():
        print("Missing values detected. Handling missing data by dropping rows with missing values.")
        df = df.dropna()
    else:
        print("No missing values detected.")

    # Ensure all predictors are numeric
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = set(df.columns) - set(numeric_columns)
    if non_numeric:
        print(f"Non-numeric columns detected: {non_numeric}. Applying one-hot encoding.")
        df = pd.get_dummies(df, drop_first=True)
    else:
        print("All columns are numeric.")

    # Verify expected ranges (example for 'alcohol')
    if not df['alcohol'].between(0, 20).all():
        raise ValueError("Alcohol content out of expected range (0-20).")
    print("Data validation completed successfully.")
    return df

def load_and_preprocess_data():
    """
    Load, preprocess, and validate the wine datasets.

    Returns:
        validated_data (DataFrame): Cleaned and combined wine dataset.
    """
    red, white = load_data()
    red, white = add_wine_type(red, white)
    combined = combine_datasets(red, white)
    validated_data = validate_data(combined)
    return validated_data

def train_and_evaluate_simple_models(df, random_state=42):
    """
    Train and evaluate simple linear regression models using individual predictors.

    Parameters:
        df (DataFrame): Preprocessed wine dataset.
        random_state (int): Seed for reproducibility.

    Returns:
        results (list of dict): Evaluation metrics for each model.
    """
    predictors = ['alcohol', 'volatile acidity', 'sulphates']
    target = 'quality'

    results = []

    for predictor in predictors:
        X = df[[predictor]]
        y = df[target]

        # Split data
        # WHY DO WE SPLIT THE DATA?
        # breifing for dummies
        """
        Splitting the data into training and testing sets is a common practice in machine learning.
        The training set is used to train the model, while the testing set is used to evaluate its performance.
        This helps to assess how well the model generalizes to unseen data.
        By splitting the data, we can avoid overfitting the model to the training data and obtain a more realistic estimate of its performance.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

        # Initialize and train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store results
        results.append({
            'Predictor': predictor,
            'R2': r2,
            'RMSE': rmse
        })

        print(f"Model with predictor '{predictor}': R2 = {r2:.4f}, RMSE = {rmse:.4f}")

    return results

def compare_models(results):
    """
    Compare the performance of different regression models and visualize the comparison.

    Parameters:
        results (list of dict): Evaluation metrics for each model.
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display table
    print("\nModel Comparison:")
    print(results_df)

    # Save to CSV
    results_df.to_csv('results/model_comparison.csv', index=False)
    print("Model comparison metrics saved to 'results/model_comparison.csv'.")

    # Visualization
    plt.figure(figsize=(14, 6))

    # R2 Bar Chart
    plt.subplot(1, 2, 1)
    sns.barplot(x='Predictor', y='R2', data=results_df, palette='viridis')
    plt.title('R² Comparison')
    plt.ylim(0, 1)
    plt.ylabel('R² Score')

    # RMSE Bar Chart
    plt.subplot(1, 2, 2)
    sns.barplot(x='Predictor', y='RMSE', data=results_df, palette='magma')
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')

    plt.tight_layout()
    plt.savefig('outputs/plots/model_comparison.png')
    print("Model comparison plot saved to 'outputs/plots/model_comparison.png'.")
    plt.show()

def build_and_evaluate_complex_model(df, random_state=42):
    """
    Build and evaluate an advanced polynomial regression model with non-linear features.

    Parameters:
        df (DataFrame): Preprocessed wine dataset.
        random_state (int): Seed for reproducibility.

    Returns:
        pipeline (Pipeline): Trained polynomial regression model pipeline.
    """
    predictors = ['alcohol', 'volatile acidity', 'sulphates']
    target = 'quality'

    X = df[predictors]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    # Create pipeline with polynomial features
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear_regression', LinearRegression())
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Evaluate
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Analyze coefficients
    feature_names = pipeline.named_steps['poly_features'].get_feature_names_out(predictors)
    coefficients = pipeline.named_steps['linear_regression'].coef_
    intercept = pipeline.named_steps['linear_regression'].intercept_

    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Save metrics
    complex_metrics = {
        'Model': 'Polynomial Regression (Degree=2)',
        'R2_train': r2_train,
        'RMSE_train': rmse_train,
        'R2_test': r2_test,
        'RMSE_test': rmse_test
    }

    complex_metrics_df = pd.DataFrame([complex_metrics])
    complex_metrics_df.to_csv('results/complex_model_metrics.csv', index=False)
    print("Complex model metrics saved to 'results/complex_model_metrics.csv'.")

    # Save coefficients
    coeff_df.to_csv('results/complex_model_coefficients.csv', index=False)
    print("Complex model coefficients saved to 'results/complex_model_coefficients.csv'.")

    # Display metrics
    print("\nAdvanced Regression Model Performance:")
    print(complex_metrics_df)

    # Display coefficients
    print("\nModel Coefficients:")
    print(coeff_df)

    return pipeline

def visualize_3d_predictions(model, df):
    """
    Generate a 3D visualization of the advanced regression model's predictions.

    Parameters:
        model (Pipeline): Trained polynomial regression model pipeline.
        df (DataFrame): Preprocessed wine dataset.
    """
    predictors = ['alcohol', 'volatile acidity', 'sulphates']

    # For visualization, fix 'sulphates' at its median value
    sulphates_median = df['sulphates'].median()

    # Create mesh grid
    alcohol_range = np.linspace(df['alcohol'].min(), df['alcohol'].max(), 50)
    va_range = np.linspace(df['volatile acidity'].min(), df['volatile acidity'].max(), 50)
    alcohol_mesh, va_mesh = np.meshgrid(alcohol_range, va_range)

    # Flatten the mesh grid and create DataFrame
    mesh_df = pd.DataFrame({
        'alcohol': alcohol_mesh.ravel(),
        'volatile acidity': va_mesh.ravel(),
        'sulphates': sulphates_median
    })

    # Predict using the model
    quality_pred = model.predict(mesh_df)
    quality_pred = quality_pred.reshape(alcohol_mesh.shape)

    # Plotting
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(alcohol_mesh, va_mesh, quality_pred, cmap='viridis', alpha=0.7, edgecolor='none')

    # Scatter plot of actual data
    ax.scatter(df['alcohol'], df['volatile acidity'], df['quality'], color='red', alpha=0.3, label='Actual Data')

    ax.set_xlabel('Alcohol')
    ax.set_ylabel('Volatile Acidity')
    ax.set_zlabel('Quality')
    ax.set_title('3D Visualization of Predicted Wine Quality')
    ax.legend()

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('outputs/plots/3d_quality_prediction.png')
    print("3D quality prediction plot saved to 'outputs/plots/3d_quality_prediction.png'.")
    plt.show()

def main():
    """
    Main function to execute the data analysis and visualization pipeline.
    """
    # Set random seed for reproducibility
    random_state = 42
    np.random.seed(random_state)

    # Create necessary directories
    create_directories()

    # Load and preprocess data
    data = load_and_preprocess_data()

    # Train and evaluate simple models
    simple_models_results = train_and_evaluate_simple_models(data, random_state=random_state)

    # Compare models
    compare_models(simple_models_results)

    # Build and evaluate complex model
    complex_model = build_and_evaluate_complex_model(data, random_state=random_state)

    # Visualize 3D predictions
    visualize_3d_predictions(complex_model, data)

    print("\nAll tasks completed successfully. Check the 'results/' and 'outputs/plots/' directories for outputs.")

if __name__ == "__main__":
    main()

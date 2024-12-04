import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def main():

    # --- Data Loading and Preprocessing ---
    red_wine_path = "data/winequality-red.csv"  # Updated path
    white_wine_path = "data/winequality-white.csv"  # Updated path

    red_wine = pd.read_csv(red_wine_path, sep=';')
    white_wine = pd.read_csv(white_wine_path, sep=';')

    # Combine datasets
    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

    # Select predictors and target
    predictors = ['alcohol', 'volatile acidity']
    X = wine_data[predictors]
    y = wine_data['quality']

    # Fit a quadratic regression model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Create grid for 3D visualization
    alcohol_range = np.linspace(wine_data['alcohol'].min(), wine_data['alcohol'].max(), 50)
    volatile_acidity_range = np.linspace(wine_data['volatile acidity'].min(), wine_data['volatile acidity'].max(), 50)
    alcohol_grid, volatile_acidity_grid = np.meshgrid(alcohol_range, volatile_acidity_range)

    grid_data = pd.DataFrame({
        'alcohol': alcohol_grid.ravel(),
        'volatile acidity': volatile_acidity_grid.ravel()
    })
    grid_poly = poly.transform(grid_data)
    quality_grid = model.predict(grid_poly).reshape(alcohol_grid.shape)

    # --- Data Visualization ---
    plot_dir = "outputs/plots/"  # Directory to save plots

    os.makedirs(plot_dir, exist_ok=True)  # Create plot directory if it doesn't exist

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for raw data
    scatter = ax.scatter(wine_data['alcohol'], wine_data['volatile acidity'], wine_data['quality'],
                         c=wine_data['quality'], cmap='viridis', alpha=0.6)

    # Surface plot for quadratic regression
    surface = ax.plot_surface(alcohol_grid, volatile_acidity_grid, quality_grid, alpha=0.7, cmap='viridis', edgecolor='none')

    # Labels and title
    ax.set_title("Alcohol, Volatile Acidity, and Predicted Quality", fontsize=14)
    ax.set_xlabel("Alcohol Content (%)", fontsize=12)
    ax.set_ylabel("Volatile Acidity", fontsize=12)
    ax.set_zlabel("Quality Score", fontsize=12)
    fig.colorbar(scatter, ax=ax, label="Observed Quality", shrink=0.5)

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "alcohol_volatile_quality_3d.png")
    plt.savefig(plot_path)
    plt.show()


# Call the main function
if __name__ == "__main__":
    main()

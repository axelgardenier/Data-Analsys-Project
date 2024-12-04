import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # --- Data Loading and Preprocessing ---
    red_wine_path = "data/winequality-red.csv"  # Updated path
    white_wine_path = "data/winequality-white.csv"  # Updated path

    red_wine = pd.read_csv(red_wine_path, sep=';')
    white_wine = pd.read_csv(white_wine_path, sep=';')

    # Add a column for wine type
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'

    # Combine datasets
    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

    # Perform random sampling
    sample_size = 1000  # Number of samples for scatterplot
    sampled_data = wine_data.sample(n=sample_size, random_state=42)

    # Define key predictors
    key_predictors = ['alcohol', 'volatile acidity', 'sulphates', 'density']


    # --- Data Visualization ---
    plot_dir = "outputs/plots/"  # Directory to save plots

    os.makedirs(plot_dir, exist_ok=True) # Create plot directory if it doesn't exist

    # Generate scatterplots for each predictor
    for predictor in key_predictors:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=sampled_data, x=predictor, y='quality', hue='wine_type', alpha=0.6, edgecolor=None)
        plt.title(f"{predictor.capitalize()} vs Quality by Wine Type (Sampled Data)", fontsize=14)
        plt.xlabel(predictor.capitalize(), fontsize=12)
        plt.ylabel("Quality Score", fontsize=12)
        plt.legend(title="Wine Type", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_dir + f"{predictor}_quality_scatterplot.png")
        plt.show()


# Call the main function
if __name__ == "__main__":
    main()

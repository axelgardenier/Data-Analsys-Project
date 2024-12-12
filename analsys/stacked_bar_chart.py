import pandas as pd
import matplotlib.pyplot as plt


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

    # bar chart
    print("Generating stacked bar chart...")
    bins = input("Enter the number of bins for alcohol content: ")
    plot_stacked_bar_chart(wine_data, bins=int(bins))


# Stacked bar chart
# wide bars 
# shoudl only have the 7 quality levels 
# bin the alcohol content into 10 bins
def plot_stacked_bar_chart(data, bins=10):
    """Generates a stacked bar chart showing wine quality distribution by alcohol content"""

    # Bin alcohol content into 10 equal-width intervals
    data['alcohol_bins'] = pd.cut(data['alcohol'], bins=bins)

    # Create a crosstab of quality by alcohol bins
    quality_by_alcohol = pd.crosstab(data['quality'], data['alcohol_bins'])

    # Plot stacked bar chart
    quality_by_alcohol.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title("Wine Quality Distribution by Alcohol Content", fontsize=16)
    plt.xlabel("Quality", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Alcohol Content", title_fontsize='13', fontsize='12')
    plt.tight_layout()
    plt.show()


# Save plot function
def save_plot(file_path):
    """Saves the current plot to the specified file path."""
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}.")


if __name__ == "__main__":
    main()
# Wine Quality Data Analysis

This project performs a comprehensive exploratory data analysis (EDA) and statistical modeling of the Wine Quality Dataset from the UCI Machine Learning Repository. The goal is to understand the factors influencing wine quality, identify key relationships between physicochemical properties and quality ratings, and build predictive models for wine quality.

## Project Structure

*   `simple_analsys.py`: The main script to perform data loading, preprocessing, EDA, and statistical modeling (ANOVA and Multiple Linear Regression).
*   `data/`: Directory containing the raw wine quality datasets (winequality-red.csv and winequality-white.csv).
*   `outputs/`: Directory to store the preprocessed dataset and generated visualizations (histograms, boxplots, correlation heatmap).
*   `requirements.txt`: Lists the required Python packages for this project.

## Dataset

The Wine Quality Dataset contains physicochemical and sensory data for red and white Vinho Verde wines from northern Portugal. The dataset can be accessed through the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git  (Replace with your actual repository URL)    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  (.venv\Scripts\activate on Windows)    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt    ```

4. **Run the analysis script:**
    ```bash
    python simple_analsys.py    ```

The analysis results will be printed to the console, and visualizations will be saved in the `outputs/` directory. The preprocessed dataset is also saved to `outputs/preprocessed_wine_data.csv`.

## Analysis Summary

The `simple_analsys.py` script performs the following analyses:

*   **Data Preprocessing:** Handles missing values (if any), combines red and white wine data, removes outliers in 'fixed acidity', and transforms outliers in 'volatile acidity'.
*   **Exploratory Data Analysis (EDA):**
    *   Descriptive statistics (count, mean, std, min, max, percentiles)
    *   Histograms of each feature
    *   Boxplots of features grouped by wine type
    *   Correlation matrix heatmap
    *   Quality distribution analysis
*   **Descriptive Statistics by Quality:** Calculates mean and standard deviation of numeric features for each quality rating.
*   **ANOVA:** Performs Analysis of Variance to test for significant differences in the means of each chemical property across different wine quality levels.
*   **Multiple Linear Regression:** Builds multiple linear regression models to predict wine quality based on chemical properties, including model refinement based on significant variables.

## Further Analysis

This project provides a solid foundation for further analysis, including:

*   **Feature Engineering:** Creating new features from existing ones to potentially improve model performance.
*   **Model Selection and Tuning:** Exploring different regression models (e.g., Ridge, Lasso, Elastic Net) and optimizing hyperparameters.
*   **Model Evaluation:** Using more robust evaluation metrics (e.g., cross-validation) to assess model generalization.
*   **Interactive Visualizations:** Creating interactive dashboards to explore the data and model results more dynamically.

## Contributing

Contributions to this project are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

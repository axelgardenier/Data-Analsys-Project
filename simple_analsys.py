import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score

# --- Helper Functions ---
def create_output_dirs(dirs):
    """Creates directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Output directories {dirs} are ready.")

def load_and_preprocess_data(red_wine_path, white_wine_path, output_dir="outputs"):
    """
    Loads red and white wine datasets, combines them, and handles missing values and outliers.
    Saves the preprocessed data to a CSV file.
    """
    # Load datasets
    try:
        red_wine = pd.read_csv(red_wine_path, sep=';')
        print(f"Loaded red wine data from {red_wine_path}")
    except FileNotFoundError:
        print(f"Error: File {red_wine_path} not found.")
        sys.exit(1)
    
    try:
        white_wine = pd.read_csv(white_wine_path, sep=';')
        print(f"Loaded white wine data from {white_wine_path}")
    except FileNotFoundError:
        print(f"Error: File {white_wine_path} not found.")
        sys.exit(1)
    
    # Add wine type
    red_wine['type'] = 'red'
    white_wine['type'] = 'white'
    
    # Combine datasets
    df = pd.concat([red_wine, white_wine], ignore_index=True)
    print("Combined red and white wine datasets.")
    
    # Check for missing values
    if df.isnull().sum().any():
        print("Missing values detected. Handling missing values by dropping rows with missing data.")
        df.dropna(inplace=True)
    else:
        print("No missing values detected.")
    
    # Handle outliers for 'fixed acidity' using IQR and removal
    Q1 = df['fixed acidity'].quantile(0.25)
    Q3 = df['fixed acidity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_count = df.shape[0]
    df = df[(df['fixed acidity'] >= lower_bound) & (df['fixed acidity'] <= upper_bound)]
    removed = initial_count - df.shape[0]
    print(f"Removed {removed} outliers from 'fixed acidity' using IQR method.")
    
    # Handle outliers for 'volatile acidity' using Z-score and transformation (Winsorizing)
    z_scores = np.abs(stats.zscore(df['volatile acidity']))
    threshold = 3
    df['volatile acidity'] = np.where(z_scores > threshold, 
                                      df['volatile acidity'].quantile(0.95), 
                                      df['volatile acidity'])
    print(f"Transformed outliers in 'volatile acidity' using Z-score method and Winsorizing.")
    
    # Save preprocessed data
    preprocessed_path = os.path.join(output_dir, "preprocessed_wine_data.csv")
    df.to_csv(preprocessed_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_path}")
    
    return df

def load_raw_data(red_wine_path, white_wine_path):
    """Loads and combines the raw red and white wine datasets without preprocessing."""
    try:
        red_wine = pd.read_csv(red_wine_path, sep=';')
        print(f"Loaded red wine data from {red_wine_path}")
    except FileNotFoundError:
        print(f"Error: File {red_wine_path} not found.")
        sys.exit(1)

    try:
        white_wine = pd.read_csv(white_wine_path, sep=';')
        print(f"Loaded white wine data from {white_wine_path}")
    except FileNotFoundError:
        print(f"Error: File {white_wine_path} not found.")
        sys.exit(1)

    red_wine['type'] = 'red'
    white_wine['type'] = 'white'

    raw_df = pd.concat([red_wine, white_wine], ignore_index=True)
    print("Combined red and white wine datasets (raw).")
    return raw_df

def perform_eda(df, output_dir="outputs"):
    """
    Performs Exploratory Data Analysis (EDA) including descriptive statistics and visualizations.
    Saves plots to the output directory.
    """
    print("\n--- Performing Exploratory Data Analysis (EDA) ---")
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Histograms
    df.hist(figsize=(14, 12), bins=20)
    plt.suptitle("Histograms of Wine Chemical Properties")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "histograms_all_variables.png"))
    plt.close()
    print("Histograms saved.")
    
    # Boxplots by Wine Type
    plt.figure(figsize=(14, 12))
    for i, column in enumerate(df.columns[:-2], 1):  # Exclude 'quality' and 'type'
        plt.subplot(4, 3, i)
        sns.boxplot(x='type', y=column, data=df)
        plt.title(f'Boxplot of {column} by Wine Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplots_by_type.png"))
    plt.close()
    print("Boxplots by wine type saved.")
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Wine Chemical Properties")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    print("Correlation heatmap saved.")
    
    # Quality Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='quality', data=df, palette='viridis')
    plt.title("Distribution of Wine Quality Ratings")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_distribution.png"))
    plt.close()
    print("Quality distribution plot saved.")

def descriptive_analysis_by_quality(df):
    """Performs descriptive statistical analysis grouped by wine quality."""
    print("\n--- Descriptive Statistical Analysis by Quality ---")

    numeric_df = df.select_dtypes(include=np.number)  # Select numeric columns
    quality_groups_numeric = numeric_df.groupby('quality')

    print("\nMean Values by Quality:")
    print(quality_groups_numeric.mean())

    print("\nStandard Deviation by Quality:")
    print(quality_groups_numeric.std())

    # ANOVA (Corrected)
    print("\n--- ANOVA Results ---")
    for column in numeric_df.columns:
        if column != 'quality':  # Exclude the 'quality' column itself
            groups = [group[column].values for name, group in quality_groups_numeric]
            f_val, p_val = stats.f_oneway(*groups)
            print(f"ANOVA for {column}: F-statistic = {f_val:.2f}, p-value = {p_val:.3f}")

def perform_mlr_all_variables(df):
    """
    Performs Multiple Linear Regression using all independent variables.
    """
    print("\n--- Multiple Linear Regression (All Variables) ---")
    X = df.drop(columns=['quality', 'type'])
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    y = df['quality']
    
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    # Check for multicollinearity
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)
    
    return model

def perform_mlr_alternate_models(df):
    """
    Performs Multiple Linear Regression using two alternate models:
    - Model 2: Excludes variables with high multicollinearity (VIF > 5)
    - Model 3: Includes only statistically significant predictors (p < 0.05)
    """
    print("\n--- Multiple Linear Regression (Alternate Models) ---")
    
    # Model 2: Exclude variables with VIF > 5
    print("\n--- Model 2: Excluding Variables with VIF > 5 ---")
    X = df.drop(columns=['quality', 'type'])
    X = sm.add_constant(X)
    
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    high_vif = vif[vif["VIF"] > 5]["feature"].tolist()
    high_vif = [var for var in high_vif if var != 'const']  # Exclude constant
    print(f"Variables with VIF > 5: {high_vif}")
    
    X_model2 = X.drop(columns=high_vif)
    model2 = sm.OLS(df['quality'], X_model2).fit()
    print(model2.summary())
    
    # Model 3: Include only significant predictors (p < 0.05)
    print("\n--- Model 3: Including Only Significant Predictors (p < 0.05) ---")
    significant_vars = model2.pvalues[model2.pvalues < 0.05].index.tolist()
    significant_vars = [var for var in significant_vars if var != 'const']  # Exclude constant
    print(f"Significant variables: {significant_vars}")
    
    X_model3 = X_model2[significant_vars]
    X_model3 = sm.add_constant(X_model3)
    model3 = sm.OLS(df['quality'], X_model3).fit()
    print(model3.summary())
    
    return model2, model3

def compare_models(models, X_test, y_test, model_names=None):
    """
    Compares the performance of multiple regression models.

    Args:
        models: A list of fitted statsmodels OLS models.
        X_test: The test data (independent variables).
        y_test: The test data (dependent variable).
        model_names (optional): A list of model names for display.

    Returns:
        pandas.DataFrame: A DataFrame with model names, RMSE, and R-squared.
    """
    results = []
    for i, model in enumerate(models):
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        model_name = model_names[i] if model_names else f"Model {i+1}"
        results.append({"Model": model_name, "RMSE": rmse, "R-squared": r2})

    return pd.DataFrame(results)

def main():
    # Configuration
    red_wine_path = "data/winequality-red.csv"
    white_wine_path = "data/winequality-white.csv"
    output_directory = "outputs/simple_analsys"
    
    # Create output directories
    create_output_dirs([output_directory])
    
    # check for --raw flag
    if "--raw" in sys.argv:
        raw_df = load_raw_data(red_wine_path, white_wine_path)

        perform_eda(raw_df, output_directory)

        print("\nAnalysis Completed.")
        
        # Exit the program
        sys.exit(0)
    
    # Step 1: Data Preparation
    df = load_and_preprocess_data(red_wine_path, white_wine_path, output_directory)
    
    # Step 2: Exploratory Data Analysis (EDA)
    perform_eda(df, output_directory)
    
    # Step 3: Descriptive Statistics Analysis by Quality
    descriptive_analysis_by_quality(df)
    
    # Step 4: Multiple Linear Regression (All Variables)
    model1 = perform_mlr_all_variables(df)
    
    # Step 5: Multiple Linear Regression (Alternate Models)
    model2, model3 = perform_mlr_alternate_models(df)
    
    print("\nAnalysis Completed.")

if __name__ == "__main__":
    main()

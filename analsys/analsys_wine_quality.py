# wine_quality_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot
import os

# --- Helper Functions ---
def detect_outliers_iqr(df, column):
    """Detects outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def detect_outliers_zscore(df, column, threshold=3):
    """Detects outliers using Z-score method."""
    z = np.abs(stats.zscore(df[column]))
    outliers = df[z > threshold]
    return outliers

def handle_outliers(df, column, method='iqr', threshold=3, action='remove'):
    """Handles outliers based on specified method and action."""
    if method == 'iqr':
        outliers = detect_outliers_iqr(df, column)
    elif method == 'zscore':
        outliers = detect_outliers_zscore(df, column, threshold)
    else:
        raise ValueError("Invalid outlier detection method. Choose 'iqr' or 'zscore'.")

    if action == 'remove':
        df_filtered = df[~df.index.isin(outliers.index)].copy()
        print(f"Removed {len(outliers)} outliers from {column} using {method} method.")
        return df_filtered
    elif action == 'transform':
        # Example transformation - Winsorize
        df_transformed = df.copy()
        if not outliers.empty:
            lower_bound = df[column].quantile(0.05)
            upper_bound = df[column].quantile(0.95)
            df_transformed[column] = np.clip(df_transformed[column], lower_bound, upper_bound)
        print(f"Transformed outliers in {column} using {method} method and Winsorizing.")
        return df_transformed
    else:
        raise ValueError("Invalid action for outliers. Choose 'remove' or 'transform'.")

def check_multicollinearity(df, exclude_cols=None):
    """Checks for multicollinearity using VIF."""
    if exclude_cols is None:
        exclude_cols = []
    X = df.drop(columns=exclude_cols)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nMulticollinearity Check (VIF):")
    print(vif_data)
    return vif_data

def plot_residuals(model, fitted_values, title="Residual Plot"):
    """Plots residuals vs fitted values."""
    plt.figure(figsize=(8, 6))
    sns.residplot(x=fitted_values, y=model.resid, lowess=True, line_kws={'color': 'red', 'lw': 1})
    plt.title(title)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.show()

def create_output_dirs(dirs):
    """Creates directories if they don't exist."""
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# --- Data Import and Preprocessing ---
def load_and_preprocess_data(red_wine_path, white_wine_path, output_dir="outputs"):
    """Loads, preprocesses, and saves cleaned data."""
    create_output_dirs([output_dir])

    red_wine = pd.read_csv(red_wine_path, sep=";")
    white_wine = pd.read_csv(white_wine_path, sep=";")

    # Combine datasets with a type indicator
    red_wine['type'] = 'red'
    white_wine['type'] = 'white'
    df = pd.concat([red_wine, white_wine], ignore_index=True)

    # --- Outlier Handling ---
    # Example of handling outliers for 'fixed acidity' using IQR and removal
    df = handle_outliers(df, 'fixed acidity', method='iqr', action='remove')
    # Example of handling outliers for 'volatile acidity' using Z-score and transformation
    df = handle_outliers(df, 'volatile acidity', method='zscore', action='transform')

    # Save the preprocessed data
    df.to_csv(os.path.join(output_dir, "preprocessed_wine_data.csv"), index=False)
    print("\nData loaded, preprocessed, and saved to 'outputs/preprocessed_wine_data.csv'")
    return df

# --- Exploratory Data Analysis (EDA) ---
def perform_eda(df, output_dir="outputs"):
    """Performs EDA and saves visualizations."""
    create_output_dirs([output_dir])

    print("\n--- Exploratory Data Analysis ---")
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Histograms for all variables
    print("\nGenerating Histograms...")
    df.hist(figsize=(14, 12), bins=20)
    plt.suptitle("Histograms of Wine Chemical Properties")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "histograms_all_variables.png"))
    plt.show()

    # Boxplots for all variables
    print("\nGenerating Boxplots...")
    plt.figure(figsize=(14, 12))
    for i, col in enumerate(df.columns[:-1]):  # Exclude 'type' column
        plt.subplot(4, 3, i + 1)
        sns.boxplot(x='type', y=col, data=df)
        plt.title(f'Boxplot of {col} by Wine Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplots_by_type.png"))
    plt.show()

    # Correlation heatmap
    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Wine Chemical Properties")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.show()

    # Quality distribution
    print("\nGenerating Quality Distribution Plots...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='quality', data=df)
    plt.title("Distribution of Wine Quality Ratings")
    plt.savefig(os.path.join(output_dir, "quality_distribution.png"))
    plt.show()

    print("\nEDA completed and visualizations saved to 'outputs/' directory.")

# wine_quality_analysis.py CONTINUED

# --- Descriptive Statistical Analysis by Quality ---
def descriptive_analysis_by_quality(df):
    """Performs descriptive analysis grouped by quality."""
    print("\n--- Descriptive Statistical Analysis by Quality ---")
    quality_groups = df.groupby('quality')
    print("\nMean by Quality:")
    print(quality_groups.mean(numeric_only=True))
    print("\nStandard Deviation by Quality:")
    print(quality_groups.std(numeric_only=True))

    # ANOVA test for each feature
    print("\n--- ANOVA Results ---")
    for col in df.columns:
        if col not in ['quality', 'type']:
            fvalue, pvalue = stats.f_oneway(*[group[col].values for name, group in quality_groups])
            print(f"ANOVA for {col}: F-statistic = {fvalue:.2f}, p-value = {pvalue:.3f}")

# --- Inferential Statistical Analysis ---
def inferential_analysis(df, output_dir="outputs"):
    """Performs inferential analysis using multiple linear regression."""
    create_output_dirs([output_dir])

    print("\n--- Inferential Statistical Analysis ---")
    print("\nFormulating Hypotheses:")
    print("Null Hypothesis (H₀): Chemical properties do not significantly affect wine quality.")
    print("Alternative Hypothesis (H₁): At least one chemical property significantly affects wine quality.")

    print("\n--- Multiple Linear Regression ---")
    # Prepare data for regression
    X = df.drop(columns=['quality', 'type'])  # Drop 'quality' and 'type' for now
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features if any
    X = sm.add_constant(X)  # Add a constant term for the intercept
    y = df['quality']

    # Fit the model
    model = sm.OLS(y, X).fit()
    print("\nRegression Summary:")
    print(model.summary())

    # Check for multicollinearity
    check_multicollinearity(X, exclude_cols=['const'])

    # Model Evaluation
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\nModel Evaluation - MSE: {mse:.2f}, R-squared: {r2:.2f}")

    # Assumption Checks
    print("\n--- Model Assumption Checks ---")
    # Normality of Residuals
    print("\nGenerating Q-Q Plot for Normality of Residuals...")
    plt.figure(figsize=(8, 6))
    qqplot(model.resid, line='q', fit=True)
    plt.title("Q-Q Plot of Residuals")
    plt.savefig(os.path.join(output_dir, "qq_plot_residuals.png"))
    plt.show()

    # Homoscedasticity
    print("\nGenerating Residual Plot for Homoscedasticity...")
    plot_residuals(model, y_pred, title="Residual Plot")
    plt.savefig(os.path.join(output_dir, "residual_plot.png"))
    plt.show()

    # Independence of Residuals (Durbin-Watson is in the regression summary)

    # --- Interpretation of Results ---
    print("\n--- Interpretation of Results ---")
    significant_predictors = model.pvalues[model.pvalues < 0.05]
    print("\nSignificant Predictors (p < 0.05):")
    print(significant_predictors)

    print("\nCoefficients:")
    print(model.params)

    # You can add more detailed interpretation and discussion here based on the model results

    return model

# --- Visualization ---
def visualize_results(df, model, output_dir="outputs"):
    """Generates visualizations for results interpretation."""
    create_output_dirs([output_dir])

    print("\n--- Visualization ---")

    # Scatter Plots of Significant Predictors vs. Quality
    print("\nGenerating Scatter Plots of Significant Predictors vs. Quality...")
    significant_predictors = model.pvalues[model.pvalues < 0.05].index.drop('const', errors='ignore')
    for predictor in significant_predictors:
        if predictor in df.columns:
            plt.figure(figsize=(8, 6))
            sns.regplot(x=df[predictor], y=df['quality'], scatter_kws={'s':10}, line_kws={'color':'red'})
            plt.title(f"Scatter Plot of {predictor} vs. Wine Quality")
            plt.xlabel(predictor)
            plt.ylabel("Quality")
            plt.savefig(os.path.join(output_dir, f"scatter_plot_{predictor}_vs_quality.png"))
            plt.show()

    print("\nVisualizations saved to 'outputs/' directory.")

# --- Recommendations and Insights ---
def provide_recommendations(model):
    """Provides recommendations based on the analysis results."""
    print("\n--- Recommendations and Insights ---")
    print("\nKey Findings:")
    significant_predictors = model.pvalues[model.pvalues < 0.05]
    if not significant_predictors.empty:
        print("The following chemical properties were found to significantly affect wine quality (p < 0.05):")
        for predictor, p_value in significant_predictors.items():
            if predictor != 'const':
                coefficient = model.params[predictor]
                direction = "positively" if coefficient > 0 else "negatively"
                print(f"- {predictor}: coefficient = {coefficient:.2f}, p-value = {p_value:.3f} (affects quality {direction})")
    else:
        print("No chemical properties were found to have a statistically significant impact on wine quality at the 0.05 significance level.")

    print("\nActionable Recommendations for Wine Producers:")
    if not significant_predictors.empty:
        for predictor, p_value in significant_predictors.items():
            if predictor != 'const':
                coefficient = model.params[predictor]
                if coefficient > 0:
                    print(f"- Consider enhancing the levels of {predictor} to potentially improve wine quality.")
                else:
                    print(f"- Consider reducing the levels of {predictor} to potentially improve wine quality.")
    else:
        print("- Since no significant predictors were found, focus on maintaining consistency in production processes and explore other factors that might influence quality.")

    print("\nFurther Research:")
    print("- Explore non-linear relationships between chemical properties and wine quality.")
    print("- Investigate the interaction effects between different chemical properties.")
    print("- Consider additional factors such as vineyard location, climate data, and winemaking techniques.")

# --- Main Execution ---
if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error, r2_score
    print("Starting Wine Quality Analysis...")

    # --- Configuration ---
    red_wine_path = r"C:\Users\Alex\Desktop\VSCode\prin_of_lang\Data-Analsys-Project\data\winequality-red.csv"
    white_wine_path = r"C:\Users\Alex\Desktop\VSCode\prin_of_lang\Data-Analsys-Project\data\winequality-white.csv"
    output_directory = "outputs"
    create_output_dirs([output_directory])

    # --- Step 1: Data Import and Preprocessing ---
    wine_data = load_and_preprocess_data(red_wine_path, white_wine_path, output_directory)

    # --- Step 2: Exploratory Data Analysis (EDA) ---
    perform_eda(wine_data.copy(), output_directory)

    # --- Step 3: Descriptive Statistical Analysis by Quality ---
    descriptive_analysis_by_quality(wine_data.copy())

    # --- Step 4: Inferential Statistical Analysis ---
    regression_model = inferential_analysis(wine_data.copy(), output_directory)

    # --- Step 5: Visualization ---
    visualize_results(wine_data.copy(), regression_model, output_directory)

    # --- Step 6: Recommendations and Insights ---
    provide_recommendations(regression_model)

    print("\nWine Quality Analysis Completed.")
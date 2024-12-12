# Wine Quality Analysis Project

## Overview

This project analyzes the Wine Quality Dataset from the UCI Machine Learning Repository. It explores the relationship between various physicochemical properties of wine and its perceived quality. The analysis includes data loading, preprocessing, exploratory data analysis (EDA), and simple linear regression modeling.

## Project Structure

-   **`Presentation_code.py`**: Main script containing the analysis workflow.
-   **`data/`**: Directory for storing the raw wine quality datasets (`winequality-red.csv` and `winequality-white.csv`).
-   **`outputs/`**: Directory to save generated visualizations and preprocessed data.
-   **`outputs/final_vis`**: Directory to save the final version of the visualizations.
-   **`Wine Quality Analysis.md`**: Markdown file containing a detailed report of the analysis.

## Getting Started

### Prerequisites

-   Python 3.x
-   Required Python packages (install using `pip install -r requirements.txt`):
    -   pandas
    -   numpy
    -   matplotlib
    -   seaborn
    -   scikit-learn

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/axelgardenier/Data-Analsys-Project.git
    cd Data-Analsys-Project
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    ```

3. **Activate the virtual environment:**

    -   On Windows:

        ```bash
        .\\venv\\Scripts\\activate
        ```

    -   On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis

1. **Execute the `Presentation_code.py` script:**

    ```bash
    python Presentation_code.py
    ```

2. **View the results:**

    -   The script will print descriptive statistics, regression results, and other analysis output to the console.
    -   Visualizations (histograms, correlation heatmap, scatter plots) will be saved in the `outputs/final_vis` directory.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
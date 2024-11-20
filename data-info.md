# Data Analysis Project Summary

## Project Overview

This project involves an initial analysis of a dataset, focusing on various key statistics, data integrity checks, and insights. Below are the results of the exploratory data analysis conducted with `init_analsys.py`.

### Dataset Overview

- **Number of Rows**: 6,497  
- **Number of Columns**: 11

#### Column Names:

- `fixed_acidity`  
- `volatile_acidity`  
- `citric_acid`  
- `residual_sugar`  
- `chlorides`  
- `free_sulfur_dioxide`  
- `total_sulfur_dioxide`  
- `density`  
- `pH`  
- `sulphates`  
- `alcohol`

### Data Types

All columns in the dataset are of type `float64`.

### Missing Values

There are **no missing values** in any of the columns. Each column has complete data.

### Summary Statistics

The table below provides a brief summary of the dataset, including count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum values:

| Column | Count | Mean | Std Dev | Min | 25% | 50% | 75% | Max |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| `fixed_acidity` | 6497 | 7.215 | 1.296 | 3.8 | 6.4 | 7.0 | 7.7 | 15.9 |
| `volatile_acidity` | 6497 | 0.340 | 0.165 | 0.08 | 0.23 | 0.29 | 0.40 | 1.58 |
| `citric_acid` | 6497 | 0.319 | 0.145 | 0.0 | 0.25 | 0.31 | 0.39 | 1.66 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `pH` | 6497 | 3.219 | 0.161 | 2.72 | 3.11 | 3.21 | 3.32 | 4.01 |
| `sulphates` | 6497 | 0.531 | 0.149 | 0.22 | 0.43 | 0.51 | 0.60 | 2.00 |
| `alcohol` | 6497 | 10.492 | 1.193 | 8.0 | 9.5 | 10.3 | 11.3 | 14.9 |

### Unique Values

The number of unique values for each column varies, for example:

- `fixed_acidity`: 106 unique values  
- `volatile_acidity`: 187 unique values  
- `residual_sugar`: 316 unique values  
- `density`: 998 unique values  
- `alcohol`: 111 unique values

### Frequency of Unique Values

Below is a glimpse of the most frequent values in some key columns:

- **`fixed_acidity`**:  
  - `6.80`: 354 occurrences  
  - `6.60`: 327 occurrences  
  - `6.40`: 305 occurrences  
- **`volatile_acidity`**:  
  - `0.280`: 286 occurrences  
  - `0.240`: 266 occurrences  
  - `0.260`: 256 occurrences

### Correlation Matrix

The following correlation matrix shows the relationships between the numerical variables in the dataset:

|  | `fixed_acidity` | `volatile_acidity` | `citric_acid` | ... | `sulphates` | `alcohol` |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **`fixed_acidity`** | 1.000 | 0.219 | 0.324 | ... | 0.299 | \-0.095 |
| **`volatile_acidity`** | 0.219 | 1.000 | \-0.378 | ... | 0.226 | \-0.038 |
| **`citric_acid`** | 0.324 | \-0.378 | 1.000 | ... | 0.056 | \-0.010 |
| ... | ... | ... | ... | ... | ... | ... |
| **`alcohol`** | \-0.095 | \-0.038 | \-0.010 | ... | \-0.003 | 1.000 |

### Key Insights

- There is a moderate positive correlation between `fixed_acidity` and `citric_acid` (`0.324`).  
- `density` and `alcohol` have a strong negative correlation (`-0.687`), suggesting that higher alcohol content is associated with lower density.  
- `chlorides` and `volatile_acidity` show a positive correlation (`0.377`), implying a potential relationship between the two properties.

## Further Analysis

Based on the exploratory analysis:

- Further modeling can explore relationships between key features (e.g., alcohol content vs. density).  
- Feature selection might focus on removing low-variance features to reduce model complexity.  
- Further analysis could consider clustering wines by acidity levels or employing regression to predict alcohol content.

### How to Run

To run the analysis script:

$ python init\_analsys.py  
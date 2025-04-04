# Principal Component Analysis (PCA) Tutorial

This repository contains a Python script for learning Principal Component Analysis (PCA). The script demonstrates the complete PCA workflow from data loading to visualization and interpretation.

## Setup Instructions

1. **Clone the repository** (if applicable)

2. **Create a virtual environment**
   ```bash
   python -m venv pca_venv
   ```

3. **Activate the virtual environment**
   - On macOS/Linux:
     ```bash
     source pca_venv/bin/activate
     ```
   - On Windows:
     ```bash
     pca_venv\Scripts\activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the PCA script**
   ```bash
   python pca_analysis.py
   ```

## Command Line Arguments

The script supports several command-line arguments:

- `--csv PATH` - Path to your CSV data file (uses the Iris dataset by default)
- `--save-plots` - Save plots to 'pca_output' directory instead of displaying them
- `--variance FLOAT` - Target explained variance (default: 0.95)
- `--components INT` - Force a specific number of components

Example usage:
```bash
# Run with default settings (Iris dataset)
python pca_analysis.py

# Run with your own CSV file
python pca_analysis.py --csv your_data.csv

# Save plots to files instead of displaying them
python pca_analysis.py --save-plots

# Specify target variance and number of components
python pca_analysis.py --variance 0.9 --components 2
```

## What You'll Learn

- Data preprocessing for PCA
- Implementing PCA using scikit-learn
- Determining the optimal number of principal components
- Visualizing PCA results
- Interpreting principal components in terms of original features

## Dataset

The script uses the Iris dataset by default, but you can specify your own CSV data file using the `--csv` argument.

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Output

When using the `--save-plots` option, the script will:

1. Create a directory called 'pca_output'
2. Save all visualizations as PNG files in this directory
3. Display text output in the terminal showing the analysis progress and results
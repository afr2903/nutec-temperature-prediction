#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Principal Component Analysis (PCA) Tutorial Script

This script demonstrates a complete Principal Component Analysis workflow
from data loading to visualization and interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import os

# Set plotting style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# For reproducibility
np.random.seed(42)

def create_output_dir():
    """Create output directory for saving plots if it doesn't exist."""
    os.makedirs('pca_output', exist_ok=True)
    print("Output directory 'pca_output' created or already exists.")

def load_data(csv_path=None):
    """
    Load data for PCA analysis.
    
    Args:
        csv_path: Path to CSV file. If None, use Iris dataset.
        
    Returns:
        DataFrame containing the data.
    """
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            print(f"Data loaded successfully from {csv_path}!")
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}")
            print("Falling back to Iris dataset...")
            return load_iris_dataset()
    else:
        return load_iris_dataset()
    
    return df

def load_iris_dataset():
    """Load the Iris dataset as a fallback."""
    print("Loading Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

def explore_data(df):
    """
    Perform basic data exploration and return numerical columns.
    
    Args:
        df: DataFrame to explore.
        
    Returns:
        List of numerical column names suitable for PCA.
    """
    print("First 5 rows:")
    print(df.head())
    
    print("\nData Info:")
    df.info()
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Select numerical columns (removing target if present)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'target' in numerical_cols:
        numerical_cols.remove('target')
    
    print(f"\nUsing numerical columns: {numerical_cols}")
    
    # Check for missing values
    print("\nMissing values count:")
    print(df[numerical_cols].isnull().sum())
    
    return numerical_cols

def visualize_feature_distributions(df, numerical_cols, show_plots=True):
    """
    Create visualizations of the original data features.
    
    Args:
        df: DataFrame containing the data.
        numerical_cols: List of numerical column names.
        show_plots: Whether to display plots (True) or save to file (False).
    """
    # If species column exists, use it for coloring
    hue_col = 'species' if 'species' in df.columns else None
    
    # Create a pair plot if there are at least 2 numerical columns
    if len(numerical_cols) >= 2:
        print("Creating pair plot of features...")
        plt.figure(figsize=(12, 10))
        pair_plot = sns.pairplot(df, vars=numerical_cols, hue=hue_col, height=2.5)
        plt.suptitle('Pair Plot of Features', y=1.02)
        
        if show_plots:
            plt.show()
        else:
            pair_plot.savefig('pca_output/feature_pairplot.png')
            plt.close()
            print("Pair plot saved to 'pca_output/feature_pairplot.png'")

def preprocess_data(df, numerical_cols):
    """
    Preprocess data for PCA by standardizing.
    
    Args:
        df: DataFrame containing the data.
        numerical_cols: List of numerical column names.
        
    Returns:
        Tuple of (standardized data array, DataFrame with standardized data).
    """
    data_for_pca = df[numerical_cols]
    
    # Standardize the data (zero mean and unit variance)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_pca)
    print("Data shape after scaling:", scaled_data.shape)
    
    # Create DataFrame with scaled data (for visualization)
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)
    print("First 5 rows of scaled data:")
    print(scaled_df.head())
    
    return scaled_data, scaled_df

def perform_pca_analysis(scaled_data):
    """
    Perform initial PCA to analyze variance explained.
    
    Args:
        scaled_data: Standardized data array.
        
    Returns:
        Tuple of (PCA object, explained variance ratio, cumulative variance).
    """
    # Initialize PCA without specifying n_components
    pca = PCA()
    pca.fit(scaled_data)
    
    # Examine the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\nExplained Variance Ratio per Component:")
    print(explained_variance_ratio)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print("\nCumulative Explained Variance:")
    print(cumulative_variance)
    
    return pca, explained_variance_ratio, cumulative_variance

def visualize_explained_variance(explained_variance_ratio, cumulative_variance, show_plots=True):
    """
    Create scree plot to visualize explained variance.
    
    Args:
        explained_variance_ratio: Array of explained variance ratios.
        cumulative_variance: Array of cumulative explained variance.
        show_plots: Whether to display plots (True) or save to file (False).
    """
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7)
    plt.step(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, where='mid')
    plt.title('Explained Variance by Each Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    
    # Plot 2: Cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.xticks(range(1, len(cumulative_variance) + 1))
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance Threshold')
    plt.legend()
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    else:
        plt.savefig('pca_output/explained_variance_plots.png')
        plt.close()
        print("Explained variance plots saved to 'pca_output/explained_variance_plots.png'")

def determine_optimal_components(cumulative_variance, target_variance=0.95):
    """
    Determine optimal number of components based on explained variance.
    
    Args:
        cumulative_variance: Array of cumulative explained variance.
        target_variance: Target proportion of variance to explain (default: 0.95).
        
    Returns:
        Number of components needed to reach target variance.
    """
    n_components = np.argmax(cumulative_variance >= target_variance) + 1
    print(f"\nNumber of components to capture >= {target_variance*100}% variance: {n_components}")
    return n_components

def reduce_dimensions(scaled_data, n_components, df=None, target_col=None):
    """
    Apply PCA with specified number of components to reduce dimensionality.
    
    Args:
        scaled_data: Standardized data array.
        n_components: Number of components to use.
        df: Original DataFrame (optional).
        target_col: Name of target column for coloring (optional).
        
    Returns:
        Tuple of (PCA object, DataFrame with principal components).
    """
    # Apply PCA with n_components
    pca_final = PCA(n_components=n_components)
    principal_components = pca_final.fit_transform(scaled_data)
    
    # Create DataFrame for the principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Add target variable if provided
    if df is not None and target_col is not None and target_col in df.columns:
        pca_df[target_col] = df[target_col]
    
    print(f"\nShape of data after PCA (n_components={n_components}):", pca_df.shape)
    print("\nFirst 5 rows of PCA results:")
    print(pca_df.head())
    
    return pca_final, pca_df

def visualize_reduced_data(pca_df, hue_col=None, show_plots=True):
    """
    Visualize the first two principal components.
    
    Args:
        pca_df: DataFrame with principal components.
        hue_col: Column name to use for coloring points (optional).
        show_plots: Whether to display plots (True) or save to file (False).
    """
    if 'PC1' not in pca_df.columns or 'PC2' not in pca_df.columns:
        print("Error: PC1 and PC2 columns not found in PCA results.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    if hue_col and hue_col in pca_df.columns:
        scatter = sns.scatterplot(x='PC1', y='PC2', hue=hue_col, data=pca_df, s=100, alpha=0.8, palette='viridis')
        
        # Add annotations for centroids of each group
        for group in pca_df[hue_col].unique():
            subset = pca_df[pca_df[hue_col] == group]
            center_x = subset['PC1'].mean()
            center_y = subset['PC2'].mean()
            plt.annotate(group, (center_x, center_y), fontsize=12, 
                         ha='center', va='center', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.legend(title=hue_col.capitalize(), fontsize=10)
    else:
        scatter = sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=100, alpha=0.8)
    
    # Add title and labels
    plt.title('PCA: Data Projected onto First Two Principal Components', fontsize=15)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    else:
        plt.savefig('pca_output/pca_scatter_plot.png')
        plt.close()
        print("PCA scatter plot saved to 'pca_output/pca_scatter_plot.png'")

def analyze_feature_contributions(pca_model, feature_names, show_plots=True):
    """
    Analyze and visualize how original features contribute to principal components.
    
    Args:
        pca_model: Fitted PCA model.
        feature_names: List of original feature names.
        show_plots: Whether to display plots (True) or save to file (False).
    """
    n_components = pca_model.components_.shape[0]
    
    # Get the loadings (coefficients)
    loadings = pd.DataFrame(
        data=pca_model.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    print("Loadings (Feature Contributions to Each Principal Component):")
    print(loadings)
    
    # Visualize the loadings
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".3f")
    plt.title('Feature Contributions to Principal Components', fontsize=15)
    plt.ylabel('Original Features', fontsize=12)
    plt.xlabel('Principal Components', fontsize=12)
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    else:
        plt.savefig('pca_output/feature_contributions_heatmap.png')
        plt.close()
        print("Feature contributions heatmap saved to 'pca_output/feature_contributions_heatmap.png'")
    
    return loadings

def summarize_pca_results(pca_model, loadings, k):
    """
    Summarize the PCA results.
    
    Args:
        pca_model: Fitted PCA model.
        loadings: DataFrame with feature loadings.
        k: Number of components used.
    """
    # Calculate the total variance explained
    explained_variance_ratio = pca_model.explained_variance_ratio_
    total_variance_explained = sum(explained_variance_ratio[:k])
    print(f"\nThe first {k} principal components explain {total_variance_explained:.2%} of the total variance in the data.")
    
    # Find the most important feature for each principal component
    for i in range(k):
        pc = f'PC{i+1}'
        # Get absolute loadings for this component
        abs_loadings = loadings[pc].abs()
        # Find feature with maximum absolute loading
        max_feature = abs_loadings.idxmax()
        max_value = loadings.loc[max_feature, pc]
        print(f"\n{pc} is most strongly influenced by '{max_feature}' with a loading of {max_value:.3f}")
        
        # Get top 2 contributing features
        top_features = abs_loadings.sort_values(ascending=False).head(2).index.tolist()
        print(f"Top features contributing to {pc}: {', '.join(top_features)}")

def main(csv_path=None, show_plots=True, target_variance=0.95, components_override=None):
    """
    Run the complete PCA analysis workflow.
    
    Args:
        csv_path: Path to CSV file (optional).
        show_plots: Whether to display plots (True) or save to file (False).
        target_variance: Target proportion of variance to explain (default: 0.95).
        components_override: Override automatic component selection with this value (optional).
    """
    # Create output directory if saving plots
    if not show_plots:
        create_output_dir()
    
    # Step 1: Load Data
    df = load_data(csv_path)
    
    # Step 2: Explore Data
    numerical_cols = explore_data(df)
    
    # Step 3: Visualize Feature Distributions
    visualize_feature_distributions(df, numerical_cols, show_plots)
    
    # Step 4: Preprocess Data
    scaled_data, scaled_df = preprocess_data(df, numerical_cols)
    
    # Step 5: Perform initial PCA analysis
    pca, explained_variance_ratio, cumulative_variance = perform_pca_analysis(scaled_data)
    
    # Step 6: Visualize explained variance
    visualize_explained_variance(explained_variance_ratio, cumulative_variance, show_plots)
    
    # Step 7: Determine optimal number of components
    n_components = determine_optimal_components(cumulative_variance, target_variance)
    
    # Allow override of calculated components
    if components_override is not None:
        n_components = min(components_override, len(numerical_cols))
        print(f"Overriding with {n_components} components as specified.")
    
    # For visualization purposes, ensure at least 2 components if possible
    k = max(2, n_components) if len(numerical_cols) >= 2 else n_components
    
    # Step 8: Reduce dimensions
    hue_col = 'species' if 'species' in df.columns else None
    pca_final, pca_df = reduce_dimensions(scaled_data, k, df, hue_col)
    
    # Step 9: Visualize reduced data (if at least 2 components)
    if k >= 2:
        visualize_reduced_data(pca_df, hue_col, show_plots)
    
    # Step 10: Analyze feature contributions
    loadings = analyze_feature_contributions(pca_final, numerical_cols, show_plots)
    
    # Step 11: Summarize results
    summarize_pca_results(pca_final, loadings, k)
    
    print("\nPCA analysis complete!")
    return pca_final, pca_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform PCA analysis on a dataset.')
    parser.add_argument('--csv', type=str, help='Path to CSV file (optional, uses Iris dataset if not provided)')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files instead of displaying them')
    parser.add_argument('--variance', type=float, default=0.95, help='Target explained variance (default: 0.95)')
    parser.add_argument('--components', type=int, help='Force specific number of components (optional)')
    
    args = parser.parse_args()
    
    # Run the analysis
    main(
        csv_path=args.csv,
        show_plots=not args.save_plots,
        target_variance=args.variance,
        components_override=args.components
    ) 
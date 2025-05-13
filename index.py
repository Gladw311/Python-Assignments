# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# --- Data Loading and Exploration ---
print("--- Data Exploration ---")

# Display the first few rows of the dataframe
print("\nFirst 5 rows of the dataframe:")
print(df.head())

# Get a concise summary of the dataframe, including data types and non-null values
print("\nSummary of the dataframe:")
df.info()

# Generate descriptive statistics of the numerical columns
print("\nDescriptive statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Explore the distribution of the target variable (species)
print("\nValue counts for the target variable:")
print(df['target'].value_counts())

# --- Basic Data Analysis Results ---
print("\n--- Basic Data Analysis ---")

# Calculate the mean of each feature for each species
mean_features_by_species = df.groupby('target').mean()
print("\nMean features by species:")
print(mean_features_by_species)

# Calculate the correlation matrix
correlation_matrix = df.corr(numeric_only=True)
print("\nCorrelation matrix:")
print(correlation_matrix)

# --- Visualizations ---
print("\n--- Visualizations ---")

# 1. Scatter plot of sepal length vs sepal width, colored by species
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=df)
plt.title('Sepal Length vs Sepal Width by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.grid(True)
plt.show()

# 2. Histogram of petal length for each species
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='petal length (cm)', hue='target', kde=True, multiple='stack')
plt.title('Distribution of Petal Length by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.legend(title='Species')
plt.show()

# 3. Box plot of petal width for each species
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='petal width (cm)', data=df)
plt.title('Petal Width Distribution by Species')
plt.xlabel('Species')
plt.ylabel('Petal Width (cm)')
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)
plt.grid(True)
plt.show()

# 4. Pair plot to visualize relationships between all pairs of features, colored by species
plt.figure(figsize=(10, 10))
sns.pairplot(df, hue='target')
plt.suptitle('Pair Plot of Iris Dataset Features', y=1.02)
plt.show()

# 5. Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Iris Dataset Features')
plt.show()

# --- Findings and Observations ---
print("\n--- Findings and Observations ---")
print("\nBased on the data exploration and visualizations:")
print("- The Iris dataset contains measurements for sepal length, sepal width, petal length, and petal width for three different species of Iris flowers (setosa, versicolor, virginica).")
print("- There are no missing values in this dataset.")
print("- The descriptive statistics show differences in the average measurements across the features.")
print("- The correlation matrix indicates some strong positive correlations, for example, between petal length and petal width.")
print("- The scatter plot of sepal length vs sepal width shows some overlap between the species, particularly versicolor and virginica.")
print("- The histograms and box plots reveal distinct distributions of petal length and petal width for each species, with setosa generally having smaller petal dimensions.")
print("- The pair plot provides a comprehensive view of the relationships between all pairs of features, highlighting how the species cluster based on these measurements.")
print("- The heatmap visually confirms the strong positive correlations observed earlier.")

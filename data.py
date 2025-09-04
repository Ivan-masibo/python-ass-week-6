
# Assignment: Analyzing Data with Pandas and Matplotlib

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# Task 1: Load and Explore Dataset


try:
    # Option 1: Load dataset from sklearn (Iris)
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Option 2: Load CSV dataset (uncomment below if using CSV)
    # df = pd.read_csv("your_dataset.csv")

    print("Dataset loaded successfully!\n")

except FileNotFoundError:
    print("Error: Dataset file not found. Please check your file path.")
except Exception as e:
    print("An error occurred while loading dataset:", e)

# Display first rows
print("First 5 rows of dataset:")
print(df.head())

# Check structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Clean data (drop missing rows if any)
df = df.dropna()


# Task 2: Basic Data Analysis


# Descriptive statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean
grouped = df.groupby("target").mean()
print("\nMean values per Species (0=setosa, 1=versicolor, 2=virginica):")
print(grouped)

# Interesting observation
print("\nObservation: Virginica tends to have the largest petal length and width on average.")


# Task 3: Data Visualization


# Set style
sns.set(style="whitegrid")

# 1. Line chart (trend example: cumulative petal length per row index)
plt.figure(figsize=(8,5))
df["petal length (cm)"].cumsum().plot(kind="line")
plt.title("Cumulative Petal Length")
plt.xlabel("Index")
plt.ylabel("Cumulative Petal Length (cm)")
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(8,5))
df.groupby("target")["petal length (cm)"].mean().plot(kind="bar", color="skyblue")
plt.title("Average Petal Length per Species")
plt.xlabel("Species (0=setosa,1=versicolor,2=virginica)")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal length)
plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"], bins=20, color="lightgreen", edgecolor="black")
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (relationship between sepal length and petal length)
plt.figure(figsize=(8,5))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], alpha=0.7, c=df["target"], cmap="viridis")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.show()

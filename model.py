from google.colab import files

# Upload your dataset
uploaded = files.upload()

# Install necessary libraries (if not installed)
# pip install pandas numpy scikit-learn surprise matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns

#read the dataset
df = pd.read_csv("amazon.csv")
print(df.head())

# Unique users and products
print(f"Unique users: {df['user_id'].nunique()}, Unique products: {df['product_id'].nunique()}")

# Product popularity
product_popularity = df['product_id'].value_counts()
sns.barplot(x=product_popularity.index, y=product_popularity.values)
plt.title("Product Popularity")
plt.xlabel("Product ID")
plt.ylabel("Number of Purchases")
plt.show()

# User activity
user_activity = df['user_id'].value_counts()
sns.histplot(user_activity, bins=5, kde=True)
plt.title("User Activity Distribution")
plt.xlabel("Number of Purchases")
plt.ylabel("Frequency")
plt.show()

# Inspect the 'rating' column
print("Unique values in 'rating':", df['rating'].unique())

# Clean the 'rating' column
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # Convert to numeric
df = df.dropna(subset=['rating'])  # Drop rows with NaN values
df['rating'] = df['rating'].astype(float)  # Ensure numeric type

# Validate ratings
if df['rating'].min() < 1 or df['rating'].max() > 5:
    raise ValueError("Ratings must be within the range of 1 to 5.")

# Load the dataset into Surprise
from surprise import Dataset, Reader
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

print("Data loaded successfully into Surprise!")


# Train-test split
trainset, testset = train_test_split(data.build_full_trainset().build_testset(), test_size=0.2, random_state=42)

# Train SVD (Singular Value Decomposition) model
svd = SVD()
svd.fit(data.build_full_trainset())

# Evaluate model performance
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Predict ratings for a specific user
user_id = 1
all_product_ids = df['product_id'].unique()
predictions = []

for product_id in all_product_ids:
    pred = svd.predict(user_id, product_id).est
    predictions.append((product_id, pred))

# Sort predictions by estimated rating
recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
print("Top Recommendations for User 1:")
for product_id, rating in recommendations[:5]:
    print(f"Product ID: {product_id}, Predicted Rating: {rating:.2f}")
# Predict ratings for a specific user
user_id = 'AG3D6O4STAQKAY2UVGEUV46KN35Q'  # Replace with an actual user ID from your dataset
all_product_ids = df['product_id'].unique()
predictions = []

for product_id in all_product_ids:
    pred = svd.predict(user_id, product_id).est
    predictions.append((product_id, pred))
  

# Sort predictions by estimated rating
recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
print("Top Recommendations for User:")
for product_id, rating in recommendations[:5]:
    print(f"Product ID: {product_id}, Predicted Rating: {rating:.2f}")

# Unique users and products
print(f"Unique users: {df['user_id'].nunique()}, Unique products: {df['product_id'].nunique()}")

# Product popularity visualization
product_popularity = df['product_id'].value_counts()
sns.barplot(x=product_popularity.index[:10], y=product_popularity.values[:10])  # Show top 10 products
plt.title("Product Popularity")
plt.xlabel("Product ID")
plt.ylabel("Number of Purchases")
plt.xticks(rotation=90)  # Rotate x labels for better visibility
plt.show()

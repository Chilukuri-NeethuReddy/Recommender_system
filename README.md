#Product Recommendation System

Overview
This project implements a product recommendation system using the Amazon dataset. It leverages the Singular Value Decomposition (SVD) algorithm from the Surprise library to analyze user-item interactions and generate personalized product recommendations. The project also includes data exploration, user activity visualization, and insights into product popularity.

Dataset
The dataset used in this project is sourced from Amazon and contains the following columns:

product_id: Unique identifier for each product.
product_name: Name of the product.
category: Category of the product.
discounted_price: Price after discount.
actual_price: Original price before discount.
rating: User rating of the product.
user_id: Unique identifier for each user.

Installation
To run this project in Google Colab, ensure you have the following libraries installed:
```python
!pip install pandas numpy scikit-learn surprise matplotlib seaborn
```

Here’s a polished and formatted version of your README for the Product Recommendation System:

Product Recommendation System
Overview
This project implements a product recommendation system using the Amazon dataset. It leverages the Singular Value Decomposition (SVD) algorithm from the Surprise library to analyze user-item interactions and generate personalized product recommendations. The project also includes data exploration, user activity visualization, and insights into product popularity.

Dataset
The dataset used in this project is sourced from Amazon and contains the following columns:

product_id: Unique identifier for each product.
product_name: Name of the product.
category: Category of the product.
discounted_price: Price after discount.
actual_price: Original price before discount.
rating: User rating of the product.
user_id: Unique identifier for each user.


Installation
To run this project in Google Colab, ensure you have the following libraries installed:
```python
!pip install pandas numpy scikit-learn surprise matplotlib seaborn
```

Instructions for Running the Model
1. Upload the Dataset
Upload the dataset (amazon.csv) to Google Colab using the following code:
```python
from google.colab import files

# Upload the dataset
uploaded = files.upload()
```

2. Load and Prepare Data
Load the dataset into a pandas DataFrame, explore it, and clean the data:

python
Copy code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("amazon.csv")
print(df.head())

# Display unique users and products
print(f"Unique users: {df['user_id'].nunique()}, Unique products: {df['product_id'].nunique()}")

# Product popularity visualization
product_popularity = df['product_id'].value_counts()
sns.barplot(x=product_popularity.index[:10], y=product_popularity.values[:10])
plt.title("Product Popularity")
plt.xlabel("Product ID")
plt.ylabel("Number of Purchases")
plt.xticks(rotation=90)
plt.show()

# User activity visualization
user_activity = df['user_id'].value_counts()
sns.histplot(user_activity, bins=5, kde=True)
plt.title("User Activity Distribution")
plt.xlabel("Number of Purchases")
plt.ylabel("Frequency")
plt.show()

# Clean the 'rating' column
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])

# Validate ratings range
if df['rating'].min() < 1 or df['rating'].max() > 5:
    raise ValueError("Ratings must be within the range of 1 to 5.")

# Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

print("Data loaded successfully into Surprise!")

3. Train-Test Split
Split the dataset into training and testing sets:

python
Copy code
trainset, testset = train_test_split(data.build_full_trainset().build_testset(), test_size=0.2, random_state=42)
4. Train the SVD Model
Fit the SVD model on the training set:

python
Copy code
svd = SVD()
svd.fit(data.build_full_trainset())
5. Evaluate Model Performance
Evaluate the model using cross-validation:

python
Copy code
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
6. Generate Recommendations
Generate and display personalized recommendations for a specific user:

python
Copy code
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

    
Conclusion
This project demonstrates how to build a personalized product recommendation system using the SVD algorithm. It explores data-driven insights and provides a scalable framework for generating user-specific product recommendations.



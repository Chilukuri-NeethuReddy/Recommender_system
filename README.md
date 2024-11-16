# *Product Recommendation System*

---

## *Overview*
This project implements a product recommendation system using the Amazon dataset. It leverages the *Singular Value Decomposition (SVD)* algorithm from the Surprise library to analyze user-item interactions and generate personalized product recommendations. The project also includes data exploration, user activity visualization, and insights into product popularity.

---

## *Dataset*
The dataset used in this project is sourced from Amazon and contains the following columns:

- *product_id*: Unique identifier for each product.  
- *product_name*: Name of the product.  
- *category*: Category of the product.  
- *discounted_price*: Price after discount.  
- *actual_price*: Original price before discount.  
- *rating*: User rating of the product.  
- *user_id*: Unique identifier for each user.  

*Source*: [Amazon Dataset](https://storage.googleapis.com/kaggle-data-sets/2818963/4862520/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241115%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241115T133910Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=30735740fb30dd28dbd8a5e7503c6e249831644a883dc2c45bef4bea627a098e313f44f832c367187200640dcce8e8aa9f5b6655e949675a6c35f4df53aa6321067b692d8b58368e9087f3b2ae8905e82967269ef151b6851f95569220de314a2faf78c9abec88790d0bbd7a89ef2e302229e23094f1cde5d9603ef2a78d407b71807432a061c64cef28f2adb3ae1a0c6764a8114b9d79d57ad2d1ac7e2a244a9b01a609e9b5856400aac588d852f9343ed086ed3817f2c04eee7383c7257d50d283db9dd3f6cbf8b7b54d1229c9d5ea0d9d2fe9bed1af79c2a55c14c7cb1ad6ee2b1a062c8e3ba798c8ffd9106308daceea05988a69a7e2f7e0ace656758ab9)

---

## *Installation*
To run this project in Google Colab, ensure you have the following libraries installed:
python
!pip install pandas numpy scikit-learn surprise matplotlib seaborn


---

## *Instructions for Running the Model*

### *1. Upload the Dataset*
Upload the dataset (amazon.csv) to Google Colab using the following code:
python
from google.colab import files

# Upload the dataset
uploaded = files.upload()


---

### *2. Load and Prepare Data*
Load the dataset into a pandas DataFrame, explore it, and clean the data:

python
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


#### *Output*:
python
# Display unique users and products
print(f"Unique users: {df['user_id'].nunique()}, Unique products: {df['product_id'].nunique()}")


#### *Product Popularity Visualization*:
python
product_popularity = df['product_id'].value_counts()
sns.barplot(x=product_popularity.index[:10], y=product_popularity.values[:10])
plt.title("Product Popularity")
plt.xlabel("Product ID")
plt.ylabel("Number of Purchases")
plt.xticks(rotation=90)
plt.show()


#### *User Activity Visualization*:
python
user_activity = df['user_id'].value_counts()
sns.histplot(user_activity, bins=5, kde=True)
plt.title("User Activity Distribution")
plt.xlabel("Number of Purchases")
plt.ylabel("Frequency")
plt.show()


#### *Clean the Ratings Column*:
python
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])

# Validate ratings range
if df['rating'].min() < 1 or df['rating'].max() > 5:
    raise ValueError("Ratings must be within the range of 1 to 5.")

# Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

print("Data loaded successfully into Surprise!")


---

### *3. Train-Test Split*
Split the dataset into training and testing sets:
python
trainset, testset = train_test_split(data.build_full_trainset().build_testset(), test_size=0.2, random_state=42)


---

### *4. Train the SVD Model*
Fit the SVD model on the training set:
python
svd = SVD()
svd.fit(data.build_full_trainset())


---

### *5. Evaluate Model Performance*
Evaluate the model using cross-validation:
python
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


---

### *6. Generate Recommendations*
Generate and display personalized recommendations for a specific user:
python
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


---

## *Conclusion*
This project demonstrates how to build a personalized product recommendation system using the SVD algorithm. It explores data-driven insights and provides a scalable framework for generating user-specific product recommendations.

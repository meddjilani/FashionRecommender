import os
import re
import pickle
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from tqdm import tqdm

# Import Keras modules for feature extraction
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# PARAMETERS
num_customers = 50              # Number of customers to simulate
max_purchases_per_customer = 5  # Maximum (average) purchase count per customer
top_k = 5                       # Top K similar items to retrieve per purchased item

# REGEX PATTERN FOR EXTRACTING INFO FROM FILENAMES
pattern = r'^(?P<gender>WOMEN|MEN)-(?P<cat>.+?)-(?P<id>id_\d{8}-\d{2}_\d+)_(?P<frame>\w+)(?:\.\w+)?$'

# ------------------------------------------------------------------------------
# STEP 1: FEATURE EXTRACTION FROM TRAIN FOLDER USING RESNET50
# ------------------------------------------------------------------------------

def extract_features(img_path, model):
    """
    Feature extractor using ResNet50.
    Loads an image, preprocesses it, and returns a flattened feature vector.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x).reshape(-1)
    return features

def extract_features_from_folder(folder_path, model, pattern):
    """
    Iterate over files in folder_path, parse filenames using the regex pattern,
    extract item metadata and features using the given model.
    
    Returns:
        features_dict: Dictionary mapping item_id to its feature vector.
        items_data: List of dictionaries with keys: item_id, category, gender.
    """
    features_dict = {}
    items_data = []
    filenames = [fname for fname in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, fname))]
    for fname in tqdm(filenames, desc="Extracting features"):
        match = re.search(pattern, fname)
        if match:
            item_id = match.group("id")
            gender = match.group("gender")
            cat = match.group("cat")
            # Save metadata
            items_data.append({
                'item_id': item_id,
                'category': cat,  # you can append "-id" if needed
                'gender': gender
            })
            full_path = os.path.join(folder_path, fname)
            # Extract features for this image
            features_dict[item_id] = extract_features(full_path, model)
    return features_dict, items_data

# DEFINE TRAIN FOLDER PATH (CHANGE AS NEEDED)
train_folder = "/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images"

# Load the ResNet50 model (without top layers) as the feature extractor
model = ResNet50(weights='imagenet', include_top=False)

# Extract features and metadata from the train folder
features_dict, items_data = extract_features_from_folder(train_folder, model, pattern)

# Create a DataFrame for item metadata (unique item_id, category, gender)
df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)
print("Extracted features for", len(df_items), "items.")

# ------------------------------------------------------------------------------
# STEP 2: SYNTHETIC PURCHASE DATA GENERATION
# ------------------------------------------------------------------------------

# Define date range for synthetic purchases (all dates in 2024)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
days_range = (end_date - start_date).days

synthetic_data = []

for user_id in range(1, num_customers + 1):
    # Random number of purchases for this user (could be 0)
    num_purchases = random.randint(0, max_purchases_per_customer)
    
    if num_purchases > 0:
        # First purchase: sample from all items to set the user's gender
        first_purchase = df_items.sample(1).iloc[0]
        first_gender = first_purchase['gender']
        
        # Filter items by the determined gender
        user_items_pool = df_items[df_items["gender"] == first_gender]
        
        for _ in range(num_purchases):
            random_item = user_items_pool.sample(1).iloc[0]
            item_id = random_item['item_id']
            category = random_item['category']
            gender = random_item['gender']
            
            # Generate a random date in 2024
            random_day_offset = random.randint(0, days_range)
            purchase_date = start_date + timedelta(days=random_day_offset)
            
            synthetic_data.append({
                'user_id': user_id,
                'item_id': item_id,
                'category': category,
                'gender': gender,
                'day': purchase_date.day,
                'month': purchase_date.month
            })

# Create DataFrame for synthetic purchase history
df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic.to_csv('data.csv', index=False)
print("Synthetic purchase history saved to 'data.csv'.")

# ------------------------------------------------------------------------------
# STEP 3: TOP-K SIMILARITY CALCULATION FOR A GIVEN USER
# ------------------------------------------------------------------------------

# Choose a user to analyze (for example, user_id = 1)
user_id_to_analyze = 1
user_purchases = df_synthetic[df_synthetic['user_id'] == user_id_to_analyze]

if user_purchases.empty:
    print(f"User {user_id_to_analyze} has no purchases.")
else:
    # Get unique item_ids purchased by this user
    user_purchased_ids = set(user_purchases['item_id'].unique())
    
    # Determine user gender from the first purchase record
    user_gender = user_purchases.iloc[0]['gender']
    
    # Build candidate list: items not purchased yet, but with the same gender.
    candidate_items = [
        item for item in features_dict.keys() 
        if (item not in user_purchased_ids) 
        and (df_items[df_items['item_id'] == item]['gender'].iloc[0] == user_gender)
    ]
    
    def euclidean_distance(vec_a, vec_b):
        return np.linalg.norm(vec_a - vec_b)
    
    # For each purchased item, compute Top-K similar items
    recommendations = {}
    
    for purchased_item in user_purchased_ids:
        purchased_feature = features_dict[purchased_item]
        distances = []
        for candidate in candidate_items:
            candidate_feature = features_dict[candidate]
            distance = euclidean_distance(purchased_feature, candidate_feature)
            distances.append((candidate, distance))
        # Sort by distance (ascending) and take top_k candidates
        distances.sort(key=lambda x: x[1])
        recommendations[purchased_item] = distances[:top_k]
    
    # Display recommendations for each purchased item
    print(f"\nTop-{top_k} recommendations for user {user_id_to_analyze}:")
    for item, recs in recommendations.items():
        print(f"\nFor purchased item {item}:")
        for candidate, dist in recs:
            print(f"   Candidate: {candidate} with distance {dist:.4f}")

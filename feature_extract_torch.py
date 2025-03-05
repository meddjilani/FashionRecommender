import os
import re
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from tqdm import tqdm
import torch.nn.functional as F

# PARAMETERS
num_customers = 50              # Number of customers to simulate
max_purchases_per_customer = 5  # Maximum (average) purchase count per customer
top_k = 5                       # Top K similar items to retrieve per purchased item

# REGEX PATTERN FOR EXTRACTING INFO FROM FILENAMES
pattern = r'^(?P<gender>WOMEN|MEN)-(?P<cat>.+?)-(?P<id>id_\d{8}-\d{2}_\d+)_(?P<frame>\w+)(?:\.\w+)?$'

# Set up device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------------------------
# STEP 1: FEATURE EXTRACTION USING RESNET50 (PyTorch version)
# ------------------------------------------------------------------------------

# Define image transforms (using ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
])

def extract_features(img_path, model, transform, device):
    """
    Loads an image from img_path, applies the given transform, and extracts features
    using the provided model. Returns a 1-D torch tensor (moved to CPU).
    """
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]
    with torch.no_grad():
        features = model(img_tensor)
    features = features.squeeze()  # Shape: [feature_dim]
    return features.cpu()  # Return on CPU for further processing

def extract_features_from_folder(folder_path, model, transform, device, pattern):
    """
    Iterates over files in folder_path, parses filenames with the regex pattern,
    extracts item metadata and computes features using the given model.
    
    Returns:
        features_dict: mapping item_id -> feature vector (torch tensor)
        items_data: list of dictionaries with keys: item_id, category, gender
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
            items_data.append({
                'item_id': item_id,
                'category': cat,  # Append "-id" if desired: cat + "-id"
                'gender': gender
            })
            full_path = os.path.join(folder_path, fname)
            if item_id not in features_dict:
                features_dict[item_id] = extract_features(full_path, model, transform, device)
    return features_dict, items_data

# Define train folder path (update to your actual folder)
train_folder = "/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images"

# Load ResNet50 model with pretrained weights and remove the final fc layer
model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()  # Replace fc layer with identity to get feature vector
model = model.to(device)
model.eval()

# Extract features and metadata from the train folder
features_dict, items_data = extract_features_from_folder(train_folder, model, transform, device, pattern)
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
    num_purchases = random.randint(0, max_purchases_per_customer)
    if num_purchases > 0:
        # First purchase: sample from all items to set the user's gender
        first_purchase = df_items.sample(1).iloc[0]
        first_gender = first_purchase['gender']
        # Filter items by the chosen gender
        user_items_pool = df_items[df_items["gender"] == first_gender]
        for _ in range(num_purchases):
            random_item = user_items_pool.sample(1).iloc[0]
            item_id = random_item['item_id']
            category = random_item['category']
            gender = random_item['gender']
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

df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic.to_csv('data.csv', index=False)
print("Synthetic purchase history saved to 'data.csv'.")

# ------------------------------------------------------------------------------
# STEP 3: TOP-K SIMILARITY CALCULATION USING COSINE SIMILARITY
# ------------------------------------------------------------------------------

# Choose a user to analyze (e.g., user_id = 1)
user_id_to_analyze = 1
user_purchases = df_synthetic[df_synthetic['user_id'] == user_id_to_analyze]

if user_purchases.empty:
    print(f"User {user_id_to_analyze} has no purchases.")
else:
    # Get unique item_ids purchased by the user
    user_purchased_ids = set(user_purchases['item_id'].unique())
    user_gender = user_purchases.iloc[0]['gender']
    
    # Build candidate list: items not yet purchased with the same gender
    candidate_items = [
        item for item in features_dict.keys() 
        if (item not in user_purchased_ids) 
        and (df_items[df_items['item_id'] == item]['gender'].iloc[0] == user_gender)
    ]
    
    recommendations = {}
    
    for purchased_item in user_purchased_ids:
        purchased_feature = features_dict[purchased_item]
        similarities = []
        for candidate in candidate_items:
            candidate_feature = features_dict[candidate]
            # Compute cosine similarity (higher is more similar)
            cos_sim = F.cosine_similarity(purchased_feature.unsqueeze(0),
                                          candidate_feature.unsqueeze(0),
                                          dim=1).item()
            similarities.append((candidate, cos_sim))
        # Sort by descending cosine similarity and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        recommendations[purchased_item] = similarities[:top_k]
    
    # Display recommendations for each purchased item
    print(f"\nTop-{top_k} recommendations for user {user_id_to_analyze}:")
    for item, recs in recommendations.items():
        print(f"\nFor purchased item {item}:")
        for candidate, sim in recs:
            print(f"   Candidate: {candidate} with cosine similarity {sim:.4f}")

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
import pickle

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
# STEP 1: FEATURE EXTRACTION USING RESNET50 (WITH CACHING)
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
                'category': cat,
                'gender': gender
            })
            full_path = os.path.join(folder_path, fname)
            if item_id not in features_dict:
                features_dict[item_id] = extract_features(full_path, model, transform, device)
    return features_dict, items_data

# Define train folder path (update to your actual folder)
train_folder = "/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images"

# Define paths to save/load features and items data
features_path = 'saved_features.pt'
items_data_path = 'saved_items_data.pkl'

# ------------------------------------------------------------------------------
# STEP 1A: LOAD FEATURES IF THEY EXIST, ELSE EXTRACT AND SAVE THEM
# ------------------------------------------------------------------------------
if os.path.exists(features_path) and os.path.exists(items_data_path):
    print("Loading pre-extracted features and items data from disk...")

    # Load feature dict (torch tensors)
    features_dict = torch.load(features_path)

    # Load items metadata
    with open(items_data_path, 'rb') as f:
        items_data = pickle.load(f)

    df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(df_items)} items from saved files.")

else:
    print("No saved data found. Extracting features...")

    # Load ResNet50 model with pretrained weights and remove the final fc layer
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # Replace fc layer with identity to get feature vector
    model = model.to(device)
    model.eval()

    # Extract features and metadata from the train folder
    features_dict, items_data = extract_features_from_folder(train_folder, model, transform, device, pattern)

    # Save extracted features and items data for future runs
    torch.save(features_dict, features_path)
    
    with open(items_data_path, 'wb') as f:
        pickle.dump(items_data, f)

    df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)
    print(f"Extracted and saved features for {len(df_items)} items.")

# ------------------------------------------------------------------------------
# STEP 2: SYNTHETIC PURCHASE DATA GENERATION WITH TRENDY PREFERENCES & POPULARITY
# ------------------------------------------------------------------------------

# Define date range for synthetic purchases (all dates in 2024)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
days_range = (end_date - start_date).days

synthetic_data = []

# 1. Pre-generate a trendy preference for each user (static during 2024)
user_trendy_preferences = {user_id: random.random() for user_id in range(1, num_customers + 1)}

# 2. Initialize item popularity dictionary (dummy initialization)
# Each item's initial popularity is set to a random integer between 0 and 5.
popularity_dict = {item_id: random.randint(0, 5) for item_id in df_items['item_id'].unique()}

# For each user, generate a series of purchases
for user_id in range(1, num_customers + 1):
    # Decide number of purchases (ensuring at least 1 purchase)
    num_purchases = random.randint(1, max_purchases_per_customer)
    trendy_pref = user_trendy_preferences[user_id]
    
    # First purchase: sample from all items to set the user's gender
    first_purchase = df_items.sample(1).iloc[0]
    first_gender = first_purchase['gender']
    random_day_offset = random.randint(0, days_range)
    purchase_date = start_date + timedelta(days=random_day_offset)
    synthetic_data.append({
        'user_id': user_id,
        'item_id': first_purchase['item_id'],
        'category': first_purchase['category'],
        'gender': first_gender,
        'day': purchase_date.day,
        'month': purchase_date.month
    })
    # Update popularity for the first purchased item.
    popularity_dict[first_purchase['item_id']] += 1

    # For subsequent purchases, use popularity and trendy preference
    if num_purchases > 1:
        # Filter candidate items by the user's gender.
        user_items_pool = df_items[df_items["gender"] == first_gender]
        for _ in range(num_purchases - 1):
            # Compute total popularity and set threshold (15% of total popularity)
            total_pop = sum(popularity_dict.values())
            threshold = 0.15 * total_pop
            
            # Split items into popular and non-popular based on the threshold.
            popular_items = user_items_pool[user_items_pool['item_id'].apply(lambda x: popularity_dict.get(x, 0) >= threshold)]
            non_popular_items = user_items_pool[user_items_pool['item_id'].apply(lambda x: popularity_dict.get(x, 0) < threshold)]
            
            # Based on the user's trendy preference, decide from which pool to pick:
            if random.random() < trendy_pref:
                # With probability equal to trendy_pref, pick from popular items if available
                if not popular_items.empty:
                    chosen_item = popular_items.sample(1).iloc[0]
                else:
                    chosen_item = non_popular_items.sample(1).iloc[0]
            else:
                # Otherwise, pick from non-popular items if available
                if not non_popular_items.empty:
                    chosen_item = non_popular_items.sample(1).iloc[0]
                else:
                    chosen_item = popular_items.sample(1).iloc[0]
            
            random_day_offset = random.randint(0, days_range)
            purchase_date = start_date + timedelta(days=random_day_offset)
            synthetic_data.append({
                'user_id': user_id,
                'item_id': chosen_item['item_id'],
                'category': chosen_item['category'],
                'gender': chosen_item['gender'],
                'day': purchase_date.day,
                'month': purchase_date.month
            })
            # Update the popularity count for the chosen item.
            popularity_dict[chosen_item['item_id']] += 1

# Save the synthetic purchase history to CSV
df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic.to_csv('data.csv', index=False)
print("Synthetic purchase history saved to 'data.csv'.")

# Optionally, extract category sets for MEN and WOMEN
categories_men = set(df_synthetic[df_synthetic['gender'] == 'MEN']['category'])
categories_women = set(df_synthetic[df_synthetic['gender'] == 'WOMEN']['category'])

# ------------------------------------------------------------------------------
# STEP 3: TOP-K SIMILARITY CALCULATION USING COSINE SIMILARITY WITH POPULARITY FUSION
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
    
    # Get the user's trendy preference from our pre-generated dictionary
    user_trend = user_trendy_preferences[user_id_to_analyze]
    
    for purchased_item in user_purchased_ids:
        purchased_feature = features_dict[purchased_item]
        candidates = []
        # Compute visual similarity and retrieve candidate popularity for each candidate
        for candidate in candidate_items:
            candidate_feature = features_dict[candidate]
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(purchased_feature.unsqueeze(0),
                                          candidate_feature.unsqueeze(0),
                                          dim=1).item()
            # Retrieve popularity for the candidate
            pop = popularity_dict.get(candidate, 0)
            candidates.append((candidate, cos_sim, pop))
        
        # Sort candidates by descending visual similarity and take top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]
        
        # Normalize popularity scores for these top_candidates
        pops = [pop for _, _, pop in top_candidates]
        min_pop, max_pop = min(pops), max(pops)
        def normalize(pop):
            if max_pop != min_pop:
                return (pop - min_pop) / (max_pop - min_pop)
            else:
                return 1.0
        
        # Fuse scores: weighted linear combination based on user's trendy preference
        fused_candidates = []
        for candidate, vis_sim, pop in top_candidates:
            norm_pop = normalize(pop)
            final_score = (1 - user_trend) * vis_sim + user_trend * norm_pop
            fused_candidates.append((candidate, final_score))
        
        # Re-rank top candidates based on the fused final score
        fused_candidates.sort(key=lambda x: x[1], reverse=True)
        recommendations[purchased_item] = fused_candidates
    
    # Display the final re-ranked recommendations for each purchased item
    print(f"\nFinal Top-{top_k} recommendations for user {user_id_to_analyze} after popularity fusion:")
    for item, recs in recommendations.items():
        print(f"\nFor purchased item {item}:")
        for candidate, final_score in recs:
            print(f"   Candidate: {candidate} with fused score {final_score:.4f}")

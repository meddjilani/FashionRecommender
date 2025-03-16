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

# List of backbones to try
backbones = ['resnet50', 'densenet121', 'vgg16']

# REGEX PATTERN FOR EXTRACTING INFO FROM FILENAMES
pattern = r'^(?P<gender>WOMEN|MEN)-(?P<cat>.+?)-(?P<id>id_\d{8}-\d{2}_\d+)_(?P<frame>\w+)(?:\.\w+)?$'

# Set up device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
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

def get_model(backbone_name, device):
    """
    Loads the specified pretrained model and modifies the final layer 
    so that it outputs a feature vector.
    """
    if backbone_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Identity()  # Replace final fc layer with identity
    elif backbone_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = torch.nn.Identity()  # Replace classifier with identity
    elif backbone_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        # Remove the last layer of the classifier to get a feature vector
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
    else:
        raise ValueError(f"Backbone {backbone_name} not supported")
    
    model = model.to(device)
    model.eval()
    return model

def extract_features(img_path, model, transform, device):
    """
    Loads an image, applies the given transform, and extracts features using the provided model.
    Returns a 1-D torch tensor (moved to CPU).
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

# Helper function: Compute distance between two geo-coordinates using the Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# ------------------------------------------------------------------------------
# CATEGORY SIMILARITY MATRIX SETUP (Manual Logic)
# ------------------------------------------------------------------------------

# List of all categories
categories = np.array(['Dresses', 'Tees_Tanks', 'Shorts', 'Jackets_Coats', 
                       'Blouses_Shirts', 'Pants', 'Sweaters', 'Skirts', 
                       'Rompers_Jumpsuits', 'Cardigans', 'Graphic_Tees', 
                       'Shirts_Polos', 'Sweatshirts_Hoodies', 'Leggings', 
                       'Denim', 'Jackets_Vests', 'Suiting'])

# Map each category to a group based on type
# Groups:
#   A: Tops, B: Bottoms, C: Outerwear, D: One-piece, E: Sweaters, F: Formal
category_to_group = {
    "Dresses": "D",
    "Tees_Tanks": "A",
    "Shorts": "B",
    "Jackets_Coats": "C",
    "Blouses_Shirts": "A",
    "Pants": "B",
    "Sweaters": "E",
    "Skirts": "D",
    "Rompers_Jumpsuits": "D",
    "Cardigans": "E",
    "Graphic_Tees": "A",
    "Shirts_Polos": "A",
    "Sweatshirts_Hoodies": "E",
    "Leggings": "B",
    "Denim": "B",
    "Jackets_Vests": "C",
    "Suiting": "F"
}

# Define inter-group similarities (keys are tuples of groups, sorted alphabetically)
group_similarity = {
    ("A", "B"): 0.2,
    ("A", "C"): 0.3,
    ("A", "D"): 0.1,
    ("A", "E"): 0.4,
    ("A", "F"): 0.2,
    ("B", "C"): 0.1,
    ("B", "D"): 0.1,
    ("B", "E"): 0.2,
    ("B", "F"): 0.2,
    ("C", "D"): 0.1,
    ("C", "E"): 0.4,
    ("C", "F"): 0.2,
    ("D", "E"): 0.1,
    ("D", "F"): 0.3,
    ("E", "F"): 0.2,
}

def category_similarity(cat_purchased, cat_candidate):
    """
    Compute similarity between two categories using our manual logic.
    If the categories are in the same group:
        - Return 1.0 if they are identical,
        - Otherwise return 0.8.
    If they are in different groups, return the predefined inter-group similarity.
    """
    group1 = category_to_group.get(cat_purchased)
    group2 = category_to_group.get(cat_candidate)
    if group1 is None or group2 is None:
        return 0.0
    if group1 == group2:
        return 1.0 if cat_purchased == cat_candidate else 0.8
    # Order the groups alphabetically for lookup
    key = tuple(sorted((group1, group2)))
    return group_similarity.get(key, 0.0)

# ------------------------------------------------------------------------------
# PATHS & DATASET DIRECTORY
# ------------------------------------------------------------------------------

# Define train folder path (update to your actual folder)
train_folder = "/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images"

# Define path to save/load items metadata (common to all backbones)
items_data_path = 'saved_items_data.pkl'

# ------------------------------------------------------------------------------
# STEP 1: LOAD OR EXTRACT ITEMS METADATA
# ------------------------------------------------------------------------------

if os.path.exists(items_data_path):
    print("Loading items metadata from disk...")
    with open(items_data_path, 'rb') as f:
        items_data = pickle.load(f)
    df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(df_items)} items from saved file.")
else:
    # If items metadata doesn't exist, use the first backbone to extract it.
    print("No saved items metadata found. Extracting using first backbone:", backbones[0])
    model_temp = get_model(backbones[0], device)
    features_dict_temp, items_data = extract_features_from_folder(train_folder, model_temp, transform, device, pattern)
    # Save features for the first backbone for future runs
    torch.save(features_dict_temp, f'saved_features_{backbones[0]}.pt')
    with open(items_data_path, 'wb') as f:
        pickle.dump(items_data, f)
    df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)
    print(f"Extracted and saved items metadata for {len(df_items)} items.")

# ------------------------------------------------------------------------------
# STEP 1A: FEATURE EXTRACTION WITH DIFFERENT BACKBONES (WITH CACHING)
# ------------------------------------------------------------------------------

all_features = {}

for backbone in backbones:
    features_path = f'saved_features_{backbone}.pt'
    if os.path.exists(features_path):
        print(f"Loading pre-extracted features for {backbone} from disk...")
        features_dict = torch.load(features_path)
    else:
        print(f"No saved features found for {backbone}. Extracting features...")
        model = get_model(backbone, device)
        features_dict, _ = extract_features_from_folder(train_folder, model, transform, device, pattern)
        torch.save(features_dict, features_path)
    all_features[backbone] = features_dict

# ------------------------------------------------------------------------------
# NEW IDEA: STORE ASSIGNMENT & USER LOCATION
# ------------------------------------------------------------------------------

# Define number of stores (N)
num_stores = 5  # You can change this value as needed

# Create a dictionary of stores with random coordinates (longitude, latitude)
stores = {}
for store_id in range(1, num_stores + 1):
    stores[store_id] = {
        'longitude': random.uniform(-1.80, 1.80),
        'latitude': random.uniform(-0.90, 0.90)
    }
print("Store locations:", stores)

# Distribute items among stores for each category:
# For each category, shuffle the items and assign them to a store in a round-robin fashion.
df_items['store_id'] = None
for cat, group in df_items.groupby('category'):
    indices = group.index.tolist()
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        store_assignment = (i % num_stores) + 1  # Store ids from 1 to num_stores
        df_items.at[idx, 'store_id'] = store_assignment

# Generate a random location for the user (or use provided values)
user_longitude = random.uniform(-1.8, 1.8)
user_latitude = random.uniform(-0.90, 0.90)
print("User location:", {"longitude": user_longitude, "latitude": user_latitude})

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
# STEP 3: TOP-K SIMILARITY CALCULATION USING THE NEW "RELEVANCE" SCORE
# ------------------------------------------------------------------------------

# Choose a user to analyze (e.g., user_id = 1)
user_id_to_analyze = 1
user_purchases = df_synthetic[df_synthetic['user_id'] == user_id_to_analyze]

if user_purchases.empty:
    print(f"User {user_id_to_analyze} has no purchases.")
else:
    # Get unique item_ids purchased by the user and retrieve user gender (from the first purchase)
    user_purchased_ids = set(user_purchases['item_id'].unique())
    user_gender = user_purchases.iloc[0]['gender']
    
    # For each backbone, generate recommendations based on its extracted features
    for backbone, features_dict in all_features.items():
        print(f"\n--- Recommendations using backbone: {backbone} ---")
        
        # Build candidate list: items not yet purchased with the same gender
        candidate_items = [
            item for item in features_dict.keys() 
            if (item not in user_purchased_ids) 
            and (df_items[df_items['item_id'] == item]['gender'].iloc[0] == user_gender)
        ]
        
        recommendations = {}
        
        # Define weights for each component of the Relevance score
        w_v = 0.4  # visual similarity weight
        w_p = 0.2  # popularity weight
        w_g = 0.2  # gender match weight
        w_c = 0.2  # category similarity weight
        
        # For each purchased item, compute the new fused "Relevance" score for all candidates
        for purchased_item in user_purchased_ids:
            purchased_feature = features_dict[purchased_item]
            # Get purchased item's metadata
            purchased_metadata = df_items[df_items['item_id'] == purchased_item].iloc[0]
            purchased_gender = purchased_metadata['gender']
            purchased_category = purchased_metadata['category']
            
            candidates = []
            # Compute raw scores for each candidate item
            for candidate in candidate_items:
                candidate_feature = features_dict[candidate]
                vis_sim = F.cosine_similarity(purchased_feature.unsqueeze(0),
                                              candidate_feature.unsqueeze(0),
                                              dim=1).item()
                pop = popularity_dict.get(candidate, 0)
                candidate_metadata = df_items[df_items['item_id'] == candidate].iloc[0]
                candidate_gender = candidate_metadata['gender']
                candidate_category = candidate_metadata['category']
                
                # Append tuple: (candidate, visual similarity, popularity, candidate gender, candidate category)
                candidates.append((candidate, vis_sim, pop, candidate_gender, candidate_category))
            
            # Normalize visual similarity and popularity over the candidate set using min-max normalization
            def min_max_normalize(values):
                min_val, max_val = min(values), max(values)
                if max_val == min_val:
                    return [1.0 for _ in values]
                return [(v - min_val) / (max_val - min_val) for v in values]
            
            vis_sims = [cand[1] for cand in candidates]
            pops = [cand[2] for cand in candidates]
            
            norm_vis_sims = min_max_normalize(vis_sims)
            norm_pops = min_max_normalize(pops)
            
            fused_candidates = []
            for idx, candidate_info in enumerate(candidates):
                candidate_id, _, _, candidate_gender, candidate_category = candidate_info
                # Gender match score: 1 if matching, else 0
                gender_score = 1.0 if candidate_gender == purchased_gender else 0.0
                # Category similarity score from the manual similarity matrix
                cat_score = category_similarity(purchased_category, candidate_category)
                # Compute final Relevance score as weighted sum of normalized scores and binary scores
                relevance = (w_v * norm_vis_sims[idx] +
                             w_p * norm_pops[idx] +
                             w_g * gender_score +
                             w_c * cat_score)
                fused_candidates.append((candidate_id, relevance))
            
            # Sort candidates by descending Relevance score and select top_k
            fused_candidates.sort(key=lambda x: x[1], reverse=True)
            recommendations[purchased_item] = fused_candidates[:top_k]
        
        # Display the final recommendations along with store details and distance from the user.
        print(f"\nFinal Top-{top_k} recommendations for user {user_id_to_analyze} using backbone {backbone}:")
        for item, recs in recommendations.items():
            print(f"\nFor purchased item {item}:")
            for candidate, relevance in recs:
                store_id = df_items[df_items['item_id'] == candidate]['store_id'].iloc[0]
                store_info = stores[store_id]
                distance = haversine(user_longitude, user_latitude, store_info['longitude'], store_info['latitude'])
                print(f"   Candidate: {candidate} | Relevance: {relevance:.4f} | Store: {store_id} at {store_info} | Distance: {distance:.2f} km")

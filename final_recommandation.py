import logging
import os
import re
import random
import pickle
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import argparse

# ------------------------------------------------------------------------------
# CONFIGURATION & SETUP
# ------------------------------------------------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example color codes for the non-garments in the segmentation mask
NON_GARMENT_COLORS = [
    (255, 0, 0),    # Red (Hair)
    (16, 78, 139),  # Blue (Face)
    (144, 238, 144), # Skin (Green)
    (0, 0, 0)       # Background (Black)
]


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Fashion Recommender")

    # Add arguments for configurable parameters
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--num_customers', type=int, default=15, help="Number of customers to simulate")
    parser.add_argument('--max_purchases_per_customer', type=int, default=3, help="Maximum (average) purchase count per customer")
    parser.add_argument('--top_k', type=int, default=10, help="Top K similar items to retrieve per purchased item")
    parser.add_argument('--P_max', type=int, default=100, help="Maximum number of purchases for a given item at initialization")
    parser.add_argument('--P_threshold', type=int, default=0.15, help="threshold to divide popular items from non popular items")
    parser.add_argument('--Penalty_exponent', type=int, default=4, help="controlling the sensitivity to differences between usert trendy preference and item popularity")



    parser.add_argument('--train_folder', type=str, default="/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images", help="Path to training images folder")
    parser.add_argument('--seg_folder', type=str, default="/mnt/isilon/maliousalah/FashionRecommender/data/datasets/segm", help="Path to segmentation images folder")

    return parser.parse_args()

# Parse arguments
args = parse_args()


# Set random seeds for reproducibility
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# PARAMETERS
NUM_CUSTOMERS = args.num_customers             # Number of customers to simulate
MAX_PURCHASES_PER_CUSTOMER = args.max_purchases_per_customer  # Maximum (average) purchase count per customer
TOP_K = args.top_k                            # Top K similar items to retrieve per purchased item
P_max = args.P_max # popularity range
P_threshold = args.P_threshold # popularity threshold
Penalty_exponent = args.Penalty_exponent # popularity penality exponent

# List of backbones to try
BACKBONES = ['resnet50', 'densenet121', 'vgg16']

# REGEX PATTERN FOR EXTRACTING INFO FROM FILENAMES
PATTERN = r'^(?P<gender>WOMEN|MEN)-(?P<cat>.+?)-(?P<id>id_\d{8}-\d{2}_\d+)_(?P<frame>\w+)(?:\.\w+)?$'

# Set up device: use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)

# Data paths
TRAIN_FOLDER = pathlib.Path(args.train_folder)
SEG_FOLDER   = pathlib.Path(args.seg_folder)
ITEMS_DATA_PATH = pathlib.Path("utils/saved_items_data.pkl")

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

def get_model(backbone_name: str, device: torch.device) -> torch.nn.Module:
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

def load_image(img_path: pathlib.Path) -> Image.Image:
    """
    Loads an image from a given path and converts it to RGB.
    """
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        logger.error("Error loading image %s: %s", img_path, e)
        raise

def extract_features(img_path: pathlib.Path, seg_path: pathlib.Path, model: torch.nn.Module, transform, device: torch.device) -> torch.Tensor:
    """
    Extracts features from an image after masking out non-garment regions using color-coded segmentation.
    """
    img = load_image(img_path)
    seg_img = load_image(str(seg_path))
    # Apply color-based masking
    masked_img = apply_color_segmentation(img, seg_img)

    # Transform and extract features
    img_tensor = transform(masked_img).unsqueeze(0).to(device)

    #img_tensor = transform(img).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]
    with torch.no_grad():
        features = model(img_tensor)
    features = features.squeeze()  # Shape: [feature_dim]
    return features.cpu()

def extract_features_from_folder(folder_path: pathlib.Path, seg_path: pathlib.Path, model: torch.nn.Module, transform, 
                                 device: torch.device, pattern: str):
    """
    Iterates over files in folder_path, parses filenames with the regex pattern,
    extracts item metadata and computes features using the given model and its 
    corresponding segmentation mask.
    
    Returns:
        features_dict: mapping item_id -> feature vector (torch tensor)
        items_data: list of dictionaries with keys: item_id, category, gender
    """
    features_dict = {}
    items_data = []
    filenames = [fname for fname in os.listdir(folder_path) if (folder_path / fname).is_file()]
    
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
            
            full_path = folder_path / fname
            
            if item_id not in features_dict:
                # Construct the segmentation file name.
                # E.g., from "MEN-Denim-id_00000080-01_7_additional.png" 
                # to "MEN-Denim-id_00000080-01_7_additional_segm.png".
                segm_fname = f"{fname.split('.')[0]}_segm.png"
                segm_path = seg_path / segm_fname
                
                # Extract features using the color mask based on segmentation.
                features_dict[item_id] = extract_features(
                    full_path, segm_path, model, transform, device
                )
                
    return features_dict, items_data

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes the Haversine distance between two geo-coordinates.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r

def apply_color_segmentation(original_img: Image.Image, seg_mask: Image.Image) -> Image.Image:
    """
    Keeps only pixels from the original image whose segmentation mask color 
    is not in the known non-garment color list. Everything else is set to black.
    
    Args:
        original_img (PIL.Image): original RGB image.
        seg_mask (PIL.Image): segmentation mask in RGB where specific colors 
                              indicate different body/clothing parts.
    
    Returns:
        PIL.Image: masked RGB image where only garment pixels are kept.
    """
    # Convert to NumPy arrays
    img_np = np.array(original_img)         # shape: (H, W, 3)
    seg_np = np.array(seg_mask)             # shape: (H, W, 3)

    # Start with a mask of all True (everything is kept)
    garment_mask = np.ones((seg_np.shape[0], seg_np.shape[1]), dtype=bool)

    # Remove (set False) where the color matches any in NON_GARMENT_COLORS
    for color in NON_GARMENT_COLORS:
        color_mask = np.all(seg_np == color, axis=-1)  # shape: (H, W)
        garment_mask &= ~color_mask

    # Expand to 3D so it can mask RGB image
    garment_mask_3d = np.stack([garment_mask]*3, axis=-1)

    # Apply the mask
    masked_img_np = np.where(garment_mask_3d, img_np, 0)

    # Convert back to PIL Image
    masked_img = Image.fromarray(masked_img_np.astype(np.uint8), mode="RGB")
    return masked_img

# ------------------------------------------------------------------------------
# CATEGORY SIMILARITY MATRIX SETUP (Manual Logic)
# ------------------------------------------------------------------------------

categories = np.array(['Dresses', 'Tees_Tanks', 'Shorts', 'Jackets_Coats', 
                       'Blouses_Shirts', 'Pants', 'Sweaters', 'Skirts', 
                       'Rompers_Jumpsuits', 'Cardigans', 'Graphic_Tees', 
                       'Shirts_Polos', 'Sweatshirts_Hoodies', 'Leggings', 
                       'Denim', 'Jackets_Vests', 'Suiting'])

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

def category_similarity(cat_purchased: str, cat_candidate: str) -> float:
    """
    Computes similarity between two categories using manual logic.
    Returns:
        1.0 if identical in the same group,
        0.8 if in the same group but different,
        or the predefined inter-group similarity otherwise.
    """
    group1 = category_to_group.get(cat_purchased)
    group2 = category_to_group.get(cat_candidate)
    if group1 is None or group2 is None:
        return 0.0
    if group1 == group2:
        return 1.0 if cat_purchased == cat_candidate else 0.8
    key = tuple(sorted((group1, group2)))
    return group_similarity.get(key, 0.0)

# ------------------------------------------------------------------------------
# DATA HANDLING & SIMULATION FUNCTIONS
# ------------------------------------------------------------------------------

def load_or_extract_items_data() -> pd.DataFrame:
    """
    Loads item metadata from disk if available, otherwise extracts it using the first backbone.
    """
    if ITEMS_DATA_PATH.exists():
        logger.info("Loading items metadata from disk...")
        with open(ITEMS_DATA_PATH, 'rb') as f:
            items_data = pickle.load(f)
        df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)
        logger.info("Loaded %d items from saved file.", len(df_items))
    else:
        logger.info("No saved items metadata found. Extracting using backbone: %s", BACKBONES[0])
        model_temp = get_model(BACKBONES[0], DEVICE)
        features_dict_temp, items_data = extract_features_from_folder(TRAIN_FOLDER, SEG_FOLDER, model_temp, transform, DEVICE, PATTERN)
        torch.save(features_dict_temp, f'utils/saved_features_{BACKBONES[0]}.pt')
        with open(ITEMS_DATA_PATH, 'wb') as f:
            pickle.dump(items_data, f)
        df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)
        logger.info("Extracted and saved items metadata for %d items.", len(df_items))
    return df_items

def assign_stores_to_items(df_items: pd.DataFrame, num_stores: int = 5) -> (pd.DataFrame, dict):
    """
    Assigns each item to a store (round-robin per category) and returns store locations.
    """
    stores = {}
    for store_id in range(1, num_stores + 1):
        stores[store_id] = {
            'longitude': random.uniform(-1.80, 1.80),
            'latitude': random.uniform(-0.90, 0.90)
        }
    logger.info("Store locations: %s", stores)
    
    df_items['store_id'] = None
    for cat, group in df_items.groupby('category'):
        indices = group.index.tolist()
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            store_assignment = (i % num_stores) + 1
            df_items.at[idx, 'store_id'] = store_assignment
    return df_items, stores

def generate_synthetic_purchase_data(df_items: pd.DataFrame, num_customers: int = NUM_CUSTOMERS,
                                     max_purchases: int = MAX_PURCHASES_PER_CUSTOMER) -> (pd.DataFrame, dict):
    """
    Generates synthetic purchase data with trendy preferences and dynamic item popularity.
    """
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    days_range = (end_date - start_date).days

    synthetic_data = []
    user_trendy_preferences = {user_id: random.random() for user_id in range(1, num_customers + 1)}
    popularity_dict = {item_id: random.randint(0, P_max) for item_id in df_items['item_id'].unique()}

    for user_id in range(1, num_customers + 1):
        num_purchases = random.randint(1, max_purchases)
        trendy_pref = user_trendy_preferences[user_id]
        
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
        popularity_dict[first_purchase['item_id']] += 1

        if num_purchases > 1:
            user_items_pool = df_items[df_items["gender"] == first_gender]
            for _ in range(num_purchases - 1):
                total_pop = sum(popularity_dict.values())
                threshold = P_threshold * total_pop
                
                popular_items = user_items_pool[user_items_pool['item_id'].apply(lambda x: popularity_dict.get(x, 0) >= threshold)]
                non_popular_items = user_items_pool[user_items_pool['item_id'].apply(lambda x: popularity_dict.get(x, 0) < threshold)]
                
                if random.random() < trendy_pref:
                    chosen_item = popular_items.sample(1).iloc[0] if not popular_items.empty else non_popular_items.sample(1).iloc[0]
                else:
                    chosen_item = non_popular_items.sample(1).iloc[0] if not non_popular_items.empty else popular_items.sample(1).iloc[0]
                
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
                popularity_dict[chosen_item['item_id']] += 1

    df_synthetic = pd.DataFrame(synthetic_data)
    df_synthetic.to_csv('utils/data.csv', index=False)
    logger.info("Synthetic purchase history saved to 'data.csv'.")
    with open('utils/popularity_dict.pkl', 'wb') as f:
        pickle.dump(popularity_dict, f)
    logger.info("Popularity dictionary  saved to 'popularity_dict.pkl'.")
    with open('utils/user_trendy_preferences.pkl', 'wb') as f:
        pickle.dump(user_trendy_preferences, f)
    logger.info("User trendy preferences saved to 'user_trendy_preferences.pkl'.")
    return df_synthetic, popularity_dict, user_trendy_preferences

def load_image(img_path: str) -> Image.Image:
        return Image.open(img_path).convert("RGB")

def demo_masking():
    original_path = "/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images/MEN-Denim-id_00000080-01_7_additional.png"
    seg_path = "/mnt/isilon/maliousalah/FashionRecommender/data/datasets/segm/MEN-Denim-id_00000080-01_7_additional_segm.png"

    original_img = load_image(original_path)
    seg_img = load_image(seg_path)  # load the segmentation mask in RGB

    # Apply the color-based segmentation to keep only garment pixels
    masked_img = apply_color_segmentation(original_img, seg_img)
    
    # Save the masked image to file for debugging/verification
    save_path = "MEN-Denim-id_00000080-01_7_additional_masked.png"
    masked_img.save(save_path)
    print(f"Masked image saved to {save_path}")

def min_max_normalize_key(d, key):
    values = list(d.values())
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return 0.0
    return (d[key] - min_val) / (max_val - min_val)

def compute_recommendations(user_id: int, df_synthetic: pd.DataFrame, df_items: pd.DataFrame,
                            all_features: dict, popularity_dict: dict, user_trendy_preferences: dict, stores: dict,
                            user_long: float, user_lat: float, top_k: int = TOP_K) -> None:
    """
    Computes and logs recommendations for a given user using different backbones.
    """
    user_purchases = df_synthetic[df_synthetic['user_id'] == user_id]
    if user_purchases.empty:
        logger.info("User %d has no purchases.", user_id)
        return
    
    # Drop duplicate item_id entries to ensure uniqueness in metadata_map
    metadata_map = df_items.drop_duplicates(subset=['item_id']).set_index('item_id').to_dict('index')
    
    user_purchased_ids = set(user_purchases['item_id'].unique())
    user_gender = metadata_map[next(iter(user_purchased_ids))]['gender']

    with open('utils/user_trendy_preferences.pkl', 'rb') as f:
        user_trendy_preferences = pickle.load(f)
    with open('utils/popularity_dict.pkl', 'rb') as f:
            popularity_dict = pickle.load(f)
    # Define weights for relevance score
    w_v = 0.4  # visual similarity
    w_p = 0.2  # popularity
#    w_g = 0.2  # gender match
#    w_c = 0.2  # category similarity
    results_list_per_customer = []
    for backbone, features_dict in all_features.items():
        logger.info("Recommendations using backbone: %s", backbone)
        candidate_items = [
            item for item in features_dict.keys() 
            if item not in user_purchased_ids #and metadata_map[item]['gender'] == user_gender
        ]
        recommendations = {}
        
        for purchased_item in user_purchased_ids:
            purchased_feature = features_dict[purchased_item]
            purchased_metadata = metadata_map[purchased_item]
            purchased_gender = purchased_metadata['gender']
            purchased_category = purchased_metadata['category']
            
            candidates = []
            for candidate in candidate_items:
                candidate_feature = features_dict[candidate]
                vis_sim = F.cosine_similarity(purchased_feature.unsqueeze(0),
                                              candidate_feature.unsqueeze(0), dim=1).item()
                pop = popularity_dict.get(candidate, 0)
                candidate_metadata = metadata_map[candidate]
                candidate_gender = candidate_metadata['gender']
                candidate_category = candidate_metadata['category']
                candidates.append((candidate, vis_sim, pop, candidate_gender, candidate_category))
            
            def min_max_normalize(values):
                min_val, max_val = min(values), max(values)
                if max_val == min_val:
                    return [1.0 for _ in values]
                return [(v - min_val) / (max_val - min_val) for v in values]


            # could be change to norm over the entire set and not only candidate
            vis_sims = [cand[1] for cand in candidates]
            pops = [cand[2] for cand in candidates]
            norm_vis_sims = min_max_normalize(vis_sims)
            norm_pops = min_max_normalize(pops)
            
            
            fused_candidates = []
            for idx, candidate_info in enumerate(candidates):
                candidate_id, _, _, candidate_gender, candidate_category = candidate_info
                gender_score = 1.0 if candidate_gender == purchased_gender else 0.0
                cat_score = category_similarity(purchased_category, candidate_category)
                relevance = (w_v * norm_vis_sims[idx] +
                             w_p * (1 - (norm_pops[idx] - user_trendy_preferences[user_id])** Penalty_exponent)
                             ) #Visual similarity attend to gender match and category similarity
#                             w_g * gender_score +
#                             w_c * cat_score)
                fused_candidates.append((candidate_id, relevance))
            
            fused_candidates.sort(key=lambda x: x[1], reverse=True)
            recommendations[purchased_item] = fused_candidates[:top_k]
        
        logger.info("Final Top-%d recommendations for user %d using backbone %s:", top_k, user_id, backbone)
        for item, recs in recommendations.items():
            logger.info("For purchased item %s:", item)
            for candidate, relevance in recs:
                store_id = metadata_map[candidate]['store_id']
                store_info = stores[store_id]
                distance = haversine(user_long, user_lat, store_info['longitude'], store_info['latitude'])
                logger.info("   Candidate: %s | Relevance: %.4f | Store: %d at %s | Distance: %.2f km",
                            candidate, relevance, store_id, store_info, distance)
                row = {'user_id':user_id, 'backbone':backbone, 'purchased_item':str(item)+" "+metadata_map[item]['category'], 'recommended_item': str(candidate)+" "+(metadata_map[candidate]['category']), 'relevance': relevance, 'category_similarity':category_similarity(metadata_map[item]['category'], metadata_map[candidate]['category']),'user_trendy_preferences': user_trendy_preferences[user_id], 'recommended_item_popularity': min_max_normalize_key(popularity_dict,candidate), 'user_gender':user_gender, 'recommended_item_gender': metadata_map[item]['gender']}
                results_list_per_customer.append(row)


    return results_list_per_customer

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Load or extract item metadata
    df_items = load_or_extract_items_data()
    df_items, stores = assign_stores_to_items(df_items)
    
    # Generate a random user location
    user_longitude = random.uniform(-1.8, 1.8)
    user_latitude = random.uniform(-0.90, 0.90)
    logger.info("User location: {'longitude': %.2f, 'latitude': %.2f}", user_longitude, user_latitude)
    
    # Generate synthetic purchase data
    df_synthetic, popularity_dict, user_trendy_preferences = generate_synthetic_purchase_data(df_items)
    
    # Extract features with caching for each backbone
    all_features = {}
    for backbone in BACKBONES:
        features_path = pathlib.Path(f'utils/saved_features_{backbone}.pt')
        if features_path.exists():
            logger.info("Loading pre-extracted features for %s from disk...", backbone)
            features_dict = torch.load(features_path)
        else:
            logger.info("No saved features found for %s. Extracting features...", backbone)
            model = get_model(backbone, DEVICE)
            features_dict, _ = extract_features_from_folder(TRAIN_FOLDER, SEG_FOLDER, model, transform, DEVICE, PATTERN)
            torch.save(features_dict, features_path)
        all_features[backbone] = features_dict

    results_list = []
    for customer in range(1, NUM_CUSTOMERS + 1):
        results_list_per_customer = compute_recommendations(customer, df_synthetic, df_items, all_features, popularity_dict, user_trendy_preferences, stores,
                            user_longitude, user_latitude)
        results_list.extend(results_list_per_customer)
        
    results_df = pd.DataFrame(results_list)

    # Construct filename using args
    filename = (
        f"df{args.seed}_cust{args.num_customers}_pur{args.max_purchases_per_customer}_"
        f"Pthresh{args.P_threshold}_PenExp{args.Penalty_exponent}_Pmax{args.P_max}_topk{args.top_k}.csv"
    )

    os.makedirs('results', exist_ok=True)

    # Save the DataFrame to a CSV file
    results_df.to_csv(os.path.join('results', filename), index=True)

    #demo_masking()

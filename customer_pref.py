import re
import os 
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# PARAMETERS
num_customers = 50              # number of customers to simulate
max_purchases_per_customer = 5  # maximum (average) purchase count per customer



# Regex pattern explanation:
#   ^WOMEN-              => matches the literal prefix "WOMEN-"
#   (?P<cat>.+?)-         => non-greedy capture of the category part before "-id"
#   (?P<id>id_\d{8}-\d{2}_\d+)  => capture the id part which follows the pattern: "id_" + 8 digits + "-" + 2 digits + "_" + one or more digits
#   _(?P<frame>\w+)$      => capture the frame (a word) after the underscore at the end of the string
pattern = r'^(?P<gender>WOMEN|MEN)-(?P<cat>.+?)-(?P<id>id_\d{8}-\d{2}_\d+)_(?P<frame>\w+)(?:\.\w+)?$'


def get_filenames_from_folder(folder_path):
    # List all files in the given folder (ignoring subdirectories)
    filenames = [filename for filename in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, filename))]
    return filenames

# Example usage:
folder_path = "/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images"  # Change this to your folder path
filenames = get_filenames_from_folder(folder_path)

for fname in filenames:
    match = re.search(pattern, fname)
    if match:
        # Build the category column by appending "-id" to the captured category part.
        category = match.group("cat") + "-id"
        item_id = match.group("id")
        frame = match.group("frame")
        print(f"Filename: {fname}")
        print("  Category:", category)
        print("  ID      :", item_id)
        print("  Frame   :", frame)
    else:
        print("No match for filename:", fname)



# LOAD THE DEEPFASHION TRAIN DATASET
# Assumes a CSV file with at least 'item_id' and 'category' columns
# FILE EXTRACTION FUNCTION
def get_filenames_from_folder(folder_path):
    """Retrieve all filenames from the specified folder."""
    return [
        filename for filename in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, filename))
    ]

# REGEX PATTERN FOR EXTRACTING INFO FROM FILENAMES
pattern = r'^(?P<gender>WOMEN|MEN)-(?P<cat>.+?)-(?P<id>id_\d{8}-\d{2}_\d+)_(?P<frame>\w+)(?:\.\w+)?$'

# DEFINE TRAIN FOLDER PATH (CHANGE AS NEEDED)
train_folder = "/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images"

# GET ALL FILENAMES FROM THE TRAIN SET
filenames = get_filenames_from_folder(train_folder)

# PARSE FILENAMES TO EXTRACT ITEM DETAILS
items_data = []
for fname in filenames:
    match = re.search(pattern, fname)
    if match:
        items_data.append({
            'item_id': match.group("id"),
            'category': match.group("cat"), #+ "-id",  # Append '-id' to category name
            'gender': match.group("gender")  # Extract gender (MEN/WOMEN)
        })

# CREATE A DATAFRAME OF UNIQUE ITEM-CATEGORY-GENDER PAIRS
df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)

# PREPARE SYNTHETIC DATA STORAGE
synthetic_data = []

# DEFINE DATE RANGE FOR SYNTHETIC PURCHASES (ALL DATES IN 2024)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
days_range = (end_date - start_date).days

# GENERATE SYNTHETIC PURCHASES FOR EACH CUSTOMER
for user_id in range(1, num_customers + 1):
    # Random number of purchases for this customer (could be 0)
    num_purchases = random.randint(0, max_purchases_per_customer)

    for _ in range(num_purchases):
        # Randomly select an item from the df_items DataFrame
        random_item = df_items.sample(1).iloc[0]
        item_id = random_item['item_id']
        category = random_item['category']
        gender = random_item['gender']

        # Generate a random date between start_date and end_date
        random_day_offset = random.randint(0, days_range)
        purchase_date = start_date + timedelta(days=random_day_offset)

        # Extract day and month (year is implicitly 2024)
        day = purchase_date.day
        month = purchase_date.month

        # Append the purchase record
        synthetic_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'category': category,
            'gender': gender,  # Include gender in the dataset
            'day': day,
            'month': month
        })

# CREATE A DATAFRAME FROM THE SYNTHETIC DATA
df_synthetic = pd.DataFrame(synthetic_data)
print(df_synthetic.head())
# DISPLAY THE FIRST FEW ROWS OF THE SYNTHETIC DATASET
#import ace_tools as tools
#tools.display_dataframe_to_user(name="Synthetic Fashion Dataset", dataframe=df_synthetic)

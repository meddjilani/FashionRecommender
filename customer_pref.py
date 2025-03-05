import os
import re
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# PARAMETERS
num_customers = 50              # Number of customers to simulate
max_purchases_per_customer = 5  # Maximum (average) purchase count per customer

# REGEX PATTERN FOR EXTRACTING INFO FROM FILENAMES
pattern = r'^(?P<gender>WOMEN|MEN)-(?P<cat>.+?)-(?P<id>id_\d{8}-\d{2}_\d+)_(?P<frame>\w+)(?:\.\w+)?$'

# FUNCTION TO LIST FILES IN THE TRAINING FOLDER
def get_filenames_from_folder(folder_path):
    """Retrieve all filenames from the specified folder."""
    return [
        filename for filename in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, filename))
    ]

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
            'category': match.group("cat"),  # Category without "-id"
            'gender': match.group("gender")  # Extract gender (MEN/WOMEN)
        })

# CREATE A DATAFRAME OF UNIQUE ITEM-CATEGORY-GENDER PAIRS
df_items = pd.DataFrame(items_data).drop_duplicates().reset_index(drop=True)

# SEPARATE DATASETS FOR MEN AND WOMEN
df_men_items = df_items[df_items["gender"] == "MEN"].reset_index(drop=True)
df_women_items = df_items[df_items["gender"] == "WOMEN"].reset_index(drop=True)

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

    if num_purchases > 0:
        # Select first purchase randomly from full dataset
        first_purchase = df_items.sample(1).iloc[0]
        first_gender = first_purchase['gender']  # Determine gender from first purchase

        # Use the respective gender dataset for the rest of the purchases
        if first_gender == "MEN":
            customer_items = df_men_items
        else:
            customer_items = df_women_items

        for _ in range(num_purchases):
            # Randomly select an item from the filtered dataset
            random_item = customer_items.sample(1).iloc[0]
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
                'gender': gender,  # Ensure gender consistency
                'day': day,
                'month': month
            })

# CREATE A DATAFRAME FROM THE SYNTHETIC DATA
df_synthetic = pd.DataFrame(synthetic_data)

# SAVE DATA TO CSV
df_synthetic.to_csv('data.csv', index=False)

print(df_synthetic.head(30))
# DISPLAY THE FIRST FEW ROWS OF THE SYNTHETIC DATASET
#import ace_tools as tools
#tools.display_dataframe_to_user(name="Synthetic Fashion Dataset", dataframe=df_synthetic)

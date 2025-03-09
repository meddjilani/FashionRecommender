import os
import pandas as pd

# Path to the folder where your files are stored
folder_path = '/mnt/isilon/maliousalah/FashionRecommender/data/datasets/train_images'

# List all files in the folder
files = os.listdir(folder_path)

# Extract categories from filenames
categories = []
for file in files:
    if file.endswith('.png'):  # Only consider .png files
        parts = file.split('-')
        if len(parts) > 1:
            categories.append(parts[1])

# Convert to pandas Series to use .unique()
categories_series = pd.Series(categories)
unique_categories = categories_series.unique()

print("Unique categories:", len(unique_categories))
# Fashion Recommender System

## Overview
This project implements a synthetic simulation of a fashion recommendation system. It leverages pretrained convolutional neural networks (CNNs) from PyTorch’s torchvision (including ResNet50, DenseNet121, and VGG16) to extract feature vectors from fashion item images. By parsing image filenames with a regular expression, the system extracts metadata (such as gender, category, and item ID) and builds a dataset of items. In addition, synthetic purchase data is generated to simulate customer behavior, which is then used to compute recommendations based on a fused relevance score.

## Key Features
- **Image Feature Extraction:**  
  Uses pretrained models (ResNet50, DenseNet121, VGG16) to extract image features. The final classification layers are replaced with identity mappings (or trimmed) to output feature vectors.
  
- **Metadata Extraction:**  
  Extracts metadata (gender, category, item ID, etc.) from image filenames using a regular expression.

- **Synthetic Purchase Data Generation:**  
  Simulates purchase histories for a configurable number of customers. Purchase events consider trendy preferences, item popularity, and category preferences over a defined date range.

- **Recommender System:**  
  For each purchased item, the system computes a fused "Relevance" score for candidate items. The score is a weighted combination of:
  - Visual similarity (cosine similarity between feature vectors)
  - Item popularity (updated with each synthetic purchase)
  - Gender match (binary indicator)
  - Category similarity (based on a manual similarity matrix with groupings)
  
  The system then outputs the top-K recommended items for each purchased item, along with store details and the distance from the user (calculated using the Haversine formula).

- **Store and User Location Simulation:**  
  Items are assigned to one of several stores with randomly generated geo-coordinates. A random user location is also generated to simulate the geographical aspect of recommendations.

## Project Structure
- **Feature Extraction & Caching:**  
  Extracts image features and caches them to disk. The metadata is similarly cached, allowing for faster subsequent runs.
  
- **Synthetic Data Generation:**  
  Generates synthetic purchase data that includes purchase date, customer ID, item ID, and associated metadata.
  
- **Recommendation Calculation:**  
  Calculates a fused relevance score for candidate items that are not yet purchased by a user. The recommendations take into account visual similarity, popularity, gender matching, and category similarity.


## Customization
- **Backbone Models:**  
Experiment with different CNN backbones by adding or modifying the entries in the `backbones` list.

- **Parameters:**
Adjust simulation parameters such as the number of customers, maximum purchases per customer, and top-K recommendations.

- **Category Similarity:**
Modify the manual similarity matrix and category groupings to suit different fashion item categorizations or business requirements.

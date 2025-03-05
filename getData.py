import kagglehub

# Download latest version
path = kagglehub.dataset_download("vishalbsadanand/deepfashion-1")

print("Path to dataset files:", path)
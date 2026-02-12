import kaggle
import os

# Create folders
os.makedirs("data/raw", exist_ok=True)

# Download (needs: pip install kaggle)
kaggle.api.dataset_download_files(
    'mlg-ulb/creditcardfraud', 
    path='data/raw/', 
    unzip=True
)

print("✅ Dataset downloaded: data/raw/creditcard.csv")

import os
os.environ['KAGGLE_USERNAME'] = 'suujayd'
os.environ['KAGGLE_KEY'] = 'KGAT_9b80edd8999b833d6131a4fb222180ad'

import kaggle
import os

# Create folders
os.makedirs("data/raw", exist_ok=True)

# Download dataset
kaggle.api.dataset_download_files(
    'mlg-ulb/creditcardfraud', 
    path='data/raw/', 
    unzip=True
)

print("✅ Dataset downloaded: data/raw/creditcard.csv")

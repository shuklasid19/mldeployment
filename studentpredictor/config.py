#contains all the paths for the datasets files

from pathlib import Path

class Config:
    RANDOM_SEED = 42
    ASSETS_PATH = Path("./assests")
    ORIGINAL_DATASET_FILE_PATH = ASSETS_PATH / "original_dataset" / "udemy_courses.csv"
    DATASET_PATH = ASSETS_PATH / "data" #test train split
    FEATURES_PATH = ASSETS_PATH / "features"
    MODELS_PATH = ASSETS_PATH / "models"
    METRICS_FILE_PATH = ASSETS_PATH / "metrics.json"


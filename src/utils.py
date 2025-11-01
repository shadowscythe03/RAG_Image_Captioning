"""
Utility functions and helpers.
Contains common utilities used across the project.
"""
import os
import json
import zipfile
import wget
from typing import List, Dict, Any
from PIL import Image
import matplotlib.pyplot as plt

def download_coco_dataset(base_dir: str = "./coco"):
    """
    Download COCO dataset images and annotations.
    
    Args:
        base_dir: Directory to download dataset to
    """
    print("📥 Downloading COCO dataset...")
    
    os.makedirs(base_dir, exist_ok=True)
    
    # URLs
    train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    try:
        # Download training images
        print("Downloading training images...")
        train_zip = os.path.join(base_dir, "train2017.zip")
        if not os.path.exists(train_zip):
            wget.download(train_images_url, train_zip)
        
        # Download annotations
        print("\nDownloading annotations...")
        ann_zip = os.path.join(base_dir, "annotations_trainval2017.zip")
        if not os.path.exists(ann_zip):
            wget.download(annotations_url, ann_zip)
        
        # Extract files
        print("\nExtracting training images...")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        print("✅ COCO dataset downloaded and extracted successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading COCO dataset: {e}")

def setup_ofa_environment():
    """Setup OFA model environment by cloning repository and installing dependencies."""
    print("🔧 Setting up OFA environment...")
    
    try:
        # Clone OFA repository
        os.system("git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git")
        
        # Install OFA transformers
        os.system("pip install OFA/transformers/")
        
        # Clone OFA-tiny model
        os.system("git lfs install")
        os.system("git clone https://huggingface.co/OFA-Sys/OFA-tiny")
        
        print("✅ OFA environment setup complete!")
        
    except Exception as e:
        print(f"❌ Error setting up OFA environment: {e}")

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    
    packages = [
        "chromadb",
        "torch torchvision",
        "pillow tqdm",
        "open_clip_torch",
        "matplotlib",
        "wget"
    ]
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
        
        print("✅ All dependencies installed successfully!")
        
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON file {file_path}: {e}")
        return {}

def save_json(data: Dict[str, Any], file_path: str):
    """Save data to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Data saved to {file_path}")
    except Exception as e:
        print(f"❌ Error saving JSON file {file_path}: {e}")

def get_image_info(image_path: str) -> Dict[str, Any]:
    """Get basic information about an image."""
    try:
        image = Image.open(image_path)
        return {
            "path": image_path,
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "filename": os.path.basename(image_path)
        }
    except Exception as e:
        print(f"❌ Error getting image info for {image_path}: {e}")
        return {}

def display_images_grid(image_paths: List[str], titles: List[str] = None, max_cols: int = 3):
    """
    Display multiple images in a grid layout.
    
    Args:
        image_paths: List of image file paths
        titles: Optional titles for each image
        max_cols: Maximum number of columns in grid
    """
    try:
        num_images = len(image_paths)
        cols = min(num_images, max_cols)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Handle single image case
        if num_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, img_path in enumerate(image_paths):
            if i < len(axes):
                try:
                    image = Image.open(img_path)
                    axes[i].imshow(image)
                    axes[i].axis('off')
                    
                    if titles and i < len(titles):
                        axes[i].set_title(titles[i], fontsize=10)
                    else:
                        axes[i].set_title(os.path.basename(img_path), fontsize=10)
                        
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error displaying images: {e}")

def clean_cache():
    """Clean up cache directories."""
    print("🧹 Cleaning cache...")
    
    cache_dirs = [
        "~/.cache/huggingface/hub",
        "__pycache__",
        ".pytest_cache"
    ]
    
    try:
        for cache_dir in cache_dirs:
            expanded_dir = os.path.expanduser(cache_dir)
            if os.path.exists(expanded_dir):
                os.system(f"rm -rf {expanded_dir}")
                print(f"Removed {cache_dir}")
        
        print("✅ Cache cleaned successfully!")
        
    except Exception as e:
        print(f"❌ Error cleaning cache: {e}")

def create_backup(source_dir: str, backup_name: str = None):
    """Create backup of a directory."""
    if backup_name is None:
        backup_name = f"{os.path.basename(source_dir)}_backup.zip"
    
    try:
        print(f"📦 Creating backup: {backup_name}")
        
        with zipfile.ZipFile(backup_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Backup created: {backup_name}")
        
    except Exception as e:
        print(f"❌ Error creating backup: {e}")

def log_experiment(experiment_name: str, config: Dict[str, Any], results: Dict[str, Any]):
    """Log experiment configuration and results."""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_data = {
        "experiment_name": experiment_name,
        "timestamp": str(datetime.now()),
        "config": config,
        "results": results
    }
    
    log_file = os.path.join(log_dir, f"{experiment_name}_{str(datetime.now().strftime('%Y%m%d_%H%M%S'))}.json")
    save_json(log_data, log_file)
    
    print(f"📝 Experiment logged: {log_file}")

# Import datetime for logging
from datetime import datetime
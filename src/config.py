"""
Configuration file for RAG Image Captioning project.
Contains all constants, paths, and hyperparameters.
"""
import os

# =============================
# Dataset Paths
# =============================
COCO_IMG_DIR = "./coco/train2017"
COCO_ANN_FILE = "./coco/annotations/captions_train2017.json"

# =============================
# Model Configurations  
# =============================
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# OFA Configuration
OFA_CHECKPOINT_DIR = "./OFA-tiny"
OFA_RESOLUTION = 224
OFA_MEAN = [0.5, 0.5, 0.5]
OFA_STD = [0.5, 0.5, 0.5]

# =============================
# Database Configuration
# =============================
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "coco_clip_embeddings"

# =============================
# Training Configuration
# =============================
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
MAX_TRAINING_STEPS = 100
LOG_INTERVAL = 50
EMBEDDING_BATCH_SIZE = 256

# =============================
# RAG Configuration
# =============================
DEFAULT_TOP_K = 5

# =============================
# Advanced Preprocessing Configuration
# =============================
# Segmentation and Object Detection
# Set to False if you encounter model loading issues
ENABLE_SEGMENTATION = True
ENABLE_OBJECT_DETECTION = True
CROPS_PER_MODE = 3  # Number of crops to generate per mode
MIN_CROP_SIZE = 64  # Minimum crop size (width/height)
CROP_PADDING = 10   # Padding around detected regions

# Segmentation model configuration
SEGMENTATION_MODEL = "facebook/detr-resnet-50-panoptic"
OBJECT_DETECTION_MODEL = "facebook/detr-resnet-50"

# Advanced RAG settings
CAPTIONS_PER_CROP = 3  # Number of captions to retrieve per crop
AGGREGATE_CAPTIONS = True  # Whether to intelligently aggregate captions

# Fallback configuration - automatically disable advanced features on errors
def get_safe_advanced_config():
    """Get safe configuration that disables advanced features if there are issues."""
    try:
        # Test if we can import required modules
        from transformers import DetrImageProcessor, DetrForObjectDetection, DetrForSegmentation
        return ENABLE_SEGMENTATION, ENABLE_OBJECT_DETECTION
    except ImportError:
        print("⚠️ Advanced preprocessing dependencies not available, disabling advanced features")
        return False, False

# =============================
# Output Configuration
# =============================
FINE_TUNED_MODEL_DIR = "./ofa_coco_clip_rag_finetuned"

# =============================
# Device Configuration
# =============================
def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# Path Validation
# =============================
def validate_paths():
    """Validate that required paths exist."""
    if os.path.exists(COCO_IMG_DIR) and os.path.exists(COCO_ANN_FILE):
        print("✅ COCO dataset paths validated")
        return True
    else:
        print("❌ COCO dataset paths not found")
        return False

def create_output_dirs():
    """Create necessary output directories."""
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)
    print("✅ Output directories created")
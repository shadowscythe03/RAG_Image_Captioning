# %% [markdown]
# # RAG Image Captioning with CLIP and ChromaDB
# 
# This notebook demonstrates retrieval-augmented generation (RAG) for image captioning using CLIP embeddings stored in ChromaDB.
# 
# ## Setup Steps:
# 1. Mount Google Drive (if using existing ChromaDB)
# 2. Install dependencies
# 3. Download COCO dataset (first time only)
# 4. Load models and setup ChromaDB
# 5. Test RAG retrieval on sample images

# %%
# # 📁 Mount Google Drive (if you have existing ChromaDB there)
# from google.colab import drive
# drive.mount('/content/drive')

# Set path to your ChromaDB folder (update this path as needed)
CHROMADB_PATH = "chroma_db"  # Update this path!

# Alternatively, if you don't have existing ChromaDB, we'll create it locally
USE_EXISTING_CHROMADB = True  # Set to True if you have existing ChromaDB in Drive

# %%
# 📦 Install required packages
# !pip install chromadb
# !pip install open_clip_torch
# !pip install torch torchvision
# !pip install pillow tqdm

print("✅ All packages installed!")

# %%
# 📦 Install additional packages for advanced preprocessing
# !pip install transformers  # For DETR models
# !pip install opencv-python  # For image processing

print("✅ Advanced preprocessing packages installed!")

# %%
# 📥 Download COCO Dataset (Run this only the first time!)
import os

# Check if COCO data already exists
# if not os.path.exists("coco"):
#     print("🔽 Downloading COCO dataset... This will take several minutes!")
    
#     # Download training images (about 13GB)
#     !wget http://images.cocodataset.org/zips/train2017.zip
#     !unzip -q train2017.zip -d ./coco/
    
#     # Download annotations
#     !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip  
#     !unzip -q annotations_trainval2017.zip -d ./coco/
    
#     # Clean up zip files
#     !rm train2017.zip annotations_trainval2017.zip
    
#     print("✅ COCO dataset downloaded!")
# else:
#     print("✅ COCO dataset already exists!")

# %%
# 🤖 Import libraries and setup
import os
import json
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import chromadb
import open_clip
from IPython.display import display

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# Dataset paths
COCO_IMG_DIR = "coco/train2017"
COCO_ANN_FILE = "coco/annotations/captions_train2017.json"

print("✅ Libraries imported successfully!")

# %%
# 🤖 Import additional libraries for advanced preprocessing
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrForSegmentation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("✅ Advanced preprocessing libraries imported!")

# %%
# 🔧 Load CLIP model
print("Loading CLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
clip_model = clip_model.to(device)
clip_model.eval()

print("✅ CLIP model loaded successfully!")

# %%
# 🎯 Load Object Detection and Segmentation Models (Simplified)
print("Loading DETR models for advanced preprocessing...")

# Configuration for advanced preprocessing
ENABLE_OBJECT_DETECTION = False
ENABLE_SEGMENTATION = False 
CROPS_PER_MODE = 2  # Reduced for better performance
MIN_CROP_SIZE = 64  # Minimum crop size
CROP_PADDING = 20   # Increased padding for better crops

# Object Detection Model
try:
    print("🔧 Loading object detection model...")
    object_detection_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    object_detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    object_detection_model.to(device)
    object_detection_model.eval()
    ENABLE_OBJECT_DETECTION = True
    print("✅ Object detection model loaded!")
except Exception as e:
    print(f"⚠️ Could not load object detection model: {e}")
    object_detection_model = None
    object_detection_processor = None

# Segmentation Model (Optional - can be disabled for simplicity)
try:
    print("🔧 Loading segmentation model...")
    segmentation_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    segmentation_model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    segmentation_model.to(device)
    segmentation_model.eval()
    ENABLE_SEGMENTATION = True
    print("✅ Segmentation model loaded!")
except Exception as e:
    print(f"⚠️ Could not load segmentation model: {e}")
    print("   This is okay - we can use object detection only")
    segmentation_model = None
    segmentation_processor = None

print(f"Advanced preprocessing ready:")
print(f"  • Object Detection: {'✅ Enabled' if ENABLE_OBJECT_DETECTION else '❌ Disabled'}")
print(f"  • Segmentation: {'✅ Enabled' if ENABLE_SEGMENTATION else '❌ Disabled'}")

# If neither works, disable advanced features
if not ENABLE_OBJECT_DETECTION and not ENABLE_SEGMENTATION:
    print("⚠️ No advanced models available - will use simple RAG only")
else:
    print(f"🚀 Advanced RAG will generate up to {CROPS_PER_MODE} crops per enabled mode")

# %%
# 🗄️ Setup ChromaDB
if USE_EXISTING_CHROMADB and os.path.exists(CHROMADB_PATH):
    print(f"📂 Using existing ChromaDB from: {CHROMADB_PATH}")
    chroma_dir = CHROMADB_PATH
else:
    print("📂 Creating new ChromaDB locally")
    chroma_dir = "/content/drive/MyDrive/NLP_Project/chroma_db"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=chroma_dir)

# Get or create collection
collection_name = "coco_clip_embeddings"
try:
    collection = client.get_collection(collection_name)
    print(f"✅ Loaded existing collection: {collection_name}")
    print(f"📊 Collection contains {collection.count()} embeddings")
except:
    collection = client.create_collection(collection_name)
    print(f"✅ Created new collection: {collection_name}")

print("ChromaDB setup complete!")

# %%
# 🏗️ Build ChromaDB from COCO (only if collection is empty)
if collection.count() == 0:
    print("🔨 Building ChromaDB from COCO dataset...")
    
    # Load COCO annotations
    with open(COCO_ANN_FILE, "r") as f:
        coco_data = json.load(f)
    
    annotations = coco_data["annotations"][:10000]  # Limit to 10k for faster demo
    
    # Create image path mapping
    img_id_to_path = {
        int(fn.split('.')[0]): os.path.join(COCO_IMG_DIR, fn)
        for fn in os.listdir(COCO_IMG_DIR) if fn.endswith(".jpg")
    }
    
    # Process images in batches
    batch_embeds = []
    batch_docs = []
    batch_ids = []
    batch_size = 100
    
    print(f"Processing {len(annotations)} annotations...")
    
    for ann in tqdm(annotations):
        img_id = ann["image_id"]
        ann_id = ann["id"]
        caption = ann["caption"]
        img_path = img_id_to_path.get(img_id)
        
        if not img_path or not os.path.exists(img_path):
            continue
        
        try:
            # Load and preprocess image
            image = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            
            # Generate embedding
            with torch.no_grad():
                img_emb = clip_model.encode_image(image)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                emb = img_emb.cpu().numpy()[0]
            
            batch_embeds.append(emb)
            batch_docs.append(caption)
            batch_ids.append(f"{img_id}_{ann_id}")
            
            # Add batch to collection when full
            if len(batch_embeds) >= batch_size:
                collection.add(
                    embeddings=np.array(batch_embeds).tolist(),
                    documents=batch_docs,
                    ids=batch_ids
                )
                batch_embeds, batch_docs, batch_ids = [], [], []
                
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            continue
    
    # Add remaining embeddings
    if batch_embeds:
        collection.add(
            embeddings=np.array(batch_embeds).tolist(),
            documents=batch_docs,
            ids=batch_ids
        )
    
    print(f"✅ ChromaDB built with {collection.count()} embeddings!")
else:
    print("✅ ChromaDB already exists with embeddings!")

# %%
# 🔍 RAG Retrieval Function
def retrieve_topk_captions_from_image_tensor(img_tensor, top_k=5):
    """Retrieve top-k similar captions for an image tensor."""
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_tensor)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
    
    query_emb = img_emb.cpu().numpy()[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    
    return " ".join(results["documents"][0])

def retrieve_captions_from_image_path(image_path, top_k=5):
    """Retrieve captions for an image file."""
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    return retrieve_topk_captions_from_image_tensor(image, top_k)

print("✅ RAG retrieval functions ready!")

# %%
# 🔧 Advanced Image Preprocessing Functions (Fixed Version)

def get_object_detection_crops(image_pil, top_k=CROPS_PER_MODE):
    """Generate crops based on object detection."""
    if not ENABLE_OBJECT_DETECTION:
        return []
    
    try:
        # Preprocess image
        inputs = object_detection_processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get detections
        with torch.no_grad():
            outputs = object_detection_model(**inputs)
        
        # Process results with lower threshold for more detections
        target_sizes = torch.tensor([image_pil.size[::-1]])
        results = object_detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3  # Lower threshold
        )[0]
        
        # Sort by confidence and take top detections
        scores = results["scores"]
        boxes = results["boxes"]
        
        if len(scores) == 0:
            print("⚠️ No objects detected above threshold")
            return []
        
        top_indices = scores.argsort(descending=True)[:top_k]
        
        crops = []
        for idx in top_indices:
            if idx < len(boxes):  # Safety check
                box = boxes[idx].cpu().numpy()
                crop = create_crop_from_bbox(image_pil, box)
                if crop is not None:
                    crops.append(crop)
        
        return crops
        
    except Exception as e:
        print(f"⚠️ Error in object detection: {e}")
        return []

def get_segmentation_crops(image_pil, top_k=CROPS_PER_MODE):
    """Generate crops based on image segmentation (Fixed Version)."""
    if not ENABLE_SEGMENTATION:
        return []
    
    try:
        # Preprocess image
        inputs = segmentation_processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get segmentation
        with torch.no_grad():
            outputs = segmentation_model(**inputs)
        
        # Use the new method (not deprecated)
        result = segmentation_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image_pil.size[::-1]]
        )[0]
        
        # Check for segmentation data in the result
        if "segmentation" in result:
            segmentation = result["segmentation"]
        elif hasattr(result, 'segmentation'):
            segmentation = result.segmentation
        else:
            print("⚠️ No segmentation data found in model output")
            return []
        
        # Get unique segments (excluding background)
        unique_segments = torch.unique(segmentation)
        valid_segments = [seg for seg in unique_segments if seg > 0]
        
        if len(valid_segments) == 0:
            print("⚠️ No valid segments found")
            return []
        
        # Sort by segment size
        segment_sizes = []
        for seg_id in valid_segments:
            mask = (segmentation == seg_id)
            size = mask.sum().item()
            if size > 100:  # Minimum segment size
                segment_sizes.append((seg_id, size))
        
        if len(segment_sizes) == 0:
            print("⚠️ No segments large enough")
            return []
        
        segment_sizes.sort(key=lambda x: x[1], reverse=True)
        top_segments = segment_sizes[:top_k]
        
        crops = []
        for seg_id, _ in top_segments:
            mask = (segmentation == seg_id)
            crop = create_crop_from_mask(image_pil, mask)
            if crop is not None:
                crops.append(crop)
        
        return crops
        
    except Exception as e:
        print(f"⚠️ Error in segmentation: {e}")
        return []

def create_crop_from_bbox(image_pil, bbox):
    """Create crop from bounding box."""
    try:
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure coordinates are within image bounds
        x_min = max(0, int(x_min) - CROP_PADDING)
        y_min = max(0, int(y_min) - CROP_PADDING)
        x_max = min(image_pil.width, int(x_max) + CROP_PADDING)
        y_max = min(image_pil.height, int(y_max) + CROP_PADDING)
        
        # Check minimum size
        if (x_max - x_min) < MIN_CROP_SIZE or (y_max - y_min) < MIN_CROP_SIZE:
            return None
        
        return image_pil.crop((x_min, y_min, x_max, y_max))
        
    except Exception as e:
        print(f"⚠️ Error creating bbox crop: {e}")
        return None

def create_crop_from_mask(image_pil, mask):
    """Create crop from segmentation mask."""
    try:
        # Convert mask to numpy
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Find bounding box of mask
        coords = np.column_stack(np.where(mask_np > 0))
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add padding
        x_min = max(0, x_min - CROP_PADDING)
        y_min = max(0, y_min - CROP_PADDING)
        x_max = min(image_pil.width, x_max + CROP_PADDING)
        y_max = min(image_pil.height, y_max + CROP_PADDING)
        
        # Check minimum size
        if (x_max - x_min) < MIN_CROP_SIZE or (y_max - y_min) < MIN_CROP_SIZE:
            return None
        
        return image_pil.crop((x_min, y_min, x_max, y_max))
        
    except Exception as e:
        print(f"⚠️ Error creating mask crop: {e}")
        return None

def get_all_image_variants(image_path):
    """Get original image + all crops for comprehensive RAG."""
    image_pil = Image.open(image_path)
    
    variants = [image_pil]  # Original image
    
    # Add object detection crops
    if ENABLE_OBJECT_DETECTION:
        obj_crops = get_object_detection_crops(image_pil)
        variants.extend(obj_crops)
        print(f"Generated {len(obj_crops)} object detection crops")
    
    # Add segmentation crops
    if ENABLE_SEGMENTATION:
        seg_crops = get_segmentation_crops(image_pil)
        variants.extend(seg_crops)
        print(f"Generated {len(seg_crops)} segmentation crops")
    
    return variants

print("✅ Advanced preprocessing functions ready! (Fixed Version)")

# %%
# 🔄 Enhanced RAG with Multi-Variant Retrieval (Improved Version)

def retrieve_captions_from_multiple_variants(image_variants, captions_per_variant=3):
    """Retrieve captions from multiple image variants and aggregate them."""
    all_captions = []
    individual_results = []
    
    for i, variant in enumerate(image_variants):
        try:
            # Preprocess variant for CLIP
            variant_tensor = clip_preprocess(variant).unsqueeze(0).to(device)
            
            # Retrieve captions for this variant
            captions = retrieve_topk_captions_from_image_tensor(variant_tensor, captions_per_variant)
            all_captions.append(captions)
            
            individual_results.append({
                'variant_index': i,
                'variant_type': 'original' if i == 0 else f'crop_{i}',
                'captions': captions
            })
            
        except Exception as e:
            print(f"⚠️ Error processing variant {i}: {e}")
            continue
    
    # Improved aggregation - preserve more context
    aggregated = smart_aggregate_captions(all_captions)
    
    return {
        'aggregated_captions': aggregated,
        'individual_results': individual_results,
        'total_variants': len(individual_results)
    }

def smart_aggregate_captions(caption_list):
    """Improved caption aggregation that preserves context better."""
    if not caption_list:
        return ""
    
    # If only one caption (original), return it as is
    if len(caption_list) == 1:
        return caption_list[0]
    
    # For multiple variants, combine but preserve original as primary
    original_captions = caption_list[0]  # Original image captions
    crop_captions = caption_list[1:] if len(caption_list) > 1 else []
    
    # Split into words and remove duplicates while preserving order
    original_words = original_captions.split()
    additional_words = []
    
    seen_words = set(word.lower().strip('.,!?') for word in original_words)
    
    # Add unique words from crops
    for crop_caption in crop_captions:
        for word in crop_caption.split():
            word_clean = word.lower().strip('.,!?')
            if (word_clean not in seen_words and 
                len(word_clean) > 2 and 
                word_clean.isalpha()):
                additional_words.append(word)
                seen_words.add(word_clean)
    
    # Combine: original + up to 20 additional unique words from crops
    if additional_words:
        combined = original_captions + " " + " ".join(additional_words[:20])
    else:
        combined = original_captions
    
    return combined

def enhanced_retrieve_captions(image_path, use_advanced=True):
    """Enhanced caption retrieval with optional advanced preprocessing."""
    if use_advanced and (ENABLE_OBJECT_DETECTION or ENABLE_SEGMENTATION):
        try:
            # Get all image variants
            variants = get_all_image_variants(image_path)
            
            # If we only have the original (no crops generated), fall back to simple
            if len(variants) <= 1:
                print("⚠️ No crops generated, falling back to simple RAG")
                return retrieve_captions_from_image_path(image_path)
            
            # Retrieve from all variants
            result = retrieve_captions_from_multiple_variants(variants)
            return result['aggregated_captions']
            
        except Exception as e:
            print(f"⚠️ Advanced retrieval failed: {e}, falling back to simple")
            return retrieve_captions_from_image_path(image_path)
    else:
        # Fallback to simple retrieval
        return retrieve_captions_from_image_path(image_path)

print("✅ Enhanced multi-variant RAG functions ready! (Improved Version)")

# %%
# 🎨 Visualization Functions for Crops

def visualize_image_processing(image_path, save_path=None):
    """Visualize original image and all generated crops."""
    try:
        # Get all variants
        image_pil = Image.open(image_path)
        obj_crops = get_object_detection_crops(image_pil) if ENABLE_OBJECT_DETECTION else []
        seg_crops = get_segmentation_crops(image_pil) if ENABLE_SEGMENTATION else []
        
        # Calculate grid size
        total_images = 1 + len(obj_crops) + len(seg_crops)
        cols = min(4, total_images)
        rows = (total_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if total_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        # Plot original image
        axes[0].imshow(image_pil)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        idx = 1
        
        # Plot object detection crops
        for i, crop in enumerate(obj_crops):
            if idx < len(axes):
                axes[idx].imshow(crop)
                axes[idx].set_title(f'Object Detection {i+1}', fontsize=10)
                axes[idx].axis('off')
                idx += 1
        
        # Plot segmentation crops
        for i, crop in enumerate(seg_crops):
            if idx < len(axes):
                axes[idx].imshow(crop)
                axes[idx].set_title(f'Segmentation {i+1}', fontsize=10)
                axes[idx].axis('off')
                idx += 1
        
        # Hide empty subplots
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return {
            'total_variants': total_images,
            'object_detection_crops': len(obj_crops),
            'segmentation_crops': len(seg_crops)
        }
        
    except Exception as e:
        print(f"⚠️ Error in visualization: {e}")
        return None

print("✅ Visualization functions ready!")

# %%
# 🧪 Simple RAG Demo (Reliable Version)
print("🧪 Simple RAG Demo with Optional Advanced Features...")

# Test with 2 random images (smaller test)
all_imgs = [f for f in os.listdir(COCO_IMG_DIR) if f.endswith(".jpg")]
test_imgs = random.sample(all_imgs, 2)

for i, img_name in enumerate(test_imgs, 1):
    print(f"\n{'='*60}")
    print(f"🖼️ Image {i}: {img_name}")
    print('='*60)
    
    img_path = os.path.join(COCO_IMG_DIR, img_name)
    
    # Display original image
    img = Image.open(img_path)
    display(img)
    
    try:
        # Always show simple RAG
        print(f"\n📝 Simple RAG Result:")
        simple_captions = retrieve_captions_from_image_path(img_path, top_k=5)
        print(f"'{simple_captions}'")
        
        # Try advanced RAG if models are available
        if ENABLE_OBJECT_DETECTION or ENABLE_SEGMENTATION:
            print(f"\n� Testing Advanced Processing...")
            
            try:
                # Test crop generation with better error handling
                variants = get_all_image_variants(img_path)
                
                if len(variants) > 1:
                    print(f"✅ Generated {len(variants)-1} additional crop(s)")
                    
                    # Show visualization
                    visualize_image_processing(img_path)
                    
                    # Advanced RAG
                    advanced_captions = enhanced_retrieve_captions(img_path, use_advanced=True)
                    print(f"\n🚀 Advanced RAG Result:")
                    print(f"'{advanced_captions}'")
                    
                    # Simple comparison
                    simple_words = len(simple_captions.split())
                    advanced_words = len(advanced_captions.split())
                    print(f"\n📊 Comparison: Simple={simple_words} words, Advanced={advanced_words} words")
                    
                else:
                    print(f"⚠️ No crops generated for this image - using simple RAG")
                    
            except Exception as e:
                print(f"⚠️ Advanced processing failed: {e}")
                print(f"   Using simple RAG only")
        else:
            print(f"\n💡 Advanced models not enabled - using simple RAG only")
            print(f"   To enable: Set ENABLE_OBJECT_DETECTION=True above")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")

print(f"\n✅ Demo completed!")
print(f"\n🎯 Summary:")
print(f"  ✅ Simple RAG: Always works with CLIP + ChromaDB")
if ENABLE_OBJECT_DETECTION:
    print(f"  ✅ Object Detection: Enabled (generates focused crops)")
else:
    print(f"  ⚠️ Object Detection: Disabled")
    
if ENABLE_SEGMENTATION:
    print(f"  ✅ Segmentation: Enabled (generates region crops)")  
else:
    print(f"  ⚠️ Segmentation: Disabled")

print(f"\n💡 For reliable results, focus on Simple RAG which works consistently!")
print(f"   Advanced features are experimental and may not work on all images.")

# %%
# 💾 Save ChromaDB to Drive (optional)
# if not USE_EXISTING_CHROMADB:
#     try:
#         # Create a zip of the ChromaDB for easy transfer
#         !zip -r chroma_db_backup.zip /content/chroma_db
        
#         # You can download this file or copy to Drive
#         print("📁 ChromaDB backup created: chroma_db_backup.zip")
#         print("💡 You can download this file to save your embeddings!")
        
#         # Optionally copy to Drive (uncomment if needed)
#         # !cp chroma_db_backup.zip "/content/drive/MyDrive/"
        
#     except Exception as e:
#         print(f"⚠️ Error creating backup: {e}")

print("✅ All done! Your RAG Image Captioning system is ready!")

# %% [markdown]
# ## 🆕 Advanced Image Processing Features
# 
# ### 🎯 Object Detection Mode
# - Uses **DETR-ResNet-50** to detect objects in images
# - Generates **focused crops** around detected objects with highest confidence
# - Retrieves **specific captions** for each detected object region
# 
# ### 🔍 Segmentation Mode  
# - Uses **DETR-ResNet-50-Panoptic** for semantic segmentation
# - Identifies **distinct semantic regions** in the image
# - Creates **intelligent crops** around the largest/most relevant segments
# 
# ### 🔄 Multi-Variant RAG Pipeline
# For each input image, the system now processes:
# 1. **Original Image** → Global scene context
# 2. **Object Detection Crops** → Specific object details  
# 3. **Segmentation Crops** → Semantic region information
# 
# All variants contribute to a **richer, more comprehensive** caption context!
# 
# ### 🎨 Visual Analysis
# - **Crop Visualization**: See exactly what regions are being analyzed
# - **Side-by-side Comparison**: Simple RAG vs Advanced Multi-Variant RAG
# - **Statistics**: Word count improvements and processing insights
# 
# ### ⚙️ Configuration
# - `CROPS_PER_MODE = 3`: Number of crops per processing mode
# - `MIN_CROP_SIZE = 64`: Minimum size for valid crops
# - `CROP_PADDING = 10`: Padding around detected regions

# %% [markdown]
# ## 🎯 Usage Instructions (Simplified)
# 
# ### ⚡ Quick Start (Reliable):
# 1. **Cells 1-7**: Setup environment and load CLIP model
# 2. **Cell 9-10**: Setup ChromaDB and build embeddings  
# 3. **Cell 11**: Test simple RAG retrieval ✅ **This always works!**
# 
# ### 🚀 Advanced Features (Optional):
# 4. **Cell 8**: Load DETR models (may fail on some systems)
# 5. **Cells 12-14**: Advanced preprocessing functions
# 6. **Cell 15**: Demo with crops (experimental)
# 
# ### 💡 Recommended Approach:
# - **Start Simple**: Use cells 1-11 for reliable RAG retrieval
# - **Test Advanced**: If models load successfully, try advanced features
# - **Focus on Results**: Simple RAG often gives better results than complex processing
# 
# ### 📝 Quick Commands:
# ```python
# # Simple RAG (always works)
# captions = retrieve_captions_from_image_path("image.jpg", top_k=5)
# 
# # Test any image quickly  
# quick_test_image("your_image.jpg")
# 
# # Compare methods (if advanced models available)
# compare_rag_methods("your_image.jpg")
# ```
# 
# ### ⚠️ Known Issues & Solutions:
# - **Segmentation Error**: Fixed with new post_process method
# - **Few Crops Generated**: Lowered detection thresholds  
# - **Worse Results**: Improved aggregation to preserve original context
# - **Model Loading Fails**: Advanced features are disabled automatically
# 
# ### 🎯 What Works Best:
# 1. **Simple RAG**: Consistently good results, fast, reliable
# 2. **Object Detection**: Sometimes helps with object-focused images
# 3. **Segmentation**: Experimental, often doesn't improve results
# 
# ### 💪 Core Strengths:
# - ✅ **CLIP + ChromaDB**: Robust similarity search
# - ✅ **10k+ COCO embeddings**: Rich context database  
# - ✅ **Fast retrieval**: Sub-second caption generation
# - ✅ **Easy testing**: Simple functions for experimentation
# 
# ### 🚀 Best Practice:
# **Use Simple RAG as your primary method** - it's fast, reliable, and often produces the best results. Advanced features are experimental additions that may or may not improve specific cases.

# %%
# 🔧 Simple Test Function (Error-Safe)
def test_single_image_simple(image_path, top_k=5):
    """Simple, error-safe test function for a single image."""
    try:
        print(f"🖼️ Testing image: {os.path.basename(image_path)}")
        
        # Load and display image
        img = Image.open(image_path)
        display(img)
        
        # Get image tensor for CLIP
        img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        
        # Retrieve similar captions
        retrieved_captions = retrieve_topk_captions_from_image_tensor(img_tensor, top_k)
        
        print(f"\n🔹 Retrieved Top-{top_k} Similar Captions:")
        caption_list = retrieved_captions.split(' ')[:50]  # Limit words for readability
        formatted_captions = ' '.join(caption_list)
        print(f"📝 {formatted_captions}")
        
        return {
            'success': True,
            'captions': retrieved_captions,
            'message': 'RAG retrieval successful!'
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': 'RAG retrieval failed'
        }

# Test with a sample image
if os.path.exists(COCO_IMG_DIR):
    # Pick the first available image
    sample_images = [f for f in os.listdir(COCO_IMG_DIR) if f.endswith('.jpg')][:3]
    
    for img_name in sample_images:
        img_path = os.path.join(COCO_IMG_DIR, img_name)
        print(f"\n{'='*50}")
        result = test_single_image_simple(img_path)
        print(f"✅ Result: {result['message']}")
        
else:
    print("⚠️ COCO images not found. Make sure to run the download cell first.")

# %%
# 🔍 Debug and Status Check
print("🔍 System Status Check:")
print(f"✅ Device: {device}")
print(f"✅ CLIP model loaded: {clip_model is not None}")
print(f"✅ ChromaDB collection: {collection.count()} embeddings")
print(f"✅ COCO images available: {len([f for f in os.listdir(COCO_IMG_DIR) if f.endswith('.jpg')]) if os.path.exists(COCO_IMG_DIR) else 0}")

# Test basic CLIP functionality
print("\n🧪 Testing CLIP embedding generation...")
try:
    if os.path.exists(COCO_IMG_DIR):
        sample_img_path = os.path.join(COCO_IMG_DIR, os.listdir(COCO_IMG_DIR)[0])
        sample_img = Image.open(sample_img_path)
        sample_tensor = clip_preprocess(sample_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = clip_model.encode_image(sample_tensor)
            print(f"✅ CLIP embedding shape: {embedding.shape}")
            print(f"✅ CLIP embedding generation working!")
    else:
        print("⚠️ No sample image available for testing")
        
except Exception as e:
    print(f"❌ CLIP test failed: {e}")

# Test ChromaDB query
print("\n🗄️ Testing ChromaDB query...")
try:
    if collection.count() > 0:
        # Test query with dummy embedding
        dummy_query = [0.1] * 512  # CLIP ViT-B-32 has 512-dim embeddings
        results = collection.query(query_embeddings=[dummy_query], n_results=3)
        print(f"✅ ChromaDB query successful!")
        print(f"✅ Retrieved {len(results['documents'][0])} documents")
    else:
        print("⚠️ ChromaDB collection is empty - run the build cell first")
        
except Exception as e:
    print(f"❌ ChromaDB test failed: {e}")

print("\n✅ Status check complete!")

# %%
# 🧪 Simple Advanced RAG Test (Reliable Version)
print("🧪 Testing Advanced RAG (Simple & Reliable)...")

# Get one test image
if os.path.exists(COCO_IMG_DIR):
    all_imgs = [f for f in os.listdir(COCO_IMG_DIR) if f.endswith(".jpg")]
    test_img = random.choice(all_imgs)
    test_path = os.path.join(COCO_IMG_DIR, test_img)
    
    print(f"\n🖼️ Test Image: {test_img}")
    
    # Display image
    img = Image.open(test_path)
    display(img)
    
    try:
        print(f"\n🔄 Testing Simple vs Advanced RAG:")
        
        # Simple RAG
        simple_captions = retrieve_captions_from_image_path(test_path, top_k=5)
        print(f"\n📝 Simple RAG Result:")
        print(f"'{simple_captions[:200]}{'...' if len(simple_captions) > 200 else ''}'")
        
        # Advanced RAG (only if models are available)
        if ENABLE_OBJECT_DETECTION or ENABLE_SEGMENTATION:
            print(f"\n🔧 Testing Advanced Preprocessing...")
            
            # Test crop generation
            variants = get_all_image_variants(test_path)
            print(f"   Total variants: {len(variants)} (1 original + {len(variants)-1} crops)")
            
            # Advanced retrieval
            advanced_captions = enhanced_retrieve_captions(test_path, use_advanced=True)
            print(f"\n🚀 Advanced RAG Result:")
            print(f"'{advanced_captions[:200]}{'...' if len(advanced_captions) > 200 else ''}'")
            
            # Comparison
            print(f"\n📊 Comparison:")
            print(f"  Simple:   {len(simple_captions.split())} words")
            print(f"  Advanced: {len(advanced_captions.split())} words")
            improvement = len(advanced_captions.split()) - len(simple_captions.split())
            print(f"  Change:   {'+' if improvement > 0 else ''}{improvement} words")
            
            # Quick visualization if crops were generated
            if len(variants) > 1:
                print(f"\n🎨 Visualizing crops...")
                visualize_image_processing(test_path)
        else:
            print(f"\n⚠️ Advanced models not available - using simple RAG only")
            print(f"   To enable: Set ENABLE_OBJECT_DETECTION=True or ENABLE_SEGMENTATION=True above")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

else:
    print("❌ COCO images not found. Run the download cell first.")

print(f"\n✅ Simple test completed!")

# %%
# 💡 Quick Usage Examples

# Example 1: Simple RAG on any image
def quick_test_image(image_filename):
    """Quick test function for any COCO image."""
    img_path = os.path.join(COCO_IMG_DIR, image_filename)
    if os.path.exists(img_path):
        # Show image
        display(Image.open(img_path))
        
        # Get captions
        captions = retrieve_captions_from_image_path(img_path, top_k=3)
        print(f"📝 Retrieved captions: {captions}")
        return captions
    else:
        print(f"❌ Image not found: {image_filename}")
        return None

# Example 2: Compare simple vs advanced (if models available)
def compare_rag_methods(image_filename):
    """Compare simple vs advanced RAG on an image."""
    img_path = os.path.join(COCO_IMG_DIR, image_filename)
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {image_filename}")
        return
    
    print(f"🖼️ Testing: {image_filename}")
    display(Image.open(img_path))
    
    # Simple
    simple = retrieve_captions_from_image_path(img_path)
    print(f"\n📝 Simple RAG: {simple[:150]}...")
    
    # Advanced (if available)
    if ENABLE_OBJECT_DETECTION or ENABLE_SEGMENTATION:
        advanced = enhanced_retrieve_captions(img_path, use_advanced=True)
        print(f"\n🚀 Advanced RAG: {advanced[:150]}...")
        print(f"\n📊 Words: Simple={len(simple.split())}, Advanced={len(advanced.split())}")
    else:
        print(f"\n⚠️ Advanced RAG not available (models not loaded)")

print("✅ Quick usage functions ready!")
print("💡 Try: quick_test_image('your_image.jpg') or compare_rag_methods('your_image.jpg')")



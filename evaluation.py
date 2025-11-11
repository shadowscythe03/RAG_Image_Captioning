"""
RAG Image Captioning Evaluation: Original vs Proposed Multi-Variant Setup

This script compares THREE approaches:
1. Original RAG + BLIP-2: Retrieve from original image only → Generate with LLM
2. Proposed RAG (Aggregation): Retrieve from original + crops → Concatenate text
3. Proposed RAG + BLIP-2: Retrieve from original + crops → Generate with LLM

Evaluation Metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr

Dataset:
- COCO val2017 (used as test set)
- Results saved to CSV with sample visualizations

GPU Memory Requirements:
- CLIP (ViT-B-32): ~1GB
- BLIP-2 (opt-2.7b): ~5GB
- DETR models (optional): ~10GB
- Total: ~6GB minimum (without DETR), ~16GB with DETR

Memory Optimization:
- DETR models disabled by default to save memory
- Reduced beam search from 5 to 3
- Aggressive memory clearing after each inference
- Use device_map="auto" for automatic placement

Required packages:
pip install chromadb open_clip_torch torch torchvision pillow tqdm transformers opencv-python matplotlib pycocotools pycocoevalcap pandas accelerate
"""

# 🔧 Environment Setup
import os
import sys

# Check if running in Colab
IS_COLAB = 'google.colab' in sys.modules

# Set base paths
if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content'
    GDRIVE_PATH = '/content/drive/MyDrive/NLP_Project'
    os.makedirs(GDRIVE_PATH, exist_ok=True)
else:
    BASE_PATH = '.'
    GDRIVE_PATH = '.'

# Define paths (easily convertible)
COCO_TRAIN_IMG = os.path.join(BASE_PATH, 'coco/train2017')
COCO_VAL_IMG = os.path.join(BASE_PATH, 'coco/val2017')
COCO_TRAIN_ANN = os.path.join(BASE_PATH, 'coco/annotations/captions_train2017.json')
COCO_VAL_ANN = os.path.join(BASE_PATH, 'coco/annotations/captions_val2017.json')
CHROMADB_PATH = os.path.join(GDRIVE_PATH, 'chroma_db')
RESULTS_PATH = os.path.join(GDRIVE_PATH, 'evaluation_results')

os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"🖥️ Running in: {'Colab' if IS_COLAB else 'Local/Server'}")
print(f"📁 Base path: {BASE_PATH}")
print(f"📁 Results path: {RESULTS_PATH}")
print(f"📁 ChromaDB path: {CHROMADB_PATH}")


# # 📦 Install Required Packages
# !pip install -q chromadb open_clip_torch torch torchvision pillow tqdm
# !pip install -q transformers opencv-python matplotlib
# !pip install -q pycocotools pycocoevalcap
# !pip install -q pandas

print("✅ All packages installed!")


# 📥 Step 1: Download COCO Dataset (if not present)
import os
import subprocess

def download_coco_dataset():
    """Download COCO train, val images and annotations if not present."""
    
    # Check what's missing
    missing = []
    if not os.path.exists(COCO_TRAIN_IMG):
        missing.append('train_images')
    if not os.path.exists(COCO_VAL_IMG):
        missing.append('val_images')
    if not os.path.exists(COCO_TRAIN_ANN) or not os.path.exists(COCO_VAL_ANN):
        missing.append('annotations')
    
    if not missing:
        print("✅ All COCO data already present!")
        return True
    
    print(f"🔽 Downloading missing COCO data: {', '.join(missing)}")
    
    os.makedirs(os.path.join(BASE_PATH, 'coco'), exist_ok=True)
    
    try:
        # Download train images if missing
        if 'train_images' in missing:
            print("📥 Downloading train2017 images (~13GB)...")
            subprocess.run(f"wget -q http://images.cocodataset.org/zips/train2017.zip -P {BASE_PATH}", shell=True, check=True)
            subprocess.run(f"unzip -q {BASE_PATH}/train2017.zip -d {BASE_PATH}/coco/", shell=True, check=True)
            subprocess.run(f"rm {BASE_PATH}/train2017.zip", shell=True, check=True)
            print("✅ Train images downloaded")
        
        # Download val images if missing
        if 'val_images' in missing:
            print("📥 Downloading val2017 images (~1GB)...")
            subprocess.run(f"wget -q http://images.cocodataset.org/zips/val2017.zip -P {BASE_PATH}", shell=True, check=True)
            subprocess.run(f"unzip -q {BASE_PATH}/val2017.zip -d {BASE_PATH}/coco/", shell=True, check=True)
            subprocess.run(f"rm {BASE_PATH}/val2017.zip", shell=True, check=True)
            print("✅ Val images downloaded")
        
        # Download annotations if missing
        if 'annotations' in missing:
            print("📥 Downloading annotations...")
            subprocess.run(f"wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {BASE_PATH}", shell=True, check=True)
            subprocess.run(f"unzip -q {BASE_PATH}/annotations_trainval2017.zip -d {BASE_PATH}/coco/", shell=True, check=True)
            subprocess.run(f"rm {BASE_PATH}/annotations_trainval2017.zip", shell=True, check=True)
            print("✅ Annotations downloaded")
        
        print("✅ COCO dataset download complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading COCO data: {e}")
        return False


# 🤖 Import Libraries
import json
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import chromadb
import open_clip
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrForSegmentation

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("✅ Libraries imported!")


# 🔧 Load CLIP Model
print("Loading CLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
clip_model = clip_model.to(device)
clip_model.eval()
print("✅ CLIP model loaded!")


# 🤖 Load BLIP-2 Vision-Language Model for Caption Generation
print("Loading BLIP-2 model for RAG caption generation...")
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import gc

# Clear GPU memory before loading BLIP-2
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"📊 GPU Memory before BLIP-2: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # Automatic device placement
    load_in_8bit=False  # Use 8-bit quantization if still OOM
)

# Only move to device if not using device_map
if not hasattr(blip_model, 'hf_device_map'):
    blip_model = blip_model.to(device)

blip_model.eval()

if torch.cuda.is_available():
    print(f"✅ BLIP-2 model loaded!")
    print(f"📊 GPU Memory after BLIP-2: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    print(f"📊 GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
else:
    print("✅ BLIP-2 model loaded!")
    
print("   This enables true Vision-Language RAG (image + retrieved context → LLM generation)")


# 🎯 Load DETR Models for Advanced Preprocessing
print("Loading DETR models...")

# Set to False to save GPU memory for BLIP-2
ENABLE_OBJECT_DETECTION = False
ENABLE_SEGMENTATION = False
CROPS_PER_MODE = 2
MIN_CROP_SIZE = 64
CROP_PADDING = 20

# print("⚠️ DETR models disabled to save GPU memory for BLIP-2")
# print("   To enable crops: uncomment DETR loading code below")

# Initialize as None
object_detection_processor = None
object_detection_model = None
segmentation_processor = None
segmentation_model = None

# Uncomment below to enable DETR models (requires more GPU memory ~10GB)

# Clear memory before loading DETR
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Object Detection
try:
    object_detection_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    object_detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    object_detection_model.to(device)
    object_detection_model.eval()
    ENABLE_OBJECT_DETECTION = True
    print("✅ Object detection model loaded")
except Exception as e:
    print(f"⚠️ Object detection not available: {e}")

# Segmentation
try:
    segmentation_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    segmentation_model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    segmentation_model.to(device)
    segmentation_model.eval()
    ENABLE_SEGMENTATION = True
    print("✅ Segmentation model loaded")
except Exception as e:
    print(f"⚠️ Segmentation not available: {e}")

print(f"\nAdvanced preprocessing: OD={ENABLE_OBJECT_DETECTION}, SEG={ENABLE_SEGMENTATION}")


# 🗄️ Step 2: Setup ChromaDB (Build if not exists)
def build_chromadb_from_coco(chroma_path, img_dir, ann_file, limit=10000):
    """Build ChromaDB from COCO training data."""
    print(f"🔨 Building ChromaDB at {chroma_path}...")
    
    # Initialize client
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Create collection
    collection_name = "coco_clip_embeddings"
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(collection_name)
    
    # Load annotations
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    annotations = coco_data['annotations'][:limit]
    
    # Create image mapping
    img_id_to_path = {
        int(fn.split('.')[0]): os.path.join(img_dir, fn)
        for fn in os.listdir(img_dir) if fn.endswith('.jpg')
    }
    
    # Process in batches
    batch_embeds, batch_docs, batch_ids = [], [], []
    batch_size = 100
    
    for ann in tqdm(annotations, desc="Building ChromaDB"):
        img_id = ann['image_id']
        ann_id = ann['id']
        caption = ann['caption']
        img_path = img_id_to_path.get(img_id)
        
        if not img_path or not os.path.exists(img_path):
            continue
        
        try:
            image = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                img_emb = clip_model.encode_image(image)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                emb = img_emb.cpu().numpy()[0]
            
            batch_embeds.append(emb)
            batch_docs.append(caption)
            batch_ids.append(f"{img_id}_{ann_id}")
            
            if len(batch_embeds) >= batch_size:
                collection.add(
                    embeddings=np.array(batch_embeds).tolist(),
                    documents=batch_docs,
                    ids=batch_ids
                )
                batch_embeds, batch_docs, batch_ids = [], [], []
        except Exception as e:
            continue
    
    # Add remaining
    if batch_embeds:
        collection.add(
            embeddings=np.array(batch_embeds).tolist(),
            documents=batch_docs,
            ids=batch_ids
        )
    
    print(f"✅ ChromaDB built with {collection.count()} embeddings")
    return client, collection

# Check if ChromaDB exists, if not build it
if os.path.exists(CHROMADB_PATH):
    print(f"📂 Loading existing ChromaDB from {CHROMADB_PATH}")
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    try:
        collection = client.get_collection("coco_clip_embeddings")
        print(f"✅ Loaded collection with {collection.count()} embeddings")
    except:
        print("⚠️ Collection not found, building...")
        client, collection = build_chromadb_from_coco(
            CHROMADB_PATH, COCO_TRAIN_IMG, COCO_TRAIN_ANN, limit=10000
        )
else:
    print("📂 ChromaDB not found, building from scratch...")
    client, collection = build_chromadb_from_coco(
        CHROMADB_PATH, COCO_TRAIN_IMG, COCO_TRAIN_ANN, limit=10000
    )

# Verify ChromaDB has data
collection_count = collection.count()
print(f"\n✅ ChromaDB setup complete!")
print(f"   Collection contains {collection_count} embeddings")

if collection_count == 0:
    raise ValueError("ChromaDB collection is empty! Cannot perform retrieval.")


# 🔧 Step 3: RAG Setup - Define Preprocessing Functions

def get_object_detection_crops(image_pil, top_k=CROPS_PER_MODE):
    """Generate crops from object detection."""
    if not ENABLE_OBJECT_DETECTION:
        return []
    try:
        inputs = object_detection_processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = object_detection_model(**inputs)
        
        target_sizes = torch.tensor([image_pil.size[::-1]])
        results = object_detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )[0]
        
        scores = results["scores"]
        boxes = results["boxes"]
        
        if len(scores) == 0:
            return []
        
        top_indices = scores.argsort(descending=True)[:top_k]
        
        crops = []
        for idx in top_indices:
            if idx < len(boxes):
                box = boxes[idx].cpu().numpy()
                x_min, y_min, x_max, y_max = box
                x_min = max(0, int(x_min) - CROP_PADDING)
                y_min = max(0, int(y_min) - CROP_PADDING)
                x_max = min(image_pil.width, int(x_max) + CROP_PADDING)
                y_max = min(image_pil.height, int(y_max) + CROP_PADDING)
                
                if (x_max - x_min) >= MIN_CROP_SIZE and (y_max - y_min) >= MIN_CROP_SIZE:
                    crops.append(image_pil.crop((x_min, y_min, x_max, y_max)))
        
        return crops
    except Exception as e:
        return []

def get_segmentation_crops(image_pil, top_k=CROPS_PER_MODE):
    """Generate crops from segmentation."""
    if not ENABLE_SEGMENTATION:
        return []
    try:
        inputs = segmentation_processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = segmentation_model(**inputs)
        
        result = segmentation_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image_pil.size[::-1]]
        )[0]
        
        if "segmentation" not in result:
            return []
        
        segmentation = result["segmentation"]
        unique_segments = torch.unique(segmentation)
        valid_segments = [seg for seg in unique_segments if seg > 0]
        
        segment_sizes = []
        for seg_id in valid_segments:
            mask = (segmentation == seg_id)
            size = mask.sum().item()
            if size > 100:
                segment_sizes.append((seg_id, size, mask))
        
        segment_sizes.sort(key=lambda x: x[1], reverse=True)
        
        crops = []
        for seg_id, _, mask in segment_sizes[:top_k]:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            coords = np.column_stack(np.where(mask_np > 0))
            if len(coords) == 0:
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            x_min = max(0, x_min - CROP_PADDING)
            y_min = max(0, y_min - CROP_PADDING)
            x_max = min(image_pil.width, x_max + CROP_PADDING)
            y_max = min(image_pil.height, y_max + CROP_PADDING)
            
            if (x_max - x_min) >= MIN_CROP_SIZE and (y_max - y_min) >= MIN_CROP_SIZE:
                crops.append(image_pil.crop((x_min, y_min, x_max, y_max)))
        
        return crops
    except Exception as e:
        return []

def retrieve_captions_from_tensor(img_tensor, top_k=5):
    """Retrieve captions for an image tensor."""
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_tensor)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
    
    query_emb = img_emb.cpu().numpy()[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    
    # Join captions and truncate to prevent context overflow
    captions = " ".join(results["documents"][0])
    # Truncate to ~100 words to prevent input overflow (rough approximation)
    words = captions.split()
    if len(words) > 100:
        captions = " ".join(words[:100]) + "..."
    
    return captions

print("✅ RAG preprocessing functions ready!")


# 🔄 RAG Retrieval Methods

def original_rag_setup(image_path, top_k=5):
    """
    Original RAG with LLM: 
    - Retrieve captions from ORIGINAL IMAGE ONLY (no crops)
    - Pass original image + retrieved context to BLIP-2
    - Generate caption with LLM
    """
    image = Image.open(image_path).convert('RGB')
    
    # Retrieve from original image only
    img_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    retrieved_captions = retrieve_captions_from_tensor(img_tensor, top_k)
    del img_tensor  # Free memory
    
    # Create prompt for BLIP-2
    prompt = f"""Generate a natural, descriptive caption for this image.

Retrieved captions from similar images in the dataset:
{retrieved_captions}

Based on what you see in the image and the retrieved captions above, generate a concise caption:"""
    
    # Generate caption with BLIP-2 (vision + language)
    inputs = blip_processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device, torch.float16 if torch.cuda.is_available() else torch.float32)
    
    with torch.no_grad():
        generated_ids = blip_model.generate(
            **inputs,
            max_new_tokens=50,  # Generate max 50 NEW tokens (input length doesn't count)
            num_beams=3,  # Reduced from 5 to save memory
            temperature=0.7,
            do_sample=False,
            length_penalty=1.0
        )
    
    generated_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Clean up memory
    del inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clean up prompt prefix if present
    if "caption:" in generated_caption.lower():
        generated_caption = generated_caption.split("caption:", 1)[-1].strip()
    
    return {
        'retrieved_captions': generated_caption,  # Final LLM-generated caption
        'raw_retrieved': retrieved_captions,  # Original retrieved captions
        'num_crops': 0,
        'crops': []
    }

def proposed_rag_setup(image_path, top_k=3):
    """Proposed: Retrieve captions from original + crops."""
    image = Image.open(image_path)
    
    # Get crops
    obj_crops = get_object_detection_crops(image)
    seg_crops = get_segmentation_crops(image)
    all_crops = obj_crops + seg_crops
    
    # Retrieve from original
    img_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    original_captions = retrieve_captions_from_tensor(img_tensor, top_k)
    
    # Retrieve from crops
    crop_captions = []
    for crop in all_crops:
        crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)
        crop_caption = retrieve_captions_from_tensor(crop_tensor, top_k)
        crop_captions.append(crop_caption)
    
    # Aggregate: original + unique words from crops
    original_words = original_captions.split()
    seen_words = set(w.lower().strip('.,!?') for w in original_words)
    
    additional_words = []
    for crop_cap in crop_captions:
        for word in crop_cap.split():
            word_clean = word.lower().strip('.,!?')
            if word_clean not in seen_words and len(word_clean) > 2 and word_clean.isalpha():
                additional_words.append(word)
                seen_words.add(word_clean)
    
    # Combine
    if additional_words:
        aggregated = original_captions + " " + " ".join(additional_words[:20])
    else:
        aggregated = original_captions
    
    return {
        'retrieved_captions': aggregated,
        'original_captions': original_captions,
        'crop_captions': crop_captions,
        'num_crops': len(all_crops),
        'crops': all_crops
    }

print("✅ RAG retrieval methods ready!")


# 🎯 RAG with Vision-Language Generation (BLIP-2)

def proposed_rag_with_llm(image_path, top_k=3):
    """
    True Vision-Language RAG: 
    - Retrieve captions from ChromaDB (original + crops)
    - Pass original image + retrieved context to BLIP-2
    - Generate final caption with LLM
    """
    image = Image.open(image_path).convert('RGB')
    
    # Get crops
    obj_crops = get_object_detection_crops(image)
    seg_crops = get_segmentation_crops(image)
    all_crops = obj_crops + seg_crops
    
    # Retrieve from original
    img_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    original_captions = retrieve_captions_from_tensor(img_tensor, top_k)
    
    # Retrieve from crops
    crop_captions = []
    for crop in all_crops[:3]:  # Limit to top 3 crops for context
        crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)
        crop_caption = retrieve_captions_from_tensor(crop_tensor, top_k=2)
        crop_captions.append(crop_caption)
    
    # Prepare retrieved context
    retrieved_context = f"Original image: {original_captions}"
    if crop_captions:
        retrieved_context += f"\n\nImage regions: {' | '.join(crop_captions)}"
    
    # Create prompt for BLIP-2
    prompt = f"""Generate a natural, descriptive caption for this image.

Retrieved captions from similar images in the dataset:
{retrieved_context}

Based on what you see in the image and the retrieved context above, generate a concise caption:"""
    
    # Generate caption with BLIP-2 (vision + language)
    inputs = blip_processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device, torch.float16 if torch.cuda.is_available() else torch.float32)
    
    with torch.no_grad():
        generated_ids = blip_model.generate(
            **inputs,
            max_new_tokens=50,  # Generate max 50 NEW tokens (input length doesn't count)
            num_beams=3,  # Reduced from 5 to save memory
            temperature=0.7,
            do_sample=False,
            length_penalty=1.0
        )
    
    generated_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Clean up memory
    del inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clean up prompt prefix if present
    if "caption:" in generated_caption.lower():
        generated_caption = generated_caption.split("caption:", 1)[-1].strip()
    
    return {
        'retrieved_captions': generated_caption,  # Final LLM-generated caption
        'raw_retrieved_context': retrieved_context,  # Original retrieved captions
        'original_captions': original_captions,
        'crop_captions': crop_captions,
        'num_crops': len(all_crops),
        'crops': all_crops
    }

print("✅ Vision-Language RAG with BLIP-2 ready!")


# 📊 Evaluation Setup with pycocoeval
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import tempfile

def evaluate_with_coco_metrics(image_ids, generated_captions, gt_annotations_file):
    """
    Evaluate generated captions using COCO evaluation metrics.
    
    Args:
        image_ids: List of image IDs
        generated_captions: Dict mapping image_id to caption string
        gt_annotations_file: Path to ground truth annotations
    
    Returns:
        Dict with BLEU, METEOR, ROUGE, CIDEr scores
    """
    # Load ground truth
    coco = COCO(gt_annotations_file)
    
    # Filter to only IDs that exist in ground truth annotations
    gt_img_ids = set(coco.getImgIds())
    valid_image_ids = [img_id for img_id in image_ids if img_id in gt_img_ids]
    
    # Create results in COCO format
    results = []
    for img_id in valid_image_ids:
        if img_id in generated_captions and generated_captions[img_id]:
            results.append({
                'image_id': img_id,
                'caption': generated_captions[img_id]
            })
    
    # Check if we have any results
    if len(results) == 0:
        print(f"⚠️ ERROR: No captions generated for any images!")
        print(f"   - Total image IDs: {len(image_ids)}")
        print(f"   - Valid GT image IDs: {len(valid_image_ids)}")
        print(f"   - Generated captions: {len(generated_captions)}")
        print(f"   - Sample caption keys: {list(generated_captions.keys())[:5]}")
        print(f"   - Sample caption values: {list(generated_captions.values())[:3]}")
        raise ValueError("No captions were generated. Check ChromaDB and image processing.")
    
    print(f"   Generated {len(results)} captions for evaluation")
    
    # Save results to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(results, f)
        results_file = f.name
    
    # Load results
    coco_result = coco.loadRes(results_file)
    
    # Evaluate - only on valid image IDs that have both GT and generated captions
    result_img_ids = [r['image_id'] for r in results]
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = result_img_ids
    coco_eval.evaluate()
    
    # Clean up
    os.remove(results_file)
    
    return coco_eval.eval

print("✅ Evaluation setup ready!")


# 🧪 Run Evaluation on COCO Val Set

# Load validation annotations
with open(COCO_VAL_ANN, 'r') as f:
    val_data = json.load(f)

# Get image IDs (sample for faster evaluation)
val_images = val_data['images']
NUM_TEST_IMAGES = 50  # Reduced for faster evaluation (50 images ~ 30-40 mins)
test_images = random.sample(val_images, min(NUM_TEST_IMAGES, len(val_images)))

print(f"🧪 Evaluating on {len(test_images)} images from COCO val2017")
print(f"This will take approximately {len(test_images) * 1.0:.1f} minutes...\n")

# Clear memory before starting evaluation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"📊 Starting GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

# Storage for results
original_captions = {}
proposed_captions = {}
proposed_llm_captions = {}  # NEW: LLM-generated captions
detailed_results = []

# Process each image
for idx, img_info in enumerate(tqdm(test_images, desc="Processing images")):
    img_id = img_info['id']
    img_filename = img_info['file_name']
    img_path = os.path.join(COCO_VAL_IMG, img_filename)
    
    if not os.path.exists(img_path):
        continue
    
    try:
        # Original RAG (retrieval only)
        original_result = original_rag_setup(img_path, top_k=5)
        original_captions[img_id] = original_result['retrieved_captions']
        
        # Proposed RAG (retrieval + aggregation)
        proposed_result = proposed_rag_setup(img_path, top_k=3)
        proposed_captions[img_id] = proposed_result['retrieved_captions']
        
        # NEW: Proposed RAG with LLM (retrieval + generation)
        proposed_llm_result = proposed_rag_with_llm(img_path, top_k=3)
        proposed_llm_captions[img_id] = proposed_llm_result['retrieved_captions']
        
        # Store detailed results for first 10 images
        if idx < 10:
            detailed_results.append({
                'image_id': img_id,
                'filename': img_filename,
                'original_caption': original_result['retrieved_captions'],
                'proposed_caption': proposed_result['retrieved_captions'],
                'proposed_llm_caption': proposed_llm_result['retrieved_captions'],
                'retrieved_context': proposed_llm_result.get('raw_retrieved_context', ''),
                'num_crops': proposed_result['num_crops'],
                'crops': proposed_result.get('crops', []),
                'crop_captions': proposed_result.get('crop_captions', [])
            })
    
    except Exception as e:
        print(f"\n⚠️ Error processing {img_filename}: {e}")
        # Clear memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue
    
    # Periodic memory cleanup (every 10 images)
    if (idx + 1) % 10 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"   [Memory cleanup at image {idx+1}]")

print(f"\n✅ Processed {len(original_captions)} images successfully")

# Final memory report
if torch.cuda.is_available():
    print(f"📊 Final GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")


# 📊 Compute Metrics
print("📊 Computing evaluation metrics...\n")

# Get common image IDs
common_ids = list(set(original_captions.keys()) & set(proposed_captions.keys()) & set(proposed_llm_captions.keys()))

# Evaluate Original Setup (retrieval only)
print("Evaluating Original RAG Setup (Retrieval Only)...")
original_metrics = evaluate_with_coco_metrics(
    common_ids, original_captions, COCO_VAL_ANN
)

# Evaluate Proposed Setup (retrieval + aggregation)
print("Evaluating Proposed RAG Setup (Retrieval + Aggregation)...")
proposed_metrics = evaluate_with_coco_metrics(
    common_ids, proposed_captions, COCO_VAL_ANN
)

# Evaluate Proposed Setup with LLM (retrieval + generation)
print("Evaluating Proposed RAG with BLIP-2 (Retrieval + Generation)...")
proposed_llm_metrics = evaluate_with_coco_metrics(
    common_ids, proposed_llm_captions, COCO_VAL_ANN
)

print("\n✅ Metrics computed!")


# 📈 Display and Save Results

# Create comparison DataFrame
metrics_comparison = pd.DataFrame({
    'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr'],
    'Original RAG\n(Retrieval Only)': [
        original_metrics.get('Bleu_1', 0),
        original_metrics.get('Bleu_2', 0),
        original_metrics.get('Bleu_3', 0),
        original_metrics.get('Bleu_4', 0),
        original_metrics.get('METEOR', 0),
        original_metrics.get('ROUGE_L', 0),
        original_metrics.get('CIDEr', 0)
    ],
    'Proposed RAG\n(Retrieval + Aggregation)': [
        proposed_metrics.get('Bleu_1', 0),
        proposed_metrics.get('Bleu_2', 0),
        proposed_metrics.get('Bleu_3', 0),
        proposed_metrics.get('Bleu_4', 0),
        proposed_metrics.get('METEOR', 0),
        proposed_metrics.get('ROUGE_L', 0),
        proposed_metrics.get('CIDEr', 0)
    ],
    'Proposed RAG + BLIP-2\n(Retrieval + Generation)': [
        proposed_llm_metrics.get('Bleu_1', 0),
        proposed_llm_metrics.get('Bleu_2', 0),
        proposed_llm_metrics.get('Bleu_3', 0),
        proposed_llm_metrics.get('Bleu_4', 0),
        proposed_llm_metrics.get('METEOR', 0),
        proposed_llm_metrics.get('ROUGE_L', 0),
        proposed_llm_metrics.get('CIDEr', 0)
    ]
})

# Display
print("\n" + "="*100)
print("📊 EVALUATION RESULTS COMPARISON")
print("="*100)
print(metrics_comparison.to_string(index=False))
print("="*100)

# Save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(RESULTS_PATH, f'evaluation_results_{timestamp}.csv')
metrics_comparison.to_csv(csv_path, index=False)
print(f"\n💾 Results saved to: {csv_path}")

# Save metadata
metadata = {
    'num_test_images': len(common_ids),
    'enable_object_detection': ENABLE_OBJECT_DETECTION,
    'enable_segmentation': ENABLE_SEGMENTATION,
    'crops_per_mode': CROPS_PER_MODE,
    'model': 'BLIP-2 (Salesforce/blip2-opt-2.7b)',
    'timestamp': timestamp
}
metadata_path = os.path.join(RESULTS_PATH, f'metadata_{timestamp}.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"💾 Metadata saved to: {metadata_path}")


# 🎨 Step 4: Save Sample Results with Visualizations

print("\n🎨 Saving sample results with visualizations...\n")

samples_dir = os.path.join(RESULTS_PATH, f'sample_results_{timestamp}')
os.makedirs(samples_dir, exist_ok=True)

for idx, result in enumerate(detailed_results[:5]):  # Save 5 samples
    img_path = os.path.join(COCO_VAL_IMG, result['filename'])
    
    if not os.path.exists(img_path):
        continue
    
    # Create figure
    num_crops = len(result['crops'])
    fig = plt.figure(figsize=(16, 6))
    
    # Original image
    ax1 = plt.subplot(1, num_crops + 1, 1)
    img = Image.open(img_path)
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Crop images
    for i, crop in enumerate(result['crops']):
        ax = plt.subplot(1, num_crops + 1, i + 2)
        ax.imshow(crop)
        ax.set_title(f'Crop {i+1}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(samples_dir, f'sample_{idx+1}_visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save text results
    text_path = os.path.join(samples_dir, f'sample_{idx+1}_results.txt')
    with open(text_path, 'w') as f:
        f.write(f"Image: {result['filename']}\n")
        f.write(f"Image ID: {result['image_id']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("ORIGINAL RAG SETUP (Original Image Only)\n")
        f.write("="*80 + "\n")
        f.write(f"Retrieved Captions:\n{result['original_caption']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PROPOSED RAG SETUP (Original + Crops)\n")
        f.write("="*80 + "\n")
        f.write(f"Number of Crops Generated: {result['num_crops']}\n\n")
        
        if result['crop_captions']:
            f.write("Crop Captions:\n")
            for i, cap in enumerate(result['crop_captions']):
                f.write(f"  Crop {i+1}: {cap}\n")
            f.write("\n")
        
        f.write(f"Final Aggregated Caption:\n{result['proposed_caption']}\n\n")
    
    print(f"✅ Saved sample {idx+1}: {result['filename']}")

print(f"\n💾 Sample results saved to: {samples_dir}")
print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
print(f"\n📁 All results saved in: {RESULTS_PATH}")
print(f"  - Metrics CSV: evaluation_results_{timestamp}.csv")
print(f"  - Metadata: metadata_{timestamp}.json")
print(f"  - Sample visualizations: sample_results_{timestamp}/")


# 📊 Create Summary Visualization

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(metrics_comparison['Metric']))
width = 0.25

bars1 = ax.bar(x - width, metrics_comparison['Original RAG\n(Retrieval Only)'], width,
               label='Original RAG\n(Retrieval Only)', color='steelblue')
bars2 = ax.bar(x, metrics_comparison['Proposed RAG\n(Retrieval + Aggregation)'], width,
               label='Proposed RAG\n(Retrieval + Aggregation)', color='coral')
bars3 = ax.bar(x + width, metrics_comparison['Proposed RAG + BLIP-2\n(Retrieval + Generation)'], width,
               label='Proposed RAG + BLIP-2\n(Retrieval + Generation)', color='green')

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('RAG Image Captioning: Comparison of Three Approaches',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_comparison['Metric'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(RESULTS_PATH, f'metrics_comparison_{timestamp}.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n💾 Comparison chart saved to: {chart_path}")

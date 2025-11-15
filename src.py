"""
Consolidated backend module for RAG Image Captioning.
Contains all core functionality: models, preprocessing, retrieval, and database operations.
"""
import os
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path

# Core ML imports
import open_clip
from torchvision import transforms
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrForSegmentation
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Database
import chromadb
from tqdm import tqdm


# =============================
# Configuration
# =============================
class Config:
    """Configuration constants."""
    # Model settings
    CLIP_MODEL_NAME = "ViT-B-32"
    CLIP_PRETRAINED = "openai"
    
    # LLM Caption Generation
    ENABLE_LLM_GENERATION = False  # Set to True to use BLIP-2 for caption generation
    BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"  # or "Salesforce/blip2-flan-t5-xl"
    MAX_NEW_TOKENS = 50
    NUM_BEAMS = 3
    TEMPERATURE = 0.7
    
    # Advanced preprocessing
    ENABLE_SEGMENTATION = True
    ENABLE_OBJECT_DETECTION = True
    CROPS_PER_MODE = 3
    MIN_CROP_SIZE = 64
    CROP_PADDING = 10
    
    # Segmentation/Detection models
    SEGMENTATION_MODEL = "facebook/detr-resnet-50-panoptic"
    OBJECT_DETECTION_MODEL = "facebook/detr-resnet-50"
    
    # RAG settings
    DEFAULT_TOP_K = 5
    CAPTIONS_PER_CROP = 3
    AGGREGATE_CAPTIONS = True
    
    # Database
    CHROMA_DB_DIR = "./chroma_db"
    COLLECTION_NAME = "coco_clip_embeddings"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# Model Manager
# =============================
class ModelManager:
    """Manages CLIP and preprocessing models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        
        # BLIP-2 for LLM caption generation
        self.blip2_processor = None
        self.blip2_model = None
        
    def load_clip_model(self):
        """Load and initialize CLIP model."""
        if self.clip_model is not None:
            return self.clip_model, self.clip_preprocess
            
        print(f"Loading CLIP model ({self.config.CLIP_MODEL_NAME}) on {self.device}...")
        
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.config.CLIP_MODEL_NAME,
            pretrained=self.config.CLIP_PRETRAINED
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        print(f"✅ CLIP model loaded on {self.device}")
        return self.clip_model, self.clip_preprocess
    
    def load_blip2_model(self):
        """Load and initialize BLIP-2 model for caption generation."""
        if self.blip2_model is not None:
            return self.blip2_model, self.blip2_processor
        
        if not self.config.ENABLE_LLM_GENERATION:
            print("⚠️ LLM generation is disabled in config")
            return None, None
        
        print(f"Loading BLIP-2 model ({self.config.BLIP2_MODEL_NAME}) on {self.device}...")
        print("⏳ This may take a few minutes for first-time download...")
        
        try:
            self.blip2_processor = Blip2Processor.from_pretrained(self.config.BLIP2_MODEL_NAME)
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.BLIP2_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.blip2_model = self.blip2_model.to(self.device)
            self.blip2_model.eval()
            
            print(f"✅ BLIP-2 model loaded on {self.device}")
            return self.blip2_model, self.blip2_processor
        except Exception as e:
            print(f"❌ Failed to load BLIP-2 model: {e}")
            print("⚠️ LLM caption generation will not be available")
            return None, None
    
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode image using CLIP model."""
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded. Call load_clip_model() first.")
        
        with torch.no_grad():
            img_emb = self.clip_model.encode_image(image_tensor)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            return img_emb


# =============================
# Image Preprocessor
# =============================
class ImagePreprocessor:
    """Handles advanced image preprocessing with segmentation and object detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        # Detection models
        self.segmentation_processor = None
        self.segmentation_model = None
        self.object_detection_processor = None
        self.object_detection_model = None
        
        self._initialize_detection_models()
    
    def _initialize_detection_models(self):
        """Initialize segmentation and object detection models."""
        try:
            if self.config.ENABLE_SEGMENTATION:
                print("🔧 Loading segmentation model...")
                self.segmentation_processor = DetrImageProcessor.from_pretrained(
                    self.config.SEGMENTATION_MODEL
                )
                self.segmentation_model = DetrForSegmentation.from_pretrained(
                    self.config.SEGMENTATION_MODEL
                )
                self.segmentation_model.to(self.device)
                self.segmentation_model.eval()
                print("✅ Segmentation model loaded")
            
            if self.config.ENABLE_OBJECT_DETECTION:
                print("🔧 Loading object detection model...")
                self.object_detection_processor = DetrImageProcessor.from_pretrained(
                    self.config.OBJECT_DETECTION_MODEL
                )
                self.object_detection_model = DetrForObjectDetection.from_pretrained(
                    self.config.OBJECT_DETECTION_MODEL
                )
                self.object_detection_model.to(self.device)
                self.object_detection_model.eval()
                print("✅ Object detection model loaded")
                
        except Exception as e:
            print(f"⚠️ Could not load detection models: {e}")
            print("Will use basic preprocessing only")
    
    def preprocess_for_clip(self, image: Union[str, Image.Image], 
                           clip_preprocess) -> torch.Tensor:
        """Preprocess image for CLIP model."""
        if isinstance(image, str):
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return clip_preprocess(image).unsqueeze(0)
    
    def get_segmentation_crops(self, image: Image.Image) -> List[Image.Image]:
        """Generate crops based on image segmentation."""
        if not self.config.ENABLE_SEGMENTATION or self.segmentation_model is None:
            return []
        
        try:
            inputs = self.segmentation_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.segmentation_model(**inputs)
            
            result = self.segmentation_processor.post_process_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
            
            crops = []
            segmentation = result["segmentation"]
            unique_segments = torch.unique(segmentation)
            valid_segments = [seg for seg in unique_segments if seg > 0]
            
            segment_sizes = []
            for seg_id in valid_segments:
                mask = (segmentation == seg_id)
                size = mask.sum().item()
                segment_sizes.append((seg_id, size))
            
            segment_sizes.sort(key=lambda x: x[1], reverse=True)
            top_segments = segment_sizes[:self.config.CROPS_PER_MODE]
            
            for seg_id, _ in top_segments:
                crop = self._create_crop_from_mask(image, segmentation == seg_id)
                if crop is not None:
                    crops.append(crop)
            
            return crops
            
        except Exception as e:
            print(f"⚠️ Segmentation error: {e}")
            return []
    
    def get_object_detection_crops(self, image: Image.Image) -> List[Image.Image]:
        """Generate crops based on object detection."""
        if not self.config.ENABLE_OBJECT_DETECTION or self.object_detection_model is None:
            return []
        
        try:
            inputs = self.object_detection_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.object_detection_model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.object_detection_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]
            
            scores = results["scores"]
            boxes = results["boxes"]
            labels = results["labels"]
            
            top_indices = scores.argsort(descending=True)[:self.config.CROPS_PER_MODE]
            
            crops = []
            detected_objects = []
            
            for idx in top_indices:
                box = boxes[idx].cpu().numpy()
                label_id = labels[idx].item()
                score = scores[idx].item()
                
                crop = self._create_crop_from_bbox(image, box)
                if crop is not None:
                    crops.append(crop)
                    detected_objects.append({
                        'label_id': label_id,
                        'score': score,
                        'box': box.tolist()
                    })
            
            return crops, detected_objects
            
        except Exception as e:
            print(f"⚠️ Object detection error: {e}")
            return [], []
    
    def _create_crop_from_mask(self, image: Image.Image, 
                              mask: torch.Tensor) -> Optional[Image.Image]:
        """Create crop from segmentation mask."""
        try:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            coords = np.column_stack(np.where(mask_np > 0))
            
            if len(coords) == 0:
                return None
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            x_min = max(0, x_min - self.config.CROP_PADDING)
            y_min = max(0, y_min - self.config.CROP_PADDING)
            x_max = min(image.width, x_max + self.config.CROP_PADDING)
            y_max = min(image.height, y_max + self.config.CROP_PADDING)
            
            if (x_max - x_min) < self.config.MIN_CROP_SIZE or \
               (y_max - y_min) < self.config.MIN_CROP_SIZE:
                return None
            
            return image.crop((x_min, y_min, x_max, y_max))
            
        except Exception as e:
            print(f"⚠️ Crop from mask error: {e}")
            return None
    
    def _create_crop_from_bbox(self, image: Image.Image, 
                              bbox: np.ndarray) -> Optional[Image.Image]:
        """Create crop from bounding box."""
        try:
            x_min, y_min, x_max, y_max = bbox
            
            x_min = max(0, x_min - self.config.CROP_PADDING)
            y_min = max(0, y_min - self.config.CROP_PADDING)
            x_max = min(image.width, x_max + self.config.CROP_PADDING)
            y_max = min(image.height, y_max + self.config.CROP_PADDING)
            
            if (x_max - x_min) < self.config.MIN_CROP_SIZE or \
               (y_max - y_min) < self.config.MIN_CROP_SIZE:
                return None
            
            return image.crop((x_min, y_min, x_max, y_max))
            
        except Exception as e:
            print(f"⚠️ Crop from bbox error: {e}")
            return None
    
    def get_all_variants(self, image: Union[str, Image.Image]) -> Dict:
        """Get original image and all crops/detections."""
        if isinstance(image, str):
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        result = {
            'original': image,
            'segmentation_crops': [],
            'object_detection_crops': [],
            'detected_objects': []
        }
        
        if self.config.ENABLE_SEGMENTATION:
            print("🔍 Generating segmentation crops...")
            result['segmentation_crops'] = self.get_segmentation_crops(image)
            print(f"Generated {len(result['segmentation_crops'])} segmentation crops")
        
        if self.config.ENABLE_OBJECT_DETECTION:
            print("🎯 Detecting objects...")
            crops, objects = self.get_object_detection_crops(image)
            result['object_detection_crops'] = crops
            result['detected_objects'] = objects
            print(f"Detected {len(objects)} objects with {len(crops)} crops")
        
        return result


# =============================
# Database Manager
# =============================
class DatabaseManager:
    """Manages ChromaDB operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.collection = None
    
    def initialize(self):
        """Initialize ChromaDB client and collection."""
        print(f"Initializing ChromaDB at {self.config.CHROMA_DB_DIR}...")
        
        os.makedirs(self.config.CHROMA_DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.config.CHROMA_DB_DIR)
        
        if self.config.COLLECTION_NAME in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(self.config.COLLECTION_NAME)
            print(f"✅ Loaded collection: {self.config.COLLECTION_NAME}")
        else:
            self.collection = self.client.create_collection(self.config.COLLECTION_NAME)
            print(f"✅ Created collection: {self.config.COLLECTION_NAME}")
        
        return self.collection
    
    def query_similar(self, query_embedding: list, top_k: int = 5) -> dict:
        """Query database for similar captions."""
        if self.collection is None:
            self.initialize()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        if self.collection is None:
            self.initialize()
        
        count = self.collection.count()
        return {
            'total_embeddings': count,
            'collection_name': self.config.COLLECTION_NAME
        }


# =============================
# RAG Retriever
# =============================
class RAGRetriever:
    """Handles retrieval-augmented generation."""
    
    def __init__(self, model_manager: ModelManager, 
                 db_manager: DatabaseManager,
                 preprocessor: ImagePreprocessor,
                 config: Config):
        self.model_manager = model_manager
        self.db_manager = db_manager
        self.preprocessor = preprocessor
        self.config = config
    
    def retrieve_captions_from_tensor(self, img_tensor: torch.Tensor, 
                                     top_k: int = None) -> str:
        """Retrieve top-k captions for an image tensor."""
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        img_emb = self.model_manager.encode_image(img_tensor)
        query_emb = img_emb.cpu().numpy()[0].tolist()
        
        results = self.db_manager.query_similar(query_emb, top_k)
        
        if results and "documents" in results and len(results["documents"]) > 0:
            return " ".join(results["documents"][0])
        return ""
    
    def retrieve_captions_from_image(self, image_path: str, 
                                    top_k: int = None,
                                    use_advanced: bool = True) -> Dict:
        """Retrieve captions from an image file with optional advanced preprocessing."""
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        if self.model_manager.clip_preprocess is None:
            raise ValueError("CLIP model not loaded")
        
        if use_advanced:
            return self._retrieve_with_advanced_preprocessing(image_path, top_k)
        else:
            return self._retrieve_basic(image_path, top_k)
    
    def _retrieve_basic(self, image_path: str, top_k: int) -> Dict:
        """Basic retrieval without advanced preprocessing."""
        img_tensor = self.preprocessor.preprocess_for_clip(
            image_path, 
            self.model_manager.clip_preprocess
        ).to(self.config.DEVICE)
        
        captions = self.retrieve_captions_from_tensor(img_tensor, top_k)
        
        return {
            'aggregated_caption': captions,
            'variants_processed': 1,
            'mode': 'basic'
        }
    
    def _retrieve_with_advanced_preprocessing(self, image_path: str, 
                                             top_k: int) -> Dict:
        """Retrieval with advanced preprocessing (crops + detection)."""
        try:
            variants = self.preprocessor.get_all_variants(image_path)
            
            all_captions = []
            variant_results = []
            
            # Process original
            img_tensor = self.preprocessor.preprocess_for_clip(
                variants['original'],
                self.model_manager.clip_preprocess
            ).to(self.config.DEVICE)
            
            caption = self.retrieve_captions_from_tensor(
                img_tensor, 
                self.config.CAPTIONS_PER_CROP
            )
            all_captions.append(caption)
            variant_results.append({'type': 'original', 'caption': caption})
            
            # Process segmentation crops
            for i, crop in enumerate(variants['segmentation_crops']):
                try:
                    img_tensor = self.preprocessor.preprocess_for_clip(
                        crop,
                        self.model_manager.clip_preprocess
                    ).to(self.config.DEVICE)
                    
                    caption = self.retrieve_captions_from_tensor(
                        img_tensor,
                        self.config.CAPTIONS_PER_CROP
                    )
                    all_captions.append(caption)
                    variant_results.append({'type': f'seg_{i+1}', 'caption': caption})
                except Exception as e:
                    print(f"⚠️ Error processing seg crop {i}: {e}")
            
            # Process object detection crops
            for i, crop in enumerate(variants['object_detection_crops']):
                try:
                    img_tensor = self.preprocessor.preprocess_for_clip(
                        crop,
                        self.model_manager.clip_preprocess
                    ).to(self.config.DEVICE)
                    
                    caption = self.retrieve_captions_from_tensor(
                        img_tensor,
                        self.config.CAPTIONS_PER_CROP
                    )
                    all_captions.append(caption)
                    variant_results.append({'type': f'obj_{i+1}', 'caption': caption})
                except Exception as e:
                    print(f"⚠️ Error processing obj crop {i}: {e}")
            
            # Aggregate captions
            if self.config.AGGREGATE_CAPTIONS:
                aggregated = self._aggregate_captions(all_captions)
            else:
                aggregated = " ".join(all_captions)
            
            return {
                'aggregated_caption': aggregated,
                'variants_processed': len(variant_results),
                'variant_results': variant_results,
                'detected_objects': variants.get('detected_objects', []),
                'mode': 'advanced'
            }
            
        except Exception as e:
            print(f"⚠️ Advanced preprocessing failed: {e}, falling back to basic")
            return self._retrieve_basic(image_path, top_k)
    
    def _aggregate_captions(self, captions: List[str]) -> str:
        """Aggregate multiple captions intelligently."""
        all_words = []
        for caption in captions:
            all_words.extend(caption.split())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        
        for word in all_words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in seen and len(word_clean) > 2:
                seen.add(word_clean)
                unique_words.append(word.strip())
        
        aggregated = ' '.join(unique_words)
        
        # Truncate if too long
        max_length = 500
        if len(aggregated) > max_length:
            aggregated = aggregated[:max_length].rsplit(' ', 1)[0] + '...'
        
        return aggregated
    
    def get_retrieval_stats(self, img_tensor: torch.Tensor, top_k: int = None) -> dict:
        """Get detailed retrieval statistics."""
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        img_emb = self.model_manager.encode_image(img_tensor)
        query_emb = img_emb.cpu().numpy()[0].tolist()
        
        results = self.db_manager.query_similar(query_emb, top_k)
        
        if not results or "documents" not in results:
            return {}
        
        captions = results["documents"][0] if results["documents"] else []
        distances = results.get("distances", [[]])[0]
        
        stats = {
            'num_retrieved': len(captions),
            'avg_caption_length': sum(len(cap.split()) for cap in captions) / len(captions) if captions else 0,
            'total_context_length': len(' '.join(captions).split())
        }
        
        if distances:
            stats.update({
                'min_distance': float(min(distances)),
                'max_distance': float(max(distances)),
                'avg_distance': float(sum(distances) / len(distances))
            })
        
        return stats
    
    def generate_caption_with_llm(self, image_path: str, top_k: int = None) -> Dict:
        """
        True Vision-Language RAG: 
        - Retrieve captions from ChromaDB (original + crops)
        - Pass original image + retrieved context to BLIP-2
        - Generate final caption with LLM
        """
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        # Check if BLIP-2 model is available
        if self.model_manager.blip2_model is None or self.model_manager.blip2_processor is None:
            return {
                'error': 'BLIP-2 model not loaded. Enable ENABLE_LLM_GENERATION in config.',
                'mode': 'llm_error'
            }
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            print("🔍 Retrieving captions from database...")
            
            # Get crops
            variants = self.preprocessor.get_all_variants(image)
            obj_crops = variants.get('object_detection_crops', [])
            seg_crops = variants.get('segmentation_crops', [])
            all_crops = obj_crops + seg_crops
            
            # Retrieve from original image
            img_tensor = self.preprocessor.preprocess_for_clip(
                image,
                self.model_manager.clip_preprocess
            ).to(self.config.DEVICE)
            
            original_captions = self.retrieve_captions_from_tensor(img_tensor, top_k)
            
            # Retrieve from crops (limit to top 3 crops)
            crop_captions = []
            for i, crop in enumerate(all_crops[:3]):
                try:
                    crop_tensor = self.preprocessor.preprocess_for_clip(
                        crop,
                        self.model_manager.clip_preprocess
                    ).to(self.config.DEVICE)
                    
                    crop_caption = self.retrieve_captions_from_tensor(crop_tensor, top_k=2)
                    crop_captions.append(crop_caption)
                except Exception as e:
                    print(f"⚠️ Error retrieving from crop {i}: {e}")
            
            # Prepare retrieved context
            retrieved_context = f"Original image: {original_captions}"
            if crop_captions:
                retrieved_context += f"\n\nImage regions: {' | '.join(crop_captions)}"
            
            # Create prompt for BLIP-2
            prompt = f"""Generate a natural, descriptive caption for this image.

Retrieved captions from similar images in the dataset:
{retrieved_context}

Based on what you see in the image and the retrieved context above, generate a concise caption:"""
            
            print("🤖 Generating caption with BLIP-2...")
            
            # Generate caption with BLIP-2 (vision + language)
            inputs = self.model_manager.blip2_processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(
                self.config.DEVICE,
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            with torch.no_grad():
                generated_ids = self.model_manager.blip2_model.generate(
                    **inputs,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    num_beams=self.config.NUM_BEAMS,
                    temperature=self.config.TEMPERATURE,
                    do_sample=False,
                    length_penalty=1.0
                )
            
            generated_caption = self.model_manager.blip2_processor.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            # Clean up memory
            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clean up prompt prefix if present
            if "caption:" in generated_caption.lower():
                generated_caption = generated_caption.split("caption:", 1)[-1].strip()
            
            print("✅ Caption generated successfully")
            
            return {
                'generated_caption': generated_caption,  # Final LLM-generated caption
                'raw_retrieved_context': retrieved_context,  # Original retrieved captions
                'original_captions': original_captions,
                'crop_captions': crop_captions,
                'num_crops': len(all_crops),
                'detected_objects': variants.get('detected_objects', []),
                'mode': 'llm_generation'
            }
            
        except Exception as e:
            print(f"❌ Error in LLM caption generation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'mode': 'llm_error'
            }


# =============================
# Main Pipeline Class
# =============================
class RAGImageCaptioning:
    """End-to-end pipeline for RAG image captioning."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Managers
        self.model_manager = ModelManager(config)
        self.db_manager = DatabaseManager(config)
        self.preprocessor = ImagePreprocessor(config)
        self.retriever = RAGRetriever(
            self.model_manager, 
            self.db_manager, 
            self.preprocessor,
            config
        )
    
    def setup(self):
        """Set up all components: models, database, etc."""
        print("🔧 Setting up RAG Image Captioning pipeline...")
        
        # Load models
        self.model_manager.load_clip_model()
        self.model_manager.load_blip2_model()
        
        # Initialize database
        self.db_manager.initialize()
        
        print("✅ Setup complete")
    
    def caption_image(self, image: Union[str, Image.Image], 
                     use_advanced: bool = True, 
                     top_k: int = None) -> Dict:
        """
        Generate caption for an image using RAG pipeline:
        - Retrieve relevant captions from database
        - Generate final caption using BLIP-2 (if enabled)
        """
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        print(f"🖼️ Captioning image (advanced={use_advanced}, top_k={top_k})...")
        
        # Run retrieval-augmented generation
        if self.config.ENABLE_LLM_GENERATION:
            result = self.retriever.generate_caption_with_llm(image, top_k)
        else:
            result = self.retriever.retrieve_captions_from_image(image, top_k, use_advanced)
        
        return result
    
    def add_image(self, image_path: str, caption: str):
        """Add an image and its caption to the database."""
        print(f"📥 Adding image to database: {image_path}")
        
        # TODO: Implement image addition (embedding + caption storage)
    
    def update_caption(self, image_id: str, new_caption: str):
        """Update the caption of an existing image."""
        print(f"✏️ Updating caption for image ID {image_id}")
        
        # TODO: Implement caption update
    
    def delete_image(self, image_id: str):
        """Remove an image and its caption from the database."""
        print(f"🗑️ Deleting image ID {image_id} from database")
        
        # TODO: Implement image deletion
    
    def get_image_stats(self) -> dict:
        """Get statistics about the image database."""
        return self.db_manager.get_stats()
    
    def close(self):
        """Clean up resources: close database, release models, etc."""
        print("🔒 Closing RAG Image Captioning pipeline...")
        
        # TODO: Implement cleanup (if necessary)
        
        print("✅ Closed")


# Alias for backwards compatibility
RAGImageCaptioningPipeline = RAGImageCaptioning

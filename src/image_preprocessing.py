"""
Advanced Image preprocessing utilities.
Handles image transformations, segmentation, object detection, and intelligent cropping.
"""
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from typing import Union, List, Tuple, Dict, Optional
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrForSegmentation
import cv2
from config import (
    OFA_RESOLUTION, OFA_MEAN, OFA_STD,
    ENABLE_SEGMENTATION, ENABLE_OBJECT_DETECTION,
    CROPS_PER_MODE, MIN_CROP_SIZE, CROP_PADDING,
    SEGMENTATION_MODEL, OBJECT_DETECTION_MODEL
)

class AdvancedImagePreprocessor:
    """Handles advanced image preprocessing with segmentation and object detection."""
    
    def __init__(self):
        self.clip_transform = None
        self.ofa_transform = None
        self.base_transform = None
        
        # Segmentation and Object Detection models
        self.segmentation_processor = None
        self.segmentation_model = None
        self.object_detection_processor = None
        self.object_detection_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize segmentation and object detection models."""
        try:
            if ENABLE_SEGMENTATION:
                print("🔧 Loading segmentation model...")
                self.segmentation_processor = DetrImageProcessor.from_pretrained(SEGMENTATION_MODEL)
                self.segmentation_model = DetrForSegmentation.from_pretrained(SEGMENTATION_MODEL)
                self.segmentation_model.to(self.device)
                self.segmentation_model.eval()
                print("✅ Segmentation model loaded")
            
            if ENABLE_OBJECT_DETECTION:
                print("🔧 Loading object detection model...")
                self.object_detection_processor = DetrImageProcessor.from_pretrained(OBJECT_DETECTION_MODEL)
                self.object_detection_model = DetrForObjectDetection.from_pretrained(OBJECT_DETECTION_MODEL)
                self.object_detection_model.to(self.device)
                self.object_detection_model.eval()
                print("✅ Object detection model loaded")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load advanced models: {e}")
            print("Falling back to basic preprocessing...")
            # Ensure models are None so other methods can check
            self.segmentation_processor = None
            self.segmentation_model = None
            self.object_detection_processor = None
            self.object_detection_model = None
    
    def get_clip_transform(self, clip_preprocess):
        """Get CLIP preprocessing transform."""
        self.clip_transform = clip_preprocess
        return self.clip_transform
    
    def get_ofa_transform(self):
        """Get OFA preprocessing transform."""
        if self.ofa_transform is None:
            self.ofa_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((OFA_RESOLUTION, OFA_RESOLUTION)),
                transforms.ToTensor(),
                transforms.Normalize(mean=OFA_MEAN, std=OFA_STD)
            ])
        return self.ofa_transform
    
    def get_base_transform(self, size: Tuple[int, int] = (224, 224)):
        """Get basic image preprocessing transform."""
        if self.base_transform is None:
            self.base_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return self.base_transform
    
    def preprocess_for_clip(self, image: Union[str, Image.Image], clip_preprocess) -> torch.Tensor:
        """Preprocess image for CLIP model."""
        if isinstance(image, str):
            image = Image.open(image)
        
        return clip_preprocess(image).unsqueeze(0)
    
    def preprocess_for_ofa(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocess image for OFA model."""
        if isinstance(image, str):
            image = Image.open(image)
        
        transform = self.get_ofa_transform()
        return transform(image).unsqueeze(0)
    
    def get_segmentation_crops(self, image: Image.Image) -> List[Image.Image]:
        """Generate crops based on image segmentation."""
        if not ENABLE_SEGMENTATION or self.segmentation_model is None:
            return []
        
        try:
            # Preprocess image for segmentation
            inputs = self.segmentation_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get segmentation results
            with torch.no_grad():
                outputs = self.segmentation_model(**inputs)
            
            # Process segmentation masks
            processed_sizes = inputs["pixel_values"].shape[-2:]
            result = self.segmentation_processor.post_process_panoptic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
            
            # Extract segments and create crops
            crops = []
            
            # Check if we have segmentation data
            if "segmentation" in result:
                segmentation = result["segmentation"]
            elif "segments_info" in result:
                # Handle panoptic segmentation format
                segmentation = result.get("segmentation", None)
                if segmentation is None:
                    print("⚠️ No segmentation map found in results")
                    return []
            else:
                print("⚠️ No segmentation information found in model output")
                return []
            
            # Get unique segment IDs (excluding background)
            unique_segments = torch.unique(segmentation)
            valid_segments = [seg for seg in unique_segments if seg > 0]  # Exclude background
            
            # Sort by segment size and take top segments
            segment_sizes = []
            for seg_id in valid_segments:
                mask = (segmentation == seg_id)
                size = mask.sum().item()
                if size > 100:  # Minimum segment size threshold
                    segment_sizes.append((seg_id, size))
            
            segment_sizes.sort(key=lambda x: x[1], reverse=True)
            top_segments = segment_sizes[:CROPS_PER_MODE]
            
            for seg_id, _ in top_segments:
                crop = self._create_crop_from_mask(image, segmentation == seg_id)
                if crop is not None:
                    crops.append(crop)
            
            return crops[:CROPS_PER_MODE]
            
        except Exception as e:
            print(f"⚠️ Error in segmentation cropping: {e}")
            return []
    
    def get_object_detection_crops(self, image: Image.Image) -> List[Image.Image]:
        """Generate crops based on object detection."""
        if not ENABLE_OBJECT_DETECTION or self.object_detection_model is None:
            return []
        
        try:
            # Preprocess image for object detection
            inputs = self.object_detection_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get object detection results
            with torch.no_grad():
                outputs = self.object_detection_model(**inputs)
            
            # Process detection results
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.object_detection_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]
            
            # Sort detections by confidence and take top ones
            scores = results["scores"]
            boxes = results["boxes"]
            
            # Get indices of top detections
            top_indices = scores.argsort(descending=True)[:CROPS_PER_MODE]
            
            crops = []
            for idx in top_indices:
                box = boxes[idx].cpu().numpy()
                crop = self._create_crop_from_bbox(image, box)
                if crop is not None:
                    crops.append(crop)
            
            return crops
            
        except Exception as e:
            print(f"⚠️ Error in object detection cropping: {e}")
            return []
    
    def _create_crop_from_mask(self, image: Image.Image, mask: torch.Tensor) -> Optional[Image.Image]:
        """Create a crop from a segmentation mask."""
        try:
            # Convert mask to numpy
            mask_np = mask.cpu().numpy().astype(np.uint8)
            
            # Find bounding box of the mask
            coords = np.column_stack(np.where(mask_np > 0))
            if len(coords) == 0:
                return None
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add padding
            x_min = max(0, x_min - CROP_PADDING)
            y_min = max(0, y_min - CROP_PADDING)
            x_max = min(image.width, x_max + CROP_PADDING)
            y_max = min(image.height, y_max + CROP_PADDING)
            
            # Check minimum size
            if (x_max - x_min) < MIN_CROP_SIZE or (y_max - y_min) < MIN_CROP_SIZE:
                return None
            
            # Create crop
            crop = image.crop((x_min, y_min, x_max, y_max))
            return crop
            
        except Exception as e:
            print(f"⚠️ Error creating crop from mask: {e}")
            return None
    
    def _create_crop_from_bbox(self, image: Image.Image, bbox: np.ndarray) -> Optional[Image.Image]:
        """Create a crop from a bounding box."""
        try:
            x_min, y_min, x_max, y_max = bbox
            
            # Add padding
            x_min = max(0, x_min - CROP_PADDING)
            y_min = max(0, y_min - CROP_PADDING)
            x_max = min(image.width, x_max + CROP_PADDING)
            y_max = min(image.height, y_max + CROP_PADDING)
            
            # Check minimum size
            if (x_max - x_min) < MIN_CROP_SIZE or (y_max - y_min) < MIN_CROP_SIZE:
                return None
            
            # Create crop
            crop = image.crop((x_min, y_min, x_max, y_max))
            return crop
            
        except Exception as e:
            print(f"⚠️ Error creating crop from bbox: {e}")
            return None
    
    def apply_initial_processing(self, image: Union[str, Image.Image]) -> Dict[str, Union[Image.Image, List[Image.Image]]]:
        """
        Apply advanced initial processing with segmentation and object detection.
        
        Args:
            image: Input image (path or PIL Image)
            
        Returns:
            Dictionary containing original image and crops from different modes
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        result = {
            'original': image,
            'segmentation_crops': [],
            'object_detection_crops': []
        }
        
        # Generate segmentation crops
        if ENABLE_SEGMENTATION: 
            print("🔍 Generating segmentation crops...")
            result['segmentation_crops'] = self.get_segmentation_crops(image)
            print(f"Generated {len(result['segmentation_crops'])} segmentation crops")
        
        # Generate object detection crops
        if ENABLE_OBJECT_DETECTION:
            print("🎯 Generating object detection crops...")
            result['object_detection_crops'] = self.get_object_detection_crops(image)
            print(f"Generated {len(result['object_detection_crops'])} object detection crops")
        
        return result
    
    def enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality before processing.
        Placeholder for quality enhancement techniques.
        """
        # TODO: Implement quality enhancement
        # Examples:
        # - Denoising
        # - Sharpening
        # - Brightness/contrast adjustment
        # - Histogram equalization
        
        return image
    
    def apply_domain_specific_preprocessing(self, image: Image.Image, domain: str = "general") -> Image.Image:
        """
        Apply domain-specific preprocessing based on image type.
        
        Args:
            image: Input PIL Image
            domain: Domain type ("medical", "satellite", "document", "general", etc.)
        """
        if domain == "medical":
            # TODO: Add medical image specific preprocessing
            pass
        elif domain == "satellite":
            # TODO: Add satellite image specific preprocessing  
            pass
        elif domain == "document":
            # TODO: Add document image specific preprocessing
            pass
        
        return image
    
    def get_all_image_variants(self, image: Union[str, Image.Image]) -> List[Image.Image]:
        """
        Get all image variants (original + all crops) for comprehensive RAG retrieval.
        
        Args:
            image: Input image (path or PIL Image)
            
        Returns:
            List of all image variants
        """
        processed_result = self.apply_initial_processing(image)
        
        variants = [processed_result['original']]
        variants.extend(processed_result['segmentation_crops'])
        variants.extend(processed_result['object_detection_crops'])
        
        return variants
    
    def batch_preprocess_images(self, image_paths: List[str], transform_type: str = "clip") -> torch.Tensor:
        """
        Batch preprocess multiple images.
        
        Args:
            image_paths: List of image file paths
            transform_type: Type of preprocessing ("clip", "ofa", "base")
        """
        processed_images = []
        
        for img_path in image_paths:
            try:
                # Get processed result (now returns dict)
                processed_result = self.apply_initial_processing(img_path)
                image = processed_result['original']  # Use original for now
                
                # Apply specific transform
                if transform_type == "clip":
                    # Note: Need clip_preprocess from model
                    processed_img = self.get_base_transform()(image)
                elif transform_type == "ofa":
                    processed_img = self.get_ofa_transform()(image)
                else:
                    processed_img = self.get_base_transform()(image)
                
                processed_images.append(processed_img)
                
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
                continue
        
        if processed_images:
            return torch.stack(processed_images)
        else:
            return torch.empty(0)
    
    def visualize_crops(self, image: Union[str, Image.Image], save_path: Optional[str] = None):
        """
        Visualize the original image and generated crops.
        
        Args:
            image: Input image (path or PIL Image)
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            processed_result = self.apply_initial_processing(image)
            
            # Count total images
            total_crops = len(processed_result['segmentation_crops']) + len(processed_result['object_detection_crops'])
            total_images = 1 + total_crops  # original + crops
            
            if total_images == 1:
                print("No crops generated")
                return
            
            # Create subplot grid
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
            axes[0].imshow(processed_result['original'])
            axes[0].set_title('Original', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            idx = 1
            
            # Plot segmentation crops
            for i, crop in enumerate(processed_result['segmentation_crops']):
                if idx < len(axes):
                    axes[idx].imshow(crop)
                    axes[idx].set_title(f'Segmentation {i+1}', fontsize=10)
                    axes[idx].axis('off')
                    idx += 1
            
            # Plot object detection crops
            for i, crop in enumerate(processed_result['object_detection_crops']):
                if idx < len(axes):
                    axes[idx].imshow(crop)
                    axes[idx].set_title(f'Object Detection {i+1}', fontsize=10)
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
            
        except ImportError:
            print("⚠️ Matplotlib not available for visualization")
        except Exception as e:
            print(f"⚠️ Error in visualization: {e}")

# Global preprocessor instance
image_preprocessor = AdvancedImagePreprocessor()

def preprocess_image_for_clip(image_path: str, clip_preprocess) -> torch.Tensor:
    """Convenience function for CLIP preprocessing."""
    return image_preprocessor.preprocess_for_clip(image_path, clip_preprocess)

def preprocess_image_for_ofa(image_path: str) -> torch.Tensor:
    """Convenience function for OFA preprocessing."""  
    return image_preprocessor.preprocess_for_ofa(image_path)

def apply_advanced_preprocessing(image_path: str) -> Dict[str, Union[Image.Image, List[Image.Image]]]:
    """Convenience function for advanced preprocessing with segmentation and object detection."""
    return image_preprocessor.apply_initial_processing(image_path)

def get_all_image_variants(image_path: str) -> List[Image.Image]:
    """Convenience function to get all image variants for RAG."""
    return image_preprocessor.get_all_image_variants(image_path)

def visualize_image_crops(image_path: str, save_path: Optional[str] = None):
    """Convenience function to visualize crops."""
    return image_preprocessor.visualize_crops(image_path, save_path)
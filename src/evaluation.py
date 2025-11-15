"""
Evaluation and testing utilities.
Handles model evaluation, caption generation, and performance metrics.
"""
import os
import random
from PIL import Image
from typing import List, Dict, Union
import torch

from config import COCO_IMG_DIR, DEFAULT_TOP_K
from models import model_manager
from rag_retrieval import rag_retriever
from image_preprocessing import image_preprocessor

class EvaluationManager:
    """Manages evaluation and testing of the RAG image captioning system."""
    
    def __init__(self):
        self.model_manager = model_manager
        self.rag_retriever = rag_retriever
        self.image_preprocessor = image_preprocessor
        
    def generate_caption_with_ofa(self, img_tensor: torch.Tensor, prompt: str = None) -> str:
        """
        Generate caption using OFA model.
        
        Args:
            img_tensor: Preprocessed image tensor
            prompt: Custom prompt (if None, uses default)
            
        Returns:
            Generated caption string
        """
        if self.model_manager.ofa_model is None or self.model_manager.tokenizer is None:
            print("❌ OFA model or tokenizer not loaded")
            return "Model not available"
        
        try:
            # Use default prompt if none provided
            if prompt is None:
                prompt = "what does the image describe?"
            
            # Tokenize prompt
            inputs = self.model_manager.tokenizer(prompt, return_tensors="pt").to(self.model_manager.device)
            
            # Generate caption
            with torch.no_grad():
                generated = self.model_manager.ofa_model.generate(
                    inputs.input_ids,
                    patch_images=img_tensor,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    max_length=50
                )
            
            # Decode generated caption
            caption = self.model_manager.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return caption
            
        except Exception as e:
            print(f"❌ Error generating caption: {e}")
            return "Generation failed"
    
    def generate_rag_enhanced_caption(self, image_path: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, str]:
        """
        Generate caption using RAG-enhanced context.
        
        Args:
            image_path: Path to image file
            top_k: Number of context captions to retrieve
            
        Returns:
            Dictionary with context, prompt, and generated caption
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        try:
            # Preprocess image
            if self.model_manager.clip_preprocess is None:
                print("❌ CLIP preprocessor not available")
                return {"error": "CLIP preprocessor not loaded"}
            
            # Apply initial preprocessing and get the original image
            processed_result = self.image_preprocessor.apply_initial_processing(image_path)
            original_image = processed_result['original']
            
            # Preprocess for CLIP (for RAG retrieval)
            clip_tensor = self.image_preprocessor.preprocess_for_clip(
                original_image, self.model_manager.clip_preprocess
            ).to(self.model_manager.device)
            
            # Preprocess for OFA (for generation)
            ofa_tensor = self.image_preprocessor.preprocess_for_ofa(original_image).to(self.model_manager.device)
            
            # Retrieve context captions
            context_captions = self.rag_retriever.retrieve_topk_captions_from_image_tensor(clip_tensor, top_k)
            
            # Create enhanced prompt
            rag_prompt = f"{context_captions} what does the image describe?"
            
            # Generate caption with context
            generated_caption = self.generate_caption_with_ofa(ofa_tensor, rag_prompt)
            
            return {
                "context_captions": context_captions,
                "rag_prompt": rag_prompt,
                "generated_caption": generated_caption
            }
            
        except Exception as e:
            print(f"❌ Error in RAG-enhanced generation: {e}")
            return {"error": str(e)}
    
    def evaluate_on_random_images(self, num_images: int = 10, img_dir: str = None) -> List[Dict]:
        """
        Evaluate system on random images from dataset.
        
        Args:
            num_images: Number of random images to evaluate
            img_dir: Directory containing images (defaults to COCO_IMG_DIR)
            
        Returns:
            List of evaluation results
        """
        if img_dir is None:
            img_dir = COCO_IMG_DIR
        
        if not os.path.exists(img_dir):
            print(f"❌ Image directory not found: {img_dir}")
            return []
        
        # Get random images
        all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(all_images) < num_images:
            print(f"⚠️ Only {len(all_images)} images available, using all")
            num_images = len(all_images)
        
        random_images = random.sample(all_images, num_images)
        results = []
        
        print(f"🔍 Evaluating on {num_images} random images...")
        
        for i, img_name in enumerate(random_images, 1):
            img_path = os.path.join(img_dir, img_name)
            print(f"Processing image {i}/{num_images}: {img_name}")
            
            # Generate caption
            result = self.generate_rag_enhanced_caption(img_path)
            result["image_name"] = img_name
            result["image_path"] = img_path
            
            # Add retrieval statistics
            if "error" not in result:
                try:
                    if self.model_manager.clip_preprocess:
                        clip_tensor = self.image_preprocessor.preprocess_for_clip(
                            img_path, self.model_manager.clip_preprocess
                        ).to(self.model_manager.device)
                        
                        # Only add stats if method exists
                        if hasattr(self.rag_retriever, 'get_retrieval_statistics'):
                            stats = self.rag_retriever.get_retrieval_statistics(clip_tensor)
                            result["retrieval_stats"] = stats
                except Exception as e:
                    print(f"⚠️ Could not get retrieval stats: {e}")
            
            results.append(result)
        
        return results
    
    def display_evaluation_results(self, results: List[Dict], show_images: bool = False):
        """
        Display evaluation results in a formatted way.
        
        Args:
            results: List of evaluation results
            show_images: Whether to display images (requires IPython)
        """
        for i, result in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"Image {i}: {result.get('image_name', 'Unknown')}")
            print(f"{'='*60}")
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Show image if requested
            if show_images:
                try:
                    from IPython.display import display
                    image = Image.open(result['image_path'])
                    display(image)
                except ImportError:
                    print("📷 (Image display requires IPython)")
                except Exception as e:
                    print(f"📷 Could not display image: {e}")
            
            # Show context captions
            print(f"\n🔹 Retrieved Context Captions:")
            context = result.get('context_captions', '')
            for j, caption in enumerate(context.split('. '), 1):
                if caption.strip():
                    print(f"  {j}. {caption.strip()}.")
            
            # Show generated caption
            print(f"\n🖼️ Generated Caption:")
            print(f"  {result.get('generated_caption', 'No caption generated')}")
            
            # Show retrieval statistics
            if 'retrieval_stats' in result:
                stats = result['retrieval_stats']
                print(f"\n📊 Retrieval Statistics:")
                print(f"  • Retrieved captions: {stats.get('num_retrieved', 0)}")
                print(f"  • Avg caption length: {stats.get('avg_caption_length', 0):.1f} words")
                print(f"  • Total context length: {stats.get('total_context_length', 0)} words")
                
                if 'avg_distance' in stats:
                    print(f"  • Avg similarity distance: {stats['avg_distance']:.4f}")
    
    def compare_with_and_without_rag(self, image_path: str) -> Dict:
        """
        Compare caption generation with and without RAG context.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with both results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        try:
            # Generate without RAG (simple prompt)
            ofa_tensor = self.image_preprocessor.preprocess_for_ofa(image_path).to(self.model_manager.device)
            simple_caption = self.generate_caption_with_ofa(ofa_tensor, "what does the image describe?")
            
            # Generate with RAG
            rag_result = self.generate_rag_enhanced_caption(image_path)
            
            return {
                "image_path": image_path,
                "simple_caption": simple_caption,
                "rag_enhanced": rag_result,
                "context_used": rag_result.get('context_captions', '')
            }
            
        except Exception as e:
            return {"error": str(e)}

# Global evaluation manager instance
evaluation_manager = EvaluationManager()

def run_evaluation(num_images: int = 10) -> List[Dict]:
    """Convenience function to run evaluation."""
    return evaluation_manager.evaluate_on_random_images(num_images)

def test_single_image(image_path: str) -> Dict:
    """Convenience function to test single image."""
    return evaluation_manager.generate_rag_enhanced_caption(image_path)
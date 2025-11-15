"""
RAG (Retrieval-Augmented Generation) utilities.
Handles caption retrieval and context generation.
"""
import torch
from typing import List, Union, Dict
from models import model_manager
from database import db_manager
from image_preprocessing import image_preprocessor
from config import DEFAULT_TOP_K, CAPTIONS_PER_CROP, AGGREGATE_CAPTIONS

class RAGRetriever:
    """Handles retrieval operations for RAG-based image captioning."""
    
    def __init__(self):
        self.db_manager = db_manager
        self.model_manager = model_manager
        
    def retrieve_topk_captions_from_image_tensor(self, img_tensor: torch.Tensor, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Retrieve top-k similar captions for a given image tensor.
        
        Args:
            img_tensor: Preprocessed image tensor
            top_k: Number of top captions to retrieve
            
        Returns:
            Concatenated string of retrieved captions
        """
        if self.model_manager.clip_model is None:
            raise ValueError("CLIP model not loaded. Initialize models first.")
            
        # Encode image to get embedding
        img_emb = self.model_manager.encode_image_with_clip(img_tensor)
        query_emb = img_emb.cpu().numpy()[0].tolist()
        
        # Query database
        results = self.db_manager.query_similar_captions(query_emb, top_k)
        
        # Join retrieved captions
        return " ".join(results["documents"][0])
    
    def retrieve_captions_from_multiple_images(self, image_variants: List[torch.Tensor], 
                                             captions_per_image: int = CAPTIONS_PER_CROP) -> Dict[str, Union[str, List[str]]]:
        """
        Retrieve captions from multiple image variants (original + crops).
        
        Args:
            image_variants: List of preprocessed image tensors
            captions_per_image: Number of captions to retrieve per image variant
            
        Returns:
            Dictionary with aggregated captions and individual results
        """
        all_captions = []
        individual_results = []
        
        for i, img_tensor in enumerate(image_variants):
            try:
                captions = self.retrieve_topk_captions_from_image_tensor(img_tensor, captions_per_image)
                all_captions.extend(captions.split(' '))
                individual_results.append({
                    'image_index': i,
                    'captions': captions,
                    'type': 'original' if i == 0 else f'crop_{i}'
                })
            except Exception as e:
                print(f"⚠️ Error retrieving captions for image variant {i}: {e}")
                continue
        
        # Aggregate and deduplicate captions
        if AGGREGATE_CAPTIONS:
            aggregated = self._aggregate_captions(all_captions)
        else:
            aggregated = ' '.join(all_captions)
        
        return {
            'aggregated_captions': aggregated,
            'individual_results': individual_results,
            'total_variants_processed': len(individual_results)
        }
    
    def retrieve_topk_captions_from_image_path(self, image_path: str, top_k: int = DEFAULT_TOP_K, 
                                             use_advanced_preprocessing: bool = True) -> str:
        """
        Retrieve top-k similar captions for an image file with optional advanced preprocessing.
        
        Args:
            image_path: Path to image file
            top_k: Number of top captions to retrieve (used when advanced preprocessing is disabled)
            use_advanced_preprocessing: Whether to use segmentation and object detection
            
        Returns:
            Concatenated string of retrieved captions
        """
        if self.model_manager.clip_preprocess is None:
            raise ValueError("CLIP preprocessor not loaded. Initialize models first.")
        
        if use_advanced_preprocessing:
            # Use advanced preprocessing with multiple image variants
            try:
                image_variants = image_preprocessor.get_all_image_variants(image_path)
                
                # Convert all variants to tensors
                variant_tensors = []
                for variant in image_variants:
                    tensor = image_preprocessor.preprocess_for_clip(
                        variant, self.model_manager.clip_preprocess
                    ).to(self.model_manager.device)
                    variant_tensors.append(tensor)
                
                # Retrieve captions from all variants
                result = self.retrieve_captions_from_multiple_images(variant_tensors)
                return result['aggregated_captions']
                
            except Exception as e:
                print(f"⚠️ Advanced preprocessing failed, falling back to simple mode: {e}")
                use_advanced_preprocessing = False
        
        if not use_advanced_preprocessing:
            # Fallback to simple preprocessing
            img_tensor = image_preprocessor.preprocess_for_clip(
                image_path, 
                self.model_manager.clip_preprocess
            ).to(self.model_manager.device)
            
            return self.retrieve_topk_captions_from_image_tensor(img_tensor, top_k)
    
    def retrieve_structured_captions(self, img_tensor: torch.Tensor, top_k: int = DEFAULT_TOP_K) -> dict:
        """
        Retrieve captions with additional metadata.
        
        Args:
            img_tensor: Preprocessed image tensor
            top_k: Number of top captions to retrieve
            
        Returns:
            Dictionary with captions, ids, and distances
        """
        if self.model_manager.clip_model is None:
            raise ValueError("CLIP model not loaded. Initialize models first.")
            
        # Encode image
        img_emb = self.model_manager.encode_image_with_clip(img_tensor)
        query_emb = img_emb.cpu().numpy()[0].tolist()
        
        # Query database
        results = self.db_manager.query_similar_captions(query_emb, top_k)
        
        return {
            'captions': results["documents"][0],
            'ids': results["ids"][0],
            'distances': results.get("distances", [None])[0] if "distances" in results else None
        }
    
    def create_context_prompt(self, img_tensor: torch.Tensor, top_k: int = DEFAULT_TOP_K, 
                            prompt_template: str = None) -> str:
        """
        Create a context-enriched prompt for caption generation.
        
        Args:
            img_tensor: Preprocessed image tensor
            top_k: Number of top captions to retrieve
            prompt_template: Custom prompt template
            
        Returns:
            Context-enriched prompt string
        """
        # Retrieve context captions
        context_text = self.retrieve_topk_captions_from_image_tensor(img_tensor, top_k)
        
        # Use custom template or default
        if prompt_template is None:
            prompt_template = "{context} what does the image describe?"
            
        return prompt_template.format(context=context_text)
    
    def batch_retrieve_captions(self, img_tensors: List[torch.Tensor], top_k: int = DEFAULT_TOP_K) -> List[str]:
        """
        Batch retrieve captions for multiple images.
        
        Args:
            img_tensors: List of preprocessed image tensors
            top_k: Number of top captions to retrieve per image
            
        Returns:
            List of concatenated caption strings
        """
        results = []
        for img_tensor in img_tensors:
            try:
                captions = self.retrieve_topk_captions_from_image_tensor(img_tensor, top_k)
                results.append(captions)
            except Exception as e:
                print(f"⚠️ Error retrieving captions: {e}")
                results.append("")
        
        return results
    
    def get_retrieval_statistics(self, img_tensor: torch.Tensor, top_k: int = DEFAULT_TOP_K) -> dict:
        """
        Get detailed statistics about retrieval results.
        
        Args:
            img_tensor: Preprocessed image tensor
            top_k: Number of top captions to retrieve
            
        Returns:
            Dictionary with retrieval statistics
        """
        structured_results = self.retrieve_structured_captions(img_tensor, top_k)
        
        captions = structured_results['captions']
        distances = structured_results['distances']
        
        stats = {
            'num_retrieved': len(captions),
            'avg_caption_length': sum(len(cap.split()) for cap in captions) / len(captions) if captions else 0,
            'total_context_length': len(' '.join(captions).split()),
        }
        
        if distances:
            stats.update({
                'min_distance': min(distances),
                'max_distance': max(distances),
                'avg_distance': sum(distances) / len(distances)
            })
        
        return stats
    
    def _aggregate_captions(self, caption_words: List[str]) -> str:
        """
        Intelligently aggregate captions by removing duplicates and organizing content.
        
        Args:
            caption_words: List of caption words/phrases
            
        Returns:
            Aggregated caption string
        """
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        
        for word in caption_words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in seen and len(word_clean) > 2:
                seen.add(word_clean)
                unique_words.append(word.strip())
        
        # Join and limit length
        aggregated = ' '.join(unique_words)
        
        # Truncate if too long (keep reasonable context length)
        max_length = 500  # characters
        if len(aggregated) > max_length:
            aggregated = aggregated[:max_length].rsplit(' ', 1)[0] + '...'
        
        return aggregated

# Global RAG retriever instance
rag_retriever = RAGRetriever()

def retrieve_captions_for_image(image_path: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Convenience function for caption retrieval."""
    return rag_retriever.retrieve_topk_captions_from_image_path(image_path, top_k)

def create_rag_prompt(image_path: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Convenience function for creating RAG prompts."""
    if model_manager.clip_preprocess is None:
        raise ValueError("Models not initialized. Call initialize_models() first.")
        
    img_tensor = image_preprocessor.preprocess_for_clip(
        image_path, 
        model_manager.clip_preprocess
    ).to(model_manager.device)
    
    return rag_retriever.create_context_prompt(img_tensor, top_k)
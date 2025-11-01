"""
Training utilities and functions.
Handles model fine-tuning, data loading, and training loops.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from tqdm import tqdm
from typing import Tuple, List

from config import (
    COCO_IMG_DIR, 
    COCO_ANN_FILE,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_TRAINING_STEPS,
    LOG_INTERVAL,
    FINE_TUNED_MODEL_DIR
)
from models import model_manager
from rag_retrieval import rag_retriever
from image_preprocessing import image_preprocessor

class TrainingManager:
    """Manages the training process for OFA model fine-tuning."""
    
    def __init__(self):
        self.model_manager = model_manager
        self.rag_retriever = rag_retriever
        self.optimizer = None
        self.dataloader = None
        
    def create_collate_fn(self):
        """Create collate function for dynamic padding in DataLoader."""
        def collate_fn(batch):
            imgs, captions = zip(*batch)
            imgs = torch.stack(imgs, dim=0)

            input_ids_list, attention_masks_list, labels_list = [], [], []

            for img_tensor, caption in zip(imgs, captions):
                try:
                    # Retrieve top-k captions using RAG
                    context_text = self.rag_retriever.retrieve_topk_captions_from_image_tensor(
                        img_tensor.unsqueeze(0)
                    )
                    prompt = f"{context_text} what does the image describe?"

                    # Tokenize prompt dynamically
                    if self.model_manager.tokenizer is None:
                        print("⚠️ Tokenizer not loaded, skipping batch")
                        continue
                        
                    inputs = self.model_manager.tokenizer(prompt, return_tensors="pt", padding=True)
                    input_ids_list.append(inputs.input_ids[0])
                    attention_masks_list.append(inputs.attention_mask[0])

                    # Tokenize target caption
                    label_ids = self.model_manager.tokenizer(caption, return_tensors="pt", padding=True).input_ids[0]
                    label_ids[label_ids == self.model_manager.tokenizer.pad_token_id] = -100
                    labels_list.append(label_ids)
                    
                except Exception as e:
                    print(f"⚠️ Error in collate_fn: {e}")
                    continue

            if not input_ids_list:
                return None, None, None, None

            # Pad all sequences in the batch to the max length in the batch
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, 
                batch_first=True, 
                padding_value=self.model_manager.tokenizer.pad_token_id
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_masks_list, 
                batch_first=True, 
                padding_value=0
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, 
                batch_first=True, 
                padding_value=-100
            )

            return imgs, input_ids, attention_mask, labels
            
        return collate_fn
    
    def setup_dataloader(self):
        """Setup DataLoader with COCO dataset."""
        print("Setting up DataLoader...")
        
        # Get OFA transform
        patch_transform = self.model_manager.get_patch_transform()
        
        # Create dataset
        dataset = CocoCaptions(
            root=COCO_IMG_DIR, 
            annFile=COCO_ANN_FILE, 
            transform=patch_transform
        )
        
        # Create dataloader with custom collate function
        collate_fn = self.create_collate_fn()
        self.dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        print(f"✅ DataLoader created with batch size {BATCH_SIZE}")
        return self.dataloader
    
    def setup_optimizer(self):
        """Setup optimizer for training."""
        if self.model_manager.ofa_model is None:
            print("❌ OFA model not loaded. Cannot setup optimizer.")
            return None
            
        self.optimizer = torch.optim.AdamW(
            self.model_manager.ofa_model.parameters(), 
            lr=LEARNING_RATE
        )
        print(f"✅ Optimizer setup with learning rate {LEARNING_RATE}")
        return self.optimizer
    
    def train_step(self, img_tensor, input_ids, attention_mask, labels):
        """Execute single training step."""
        if self.model_manager.ofa_model is None:
            raise ValueError("OFA model not loaded")
            
        # Move to device
        img_tensor = img_tensor.to(self.model_manager.device)
        input_ids = input_ids.to(self.model_manager.device)
        attention_mask = attention_mask.to(self.model_manager.device)
        labels = labels.to(self.model_manager.device)
        
        # Forward pass
        outputs = self.model_manager.ofa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_images=img_tensor,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def fine_tune_model(self):
        """Main fine-tuning loop."""
        print("🚀 Starting OFA fine-tuning with RAG context...")
        
        # Setup components
        if self.dataloader is None:
            self.setup_dataloader()
        if self.optimizer is None:
            self.setup_optimizer()
            
        if self.model_manager.ofa_model is None:
            print("❌ OFA model not available. Skipping training.")
            return False
        
        # Training loop
        self.model_manager.ofa_model.train()
        total_loss = 0
        step = 0
        
        for batch_data in self.dataloader:
            try:
                img_tensor, input_ids, attention_mask, labels = batch_data
                
                # Skip empty batches
                if img_tensor is None:
                    continue
                
                # Training step
                loss = self.train_step(img_tensor, input_ids, attention_mask, labels)
                total_loss += loss
                
                # Logging
                if step % LOG_INTERVAL == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"Step {step} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")
                
                step += 1
                
                # Break for demo purposes
                if step >= MAX_TRAINING_STEPS:
                    break
                    
            except Exception as e:
                print(f"⚠️ Skipped batch due to: {e}")
                continue
        
        print(f"✅ Training completed. Total steps: {step}")
        return True
    
    def save_model(self, save_dir: str = FINE_TUNED_MODEL_DIR):
        """Save the fine-tuned model."""
        if self.model_manager.ofa_model is None or self.model_manager.tokenizer is None:
            print("❌ Models not available for saving")
            return False
            
        try:
            os.makedirs(save_dir, exist_ok=True)
            self.model_manager.ofa_model.save_pretrained(save_dir)
            self.model_manager.tokenizer.save_pretrained(save_dir)
            print(f"✅ Fine-tuned model saved at {save_dir}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def validate_setup(self):
        """Validate that all components are ready for training."""
        checks = {
            "COCO dataset": os.path.exists(COCO_IMG_DIR) and os.path.exists(COCO_ANN_FILE),
            "CLIP model": self.model_manager.clip_model is not None,
            "OFA model": self.model_manager.ofa_model is not None,
            "Tokenizer": self.model_manager.tokenizer is not None,
            "ChromaDB": self.rag_retriever.db_manager.collection is not None
        }
        
        print("Training setup validation:")
        all_ready = True
        for component, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}")
            if not status:
                all_ready = False
        
        return all_ready

# Global training manager instance
training_manager = TrainingManager()

def run_training():
    """Convenience function to run the complete training pipeline."""
    print("Starting training pipeline...")
    
    # Validate setup
    if not training_manager.validate_setup():
        print("❌ Training setup validation failed")
        return False
    
    # Run training
    success = training_manager.fine_tune_model()
    
    if success:
        # Save model
        training_manager.save_model()
    
    return success
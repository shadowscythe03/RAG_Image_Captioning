"""
Model initialization and loading utilities.
Handles CLIP model, OFA model, and tokenizer setup.
"""
import torch
import open_clip
from torchvision import transforms
from config import (
    CLIP_MODEL_NAME, 
    CLIP_PRETRAINED, 
    OFA_CHECKPOINT_DIR,
    OFA_RESOLUTION,
    OFA_MEAN,
    OFA_STD,
    get_device
)

class ModelManager:
    """Manages all models used in the RAG image captioning system."""
    
    def __init__(self):
        self.device = get_device()
        self.clip_model = None
        self.clip_preprocess = None
        self.ofa_model = None
        self.tokenizer = None
        self.patch_transform = None
        
    def load_clip_model(self):
        """Load and initialize CLIP model."""
        print(f"Loading CLIP model on {self.device}...")
        
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        print(f"✅ CLIP model loaded successfully on {self.device}")
        return self.clip_model, self.clip_preprocess
    
    def load_ofa_model(self):
        """Load and initialize OFA model and tokenizer."""
        try:
            # Note: Commented out for now as OFA imports were commented in notebook
            # from transformers import OFATokenizer, OFAModel
            
            # self.tokenizer = OFATokenizer.from_pretrained(OFA_CHECKPOINT_DIR)
            # self.ofa_model = OFAModel.from_pretrained(OFA_CHECKPOINT_DIR).to(self.device)
            # self.ofa_model.train()
            
            print("⚠️ OFA model loading is commented out. Enable imports when ready.")
            return None, None
            
        except ImportError as e:
            print(f"❌ Failed to load OFA model: {e}")
            return None, None
    
    def get_patch_transform(self):
        """Get image preprocessing transform for OFA."""
        if self.patch_transform is None:
            self.patch_transform = transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((OFA_RESOLUTION, OFA_RESOLUTION)),
                transforms.ToTensor(),
                transforms.Normalize(mean=OFA_MEAN, std=OFA_STD)
            ])
        return self.patch_transform
    
    def encode_image_with_clip(self, image_tensor):
        """Encode image using CLIP model."""
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded. Call load_clip_model() first.")
            
        with torch.no_grad():
            img_emb = self.clip_model.encode_image(image_tensor)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            return img_emb
    
    def get_models(self):
        """Get all loaded models."""
        return {
            'clip_model': self.clip_model,
            'clip_preprocess': self.clip_preprocess,
            'ofa_model': self.ofa_model,
            'tokenizer': self.tokenizer,
            'patch_transform': self.patch_transform,
            'device': self.device
        }

# Global model manager instance
model_manager = ModelManager()

def initialize_models():
    """Initialize all models needed for the system."""
    print("Initializing models...")
    
    # Load CLIP model
    model_manager.load_clip_model()
    
    # Load OFA model (currently commented out)
    model_manager.load_ofa_model()
    
    # Initialize transforms
    model_manager.get_patch_transform()
    
    print("✅ Model initialization complete")
    return model_manager
"""
Database operations for ChromaDB.
Handles embedding storage, retrieval, and database management.
"""
import os
import json
import numpy as np
import chromadb
from PIL import Image
from tqdm import tqdm
from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    COCO_IMG_DIR,
    COCO_ANN_FILE,
    EMBEDDING_BATCH_SIZE
)
from models import model_manager

class ChromaDBManager:
    """Manages ChromaDB operations for image embeddings and captions."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        
    def initialize_database(self):
        """Initialize ChromaDB client and collection."""
        print(f"Initializing ChromaDB at {CHROMA_DB_DIR}...")
        
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Create or get collection
        if COLLECTION_NAME in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            print(f"✅ Loaded existing collection: {COLLECTION_NAME}")
        else:
            self.collection = self.client.create_collection(COLLECTION_NAME)
            print(f"✅ Created new collection: {COLLECTION_NAME}")
            
        return self.collection
    
    def get_collection(self):
        """Get the current collection."""
        if self.collection is None:
            self.initialize_database()
        return self.collection
    
    def build_database_from_coco(self):
        """Build the ChromaDB database from COCO dataset."""
        if not os.path.exists(COCO_ANN_FILE):
            print(f"❌ COCO annotations file not found: {COCO_ANN_FILE}")
            return False
            
        # Ensure models are loaded
        if model_manager.clip_model is None:
            model_manager.load_clip_model()
            
        # Load COCO annotations
        print("Loading COCO annotations...")
        with open(COCO_ANN_FILE, "r") as f:
            coco_data = json.load(f)
            
        annotations = coco_data["annotations"]
        img_id_to_path = {
            int(fn.split('.')[0]): os.path.join(COCO_IMG_DIR, fn)
            for fn in os.listdir(COCO_IMG_DIR) if fn.endswith(".jpg")
        }
        
        # Initialize collection
        collection = self.get_collection()
        
        # Batch processing
        batch_embeds = []
        batch_docs = []
        batch_ids = []
        
        print("🔹 Encoding images and indexing in Chroma...")
        for ann in tqdm(annotations):
            img_id = ann["image_id"]
            ann_id = ann["id"]
            caption = ann["caption"]
            img_path = img_id_to_path.get(img_id)
            
            if not img_path or not os.path.exists(img_path):
                continue
                
            try:
                # Encode image
                image = model_manager.clip_preprocess(Image.open(img_path)).unsqueeze(0).to(model_manager.device)
                img_emb = model_manager.encode_image_with_clip(image)
                emb = img_emb.cpu().numpy()[0]
                
                batch_embeds.append(emb)
                batch_docs.append(caption)
                batch_ids.append(f"{img_id}_{ann_id}")
                
                # Add batch to collection when full
                if len(batch_embeds) == EMBEDDING_BATCH_SIZE:
                    collection.add(
                        embeddings=np.array(batch_embeds).tolist(),
                        documents=batch_docs,
                        ids=batch_ids
                    )
                    batch_embeds, batch_docs, batch_ids = [], [], []
                    
            except Exception as e:
                print(f"⚠️ Error processing image {img_path}: {e}")
                continue
        
        # Add remaining embeddings
        if batch_embeds:
            collection.add(
                embeddings=np.array(batch_embeds).tolist(),
                documents=batch_docs,
                ids=batch_ids
            )
        
        print("✅ ChromaDB database built successfully")
        return True
    
    def query_similar_captions(self, query_embedding, top_k=5):
        """Query the database for similar captions based on embedding."""
        collection = self.get_collection()
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k
        )
        return results
    
    def get_database_stats(self):
        """Get statistics about the current database."""
        collection = self.get_collection()
        count = collection.count()
        print(f"📊 Database contains {count} embeddings")
        return count

# Global database manager instance
db_manager = ChromaDBManager()
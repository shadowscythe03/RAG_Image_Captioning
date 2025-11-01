"""
Test script for advanced image preprocessing with segmentation and object detection.
Demonstrates the new RAG capabilities with multiple image variants.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config import validate_paths, create_output_dirs
from src.models import initialize_models
from src.database import db_manager
from src.image_preprocessing import image_preprocessor, visualize_image_crops
from src.rag_retrieval import rag_retriever
from src.evaluation import evaluation_manager

def test_advanced_preprocessing(image_path: str):
    """Test the advanced preprocessing pipeline on a single image."""
    print("🧪 Testing Advanced Image Preprocessing Pipeline")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Step 1: Initialize system
    print("1. Initializing system...")
    try:
        create_output_dirs()
        model_manager = initialize_models()
        db_manager.initialize_database()
        print("✅ System initialized")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Step 2: Test advanced preprocessing
    print(f"\n2. Testing advanced preprocessing on: {os.path.basename(image_path)}")
    try:
        processed_result = image_preprocessor.apply_initial_processing(image_path)
        
        print(f"✅ Generated {len(processed_result['segmentation_crops'])} segmentation crops")
        print(f"✅ Generated {len(processed_result['object_detection_crops'])} object detection crops")
        
        # Visualize crops
        print("\n3. Visualizing crops...")
        visualize_image_crops(image_path, "test_crops_visualization.png")
        
    except Exception as e:
        print(f"❌ Advanced preprocessing failed: {e}")
        return
    
    # Step 3: Test RAG retrieval with multiple variants
    print("\n4. Testing RAG retrieval with multiple image variants...")
    try:
        if model_manager.clip_model is not None:
            captions = rag_retriever.retrieve_topk_captions_from_image_path(
                image_path, 
                use_advanced_preprocessing=True
            )
            
            print("✅ RAG retrieval successful!")
            print(f"\n📝 Retrieved captions:")
            print(f"{captions}")
            
            # Compare with simple preprocessing
            print("\n5. Comparing with simple preprocessing...")
            simple_captions = rag_retriever.retrieve_topk_captions_from_image_path(
                image_path, 
                use_advanced_preprocessing=False
            )
            
            print(f"\n📝 Simple preprocessing captions:")
            print(f"{simple_captions}")
            
            print(f"\n📊 Comparison:")
            print(f"  Advanced: {len(captions.split())} words")
            print(f"  Simple:   {len(simple_captions.split())} words")
            
        else:
            print("⚠️ CLIP model not loaded, skipping RAG retrieval test")
            
    except Exception as e:
        print(f"❌ RAG retrieval test failed: {e}")
    
    print("\n✅ Advanced preprocessing test completed!")

def test_configuration():
    """Test the configuration settings for advanced preprocessing."""
    print("🔧 Testing Configuration Settings")
    print("=" * 40)
    
    from src.config import (
        ENABLE_SEGMENTATION, ENABLE_OBJECT_DETECTION,
        CROPS_PER_MODE, CAPTIONS_PER_CROP,
        SEGMENTATION_MODEL, OBJECT_DETECTION_MODEL
    )
    
    print(f"Segmentation enabled: {ENABLE_SEGMENTATION}")
    print(f"Object detection enabled: {ENABLE_OBJECT_DETECTION}")
    print(f"Crops per mode: {CROPS_PER_MODE}")
    print(f"Captions per crop: {CAPTIONS_PER_CROP}")
    print(f"Segmentation model: {SEGMENTATION_MODEL}")
    print(f"Object detection model: {OBJECT_DETECTION_MODEL}")

def main():
    """Main test function."""
    print("🚀 Advanced RAG Image Captioning Test Suite")
    print("=" * 50)
    
    # Test configuration
    test_configuration()
    
    # Check if test image is provided
    if len(sys.argv) < 2:
        print("\n❓ Usage: python test_advanced_preprocessing.py <image_path>")
        print("Example: python test_advanced_preprocessing.py ../coco/train2017/000000000009.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Run advanced preprocessing test
    test_advanced_preprocessing(image_path)

if __name__ == "__main__":
    main()
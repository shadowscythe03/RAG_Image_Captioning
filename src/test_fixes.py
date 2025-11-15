"""
Simple test script to reproduce and fix the exact errors you encountered.
"""
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """Run the same test that was failing."""
    print("🚀 Setting up RAG Image Captioning System...")
    
    try:
        # Import and setup
        from config import validate_paths, create_output_dirs
        from models import initialize_models  
        from database import db_manager
        from image_preprocessing import image_preprocessor
        from rag_retrieval import rag_retriever
        from evaluation import evaluation_manager
        
        # Validate paths
        if not validate_paths():
            print("❌ Path validation failed. Please ensure COCO dataset is available.")
            return
        
        # Create output directories
        create_output_dirs()
        
        print("Initializing models...")
        model_manager = initialize_models()
        
        print("Initializing ChromaDB at ../chroma_db...")
        db_manager.initialize_database()
        
        print("✅ System setup complete!")
        
        # Test on the same image that was failing
        test_image = "../coco/train2017/train2017/000000000009.jpg"
        print(f"🖼️ Testing on image: {test_image}")
        
        # Test the preprocessing (this was causing the segmentation error)
        print("🔍 Generating segmentation crops...")
        try:
            # Load image first
            from PIL import Image
            image = Image.open(test_image).convert('RGB')
            
            # Test segmentation
            seg_crops = image_preprocessor.get_segmentation_crops(image)
            print(f"Generated {len(seg_crops)} segmentation crops")
            
            # Test object detection  
            print("🎯 Generating object detection crops...")
            obj_crops = image_preprocessor.get_object_detection_crops(image)
            print(f"Generated {len(obj_crops)} object detection crops")
            
        except Exception as e:
            print(f"⚠️ Error in advanced preprocessing: {e}")
            print("Continuing with basic preprocessing...")
        
        # Test RAG retrieval (this was causing the dict error)
        print("🔄 Testing RAG retrieval...")
        try:
            result = evaluation_manager.generate_rag_enhanced_caption(test_image)
            
            if "error" not in result:
                print("✅ Caption generation successful!")
                print(f"🔹 Context: {result.get('context_captions', 'None')}")
                print(f"🖼️ Generated Caption: {result.get('generated_caption', 'None')}")
            else:
                print(f"❌ Caption generation failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error in RAG-enhanced generation: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
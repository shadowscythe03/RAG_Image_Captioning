"""
Main script to run the RAG Image Captioning system.
Provides command-line interface and orchestrates all components.
"""
import os
import argparse
from typing import Optional

from config import validate_paths, create_output_dirs
from models import initialize_models
from database import db_manager
from training import run_training
from evaluation import run_evaluation, test_single_image

def setup_system():
    """Initialize the complete RAG image captioning system."""
    print("🚀 Setting up RAG Image Captioning System...")
    
    # Validate paths
    if not validate_paths():
        print("❌ Path validation failed. Please ensure COCO dataset is available.")
        return False
    
    # Create output directories
    create_output_dirs()
    
    # Initialize models
    model_manager = initialize_models()
    
    # Initialize database
    db_manager.initialize_database()
    
    print("✅ System setup complete!")
    return True

def build_database():
    """Build the ChromaDB database from COCO dataset."""
    print("🔧 Building ChromaDB database...")
    
    # Ensure system is set up
    if not setup_system():
        return False
    
    # Build database
    success = db_manager.build_database_from_coco()
    
    if success:
        print("✅ Database built successfully!")
        db_manager.get_database_stats()
    else:
        print("❌ Database build failed!")
    
    return success

def train_model():
    """Train the OFA model with RAG context."""
    print("🎯 Starting model training...")
    
    # Ensure system is set up
    if not setup_system():
        return False
    
    # Run training
    success = run_training()
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
    
    return success

def evaluate_system(num_images: int = 10):
    """Evaluate the system on random images."""
    print(f"🔍 Evaluating system on {num_images} images...")
    
    # Ensure system is set up
    if not setup_system():
        return None
    
    # Run evaluation
    results = run_evaluation(num_images)
    
    if results:
        print(f"✅ Evaluation completed on {len(results)} images!")
        
        # Display results
        from evaluation import evaluation_manager
        evaluation_manager.display_evaluation_results(results)
    else:
        print("❌ Evaluation failed!")
    
    return results

def test_image(image_path: str, use_advanced: bool = True):
    """Test the system on a single image with optional advanced preprocessing."""
    print(f"🖼️ Testing on image: {image_path}")
    print(f"Advanced preprocessing: {'Enabled' if use_advanced else 'Disabled'}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Ensure system is set up
    if not setup_system():
        return None
    
    # Test single image
    result = test_single_image(image_path)
    
    if "error" not in result:
        print("✅ Caption generation successful!")
        print(f"\n🔹 Context: {result.get('context_captions', 'None')}")
        print(f"🖼️ Generated Caption: {result.get('generated_caption', 'None')}")
        
        # Show additional info if advanced preprocessing was used
        if use_advanced:
            try:
                from image_preprocessing import visualize_image_crops
                print("\n🎨 Generating crop visualization...")
                visualize_image_crops(image_path)
            except Exception as e:
                print(f"⚠️ Could not generate visualization: {e}")
    else:
        print(f"❌ Caption generation failed: {result['error']}")
    
    return result

def test_advanced_preprocessing(image_path: str):
    """Test advanced preprocessing features on a single image."""
    print(f"🔬 Testing Advanced Preprocessing on: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Ensure system is set up
    if not setup_system():
        return None
    
    try:
        from image_preprocessing import image_preprocessor, visualize_image_crops
        from rag_retrieval import rag_retriever
        
        # Test preprocessing
        print("🔍 Generating image variants...")
        processed_result = image_preprocessor.apply_initial_processing(image_path)
        
        seg_crops = len(processed_result['segmentation_crops'])
        obj_crops = len(processed_result['object_detection_crops'])
        
        print(f"✅ Generated {seg_crops} segmentation crops")
        print(f"✅ Generated {obj_crops} object detection crops")
        
        # Visualize
        print("🎨 Creating visualization...")
        visualize_image_crops(image_path, "advanced_preprocessing_result.png")
        
        # Test RAG retrieval
        print("🔄 Testing RAG retrieval...")
        advanced_captions = rag_retriever.retrieve_topk_captions_from_image_path(
            image_path, use_advanced_preprocessing=True
        )
        simple_captions = rag_retriever.retrieve_topk_captions_from_image_path(
            image_path, use_advanced_preprocessing=False
        )
        
        print(f"\n📊 Results Comparison:")
        print(f"Advanced RAG: {len(advanced_captions.split())} words")
        print(f"Simple RAG:   {len(simple_captions.split())} words")
        
        print(f"\n🔹 Advanced Captions:")
        print(f"{advanced_captions}")
        
        print(f"\n🔹 Simple Captions:")
        print(f"{simple_captions}")
        
        return {
            'segmentation_crops': seg_crops,
            'object_detection_crops': obj_crops,
            'advanced_captions': advanced_captions,
            'simple_captions': simple_captions
        }
        
    except Exception as e:
        print(f"❌ Advanced preprocessing test failed: {e}")
        return None

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="RAG Image Captioning System")
    
    parser.add_argument(
        "command",
        choices=["setup", "build-db", "train", "evaluate", "test", "test-advanced"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to image file (for test command)"
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images for evaluation (default: 10)"
    )
    
    parser.add_argument(
        "--use-advanced",
        action="store_true",
        help="Use advanced preprocessing with segmentation and object detection"
    )
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_system()
        
    elif args.command == "build-db":
        build_database()
        
    elif args.command == "train":
        train_model()
        
    elif args.command == "evaluate":
        evaluate_system(args.num_images)
        
    elif args.command == "test":
        if not args.image_path:
            print("❌ --image-path required for test command")
            return
        test_image(args.image_path, args.use_advanced)
        
    elif args.command == "test-advanced":
        if not args.image_path:
            print("❌ --image-path required for test-advanced command")
            return
        test_advanced_preprocessing(args.image_path)

if __name__ == "__main__":
    main()
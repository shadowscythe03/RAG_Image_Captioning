"""
Quick debug script to test the specific issues you encountered.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_segmentation_fix():
    """Test the segmentation issue specifically."""
    try:
        from image_preprocessing import image_preprocessor
        from PIL import Image
        
        # Test image path
        test_image_path = "../coco/train2017/train2017/000000000009.jpg"
        
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found: {test_image_path}")
            return False
        
        print("🔧 Testing segmentation fix...")
        
        # Load image
        image = Image.open(test_image_path).convert('RGB')
        
        # Test segmentation crops
        print("🔍 Testing segmentation cropping...")
        seg_crops = image_preprocessor.get_segmentation_crops(image)
        print(f"Generated {len(seg_crops)} segmentation crops")
        
        # Test object detection crops  
        print("🎯 Testing object detection cropping...")
        obj_crops = image_preprocessor.get_object_detection_crops(image)
        print(f"Generated {len(obj_crops)} object detection crops")
        
        # Test full preprocessing
        print("🔧 Testing full preprocessing...")
        result = image_preprocessor.apply_initial_processing(test_image_path)
        
        print(f"✅ Full preprocessing successful:")
        print(f"  - Original image: {type(result['original'])}")
        print(f"  - Segmentation crops: {len(result['segmentation_crops'])}")
        print(f"  - Object detection crops: {len(result['object_detection_crops'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_retrieval_fix():
    """Test the RAG retrieval dict issue."""
    try:
        from models import initialize_models
        from database import db_manager
        from rag_retrieval import rag_retriever
        
        print("🔧 Testing RAG retrieval fix...")
        
        # Initialize system
        print("Initializing models...")
        model_manager = initialize_models()
        
        print("Initializing database...")
        db_manager.initialize_database()
        
        # Test image path
        test_image_path = "../coco/train2017/train2017/000000000009.jpg"
        
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found: {test_image_path}")
            return False
        
        # Test simple RAG retrieval
        print("🔍 Testing simple RAG retrieval...")
        simple_captions = rag_retriever.retrieve_topk_captions_from_image_path(
            test_image_path, use_advanced_preprocessing=False
        )
        print(f"✅ Simple RAG successful: {len(simple_captions.split())} words")
        
        # Test advanced RAG retrieval
        print("🔍 Testing advanced RAG retrieval...")
        advanced_captions = rag_retriever.retrieve_topk_captions_from_image_path(
            test_image_path, use_advanced_preprocessing=True
        )
        print(f"✅ Advanced RAG successful: {len(advanced_captions.split())} words")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_fix():
    """Test the evaluation dict issue."""
    try:
        from evaluation import evaluation_manager
        
        print("🔧 Testing evaluation fix...")
        
        # Test image path
        test_image_path = "../coco/train2017/train2017/000000000009.jpg"
        
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found: {test_image_path}")
            return False
        
        # Test caption generation
        print("🔍 Testing caption generation...")
        result = evaluation_manager.generate_rag_enhanced_caption(test_image_path)
        
        if "error" in result:
            print(f"❌ Caption generation failed: {result['error']}")
            return False
        else:
            print(f"✅ Caption generation successful!")
            print(f"  - Context: {len(result.get('context_captions', '').split())} words")
            print(f"  - Generated: {result.get('generated_caption', 'None')}")
            return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running debug tests for the reported issues...\n")
    
    # Test 1: Segmentation fix
    print("=" * 60)
    print("TEST 1: Segmentation Error Fix")
    print("=" * 60)
    test1_result = test_segmentation_fix()
    
    # Test 2: RAG retrieval fix  
    print("\n" + "=" * 60)
    print("TEST 2: RAG Retrieval Dict Fix")
    print("=" * 60)
    test2_result = test_rag_retrieval_fix()
    
    # Test 3: Evaluation fix
    print("\n" + "=" * 60)
    print("TEST 3: Evaluation Dict Fix")
    print("=" * 60)
    test3_result = test_evaluation_fix()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Segmentation Fix: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"RAG Retrieval Fix: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    print(f"Evaluation Fix: {'✅ PASSED' if test3_result else '❌ FAILED'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("\n🎉 All tests passed! The fixes should resolve your issues.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
# RAG Image Captioning

A Retrieval-Augmented Generation (RAG) system for image captioning using CLIP embeddings and OFA model.

## Project Structure

```
RAG_Image_Captioning/
├── src/
│   ├── config.py              # Configuration and constants
│   ├── models.py              # Model loading and initialization
│   ├── database.py            # ChromaDB operations
│   ├── image_preprocessing.py # Image preprocessing utilities
│   ├── rag_retrieval.py       # RAG retrieval functions
│   ├── training.py            # Training utilities
│   ├── evaluation.py          # Evaluation and testing
│   ├── main.py               # Main orchestration script
│   ├── utils.py              # Utility functions
│   └── build_db.py           # Original database building script
├── chroma_db/                 # ChromaDB database directory
├── coco/                      # COCO dataset (downloaded)
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Features

### Core Components

1. **Configuration Management** (`config.py`)
   - Centralized configuration for all components
   - Path validation and setup
   - Hyperparameter management
   - **Advanced preprocessing settings**

2. **Model Management** (`models.py`)
   - CLIP model loading and initialization
   - OFA model setup (when available)
   - Device management and model utilities

3. **Database Operations** (`database.py`)
   - ChromaDB initialization and management
   - COCO dataset embedding and indexing
   - Similarity search and retrieval

4. **Advanced Image Preprocessing** (`image_preprocessing.py`)
   - 🆕 **Image Segmentation** using DETR for semantic region extraction
   - 🆕 **Object Detection** for focused object cropping
   - 🆕 **Multi-variant RAG**: Original + Segmentation crops + Object detection crops
   - 🆕 **Intelligent crop generation** with configurable parameters
   - 🆕 **Visualization tools** for crop analysis
   - CLIP and OFA preprocessing pipelines
   - Domain-specific preprocessing support
   - Batch processing capabilities

5. **Enhanced RAG Retrieval** (`rag_retrieval.py`)
   - 🆕 **Multi-image RAG**: Retrieval from original + all crop variants
   - 🆕 **Intelligent caption aggregation** with deduplication
   - 🆕 **Configurable captions per crop**
   - Top-k caption retrieval from image embeddings
   - Context-enhanced prompt generation
   - Batch retrieval and statistics

6. **Training Pipeline** (`training.py`)
   - OFA model fine-tuning with RAG context
   - Custom collate functions and data loading
   - Training loop with logging and checkpointing

7. **Evaluation System** (`evaluation.py`)
   - Random image evaluation
   - RAG vs. non-RAG comparison
   - Performance metrics and visualization

## Advanced Image Processing Pipeline 🆕

The system now features a sophisticated preprocessing pipeline that generates multiple image variants for enhanced RAG retrieval:

### 🔍 Segmentation Mode
- Uses **DETR-based semantic segmentation** to identify distinct regions
- Extracts **top-N semantic segments** based on size and relevance
- Creates **focused crops** around each segment with intelligent padding

### 🎯 Object Detection Mode  
- Employs **DETR object detection** to locate individual objects
- Generates **bounding box crops** around detected objects
- Prioritizes **high-confidence detections** for optimal results

### 🔄 Multi-Variant RAG Retrieval
For each input image, the system now retrieves captions from:
1. **Original image** (global context)
2. **Segmentation crops** (semantic regions) 
3. **Object detection crops** (specific objects)

This provides **richer, more comprehensive context** for caption generation!

### Key Configuration Options

```python
# In config.py
ENABLE_SEGMENTATION = True          # Enable segmentation-based cropping
ENABLE_OBJECT_DETECTION = True      # Enable object detection cropping
CROPS_PER_MODE = 3                  # Number of crops per mode
CAPTIONS_PER_CROP = 3               # Captions to retrieve per crop
AGGREGATE_CAPTIONS = True           # Intelligent caption aggregation
```

### Advanced Processing Workflow

```python
# Single image processing
from image_preprocessing import apply_advanced_preprocessing

result = apply_advanced_preprocessing("path/to/image.jpg")
# Returns: {
#   'original': PIL.Image,
#   'segmentation_crops': [PIL.Image, ...],
#   'object_detection_crops': [PIL.Image, ...]
# }

# RAG retrieval with multiple variants
from rag_retrieval import retrieve_captions_for_image

captions = retrieve_captions_for_image("path/to/image.jpg")
# Automatically uses all image variants for retrieval!
```

### Visualization Tools

```python
# Visualize generated crops
from image_preprocessing import visualize_image_crops

visualize_image_crops("path/to/image.jpg", "crops_visualization.png")
# Creates a grid showing original + all generated crops
```

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup System

```bash
cd src
python main.py setup
```

### 3. Download COCO Dataset (if not already available)

```python
from utils import download_coco_dataset
download_coco_dataset()
```

### 4. Build ChromaDB Database

```bash
python main.py build-db
```

### 5. Train Model (when OFA is available)

```bash
python main.py train
```

### 6. Evaluate System

```bash
python main.py evaluate --num-images 10
```

### 7. Test Single Image

```bash
# Simple test
python main.py test --image-path path/to/your/image.jpg

# Test with advanced preprocessing
python main.py test --image-path path/to/your/image.jpg --use-advanced

# Comprehensive advanced preprocessing test
python main.py test-advanced --image-path path/to/your/image.jpg
```

## API Usage

### Quick Start

```python
from src.main import setup_system
from src.evaluation import test_single_image

# Initialize system
setup_system()

# Test an image
result = test_single_image("path/to/image.jpg")
print(f"Generated Caption: {result['generated_caption']}")
```

### Custom Preprocessing

```python
from src.image_preprocessing import image_preprocessor
from PIL import Image

# Load and apply custom preprocessing
image = Image.open("path/to/image.jpg")
processed_image = image_preprocessor.apply_initial_processing(image)

# Add your custom processing here
# processed_image = your_custom_function(processed_image)
```

### RAG Retrieval

```python
from src.rag_retrieval import retrieve_captions_for_image

# Retrieve similar captions for an image
captions = retrieve_captions_for_image("path/to/image.jpg", top_k=5)
print(f"Retrieved captions: {captions}")
```

## Configuration

Key configuration options in `config.py`:

- `CLIP_MODEL_NAME`: CLIP model variant ("ViT-B-32")
- `DEFAULT_TOP_K`: Number of captions to retrieve (5)
- `BATCH_SIZE`: Training batch size (4)
- `LEARNING_RATE`: Training learning rate (5e-5)

## Extending the System

### Adding Custom Preprocessing

1. Edit `image_preprocessing.py`
2. Implement your logic in `apply_initial_processing()`
3. Test with `test_single_image()`

### Adding New Evaluation Metrics

1. Edit `evaluation.py`
2. Add new methods to `EvaluationManager`
3. Integrate with `evaluate_system()`

### Custom Training Strategies

1. Edit `training.py`
2. Modify `TrainingManager` methods
3. Add new loss functions or optimizers

## Notes

- OFA model integration is currently commented out but ready for activation
- The system is designed to be modular and extensible
- ChromaDB provides persistent storage for embeddings
- All components include error handling and logging

## Dependencies

- PyTorch and torchvision for deep learning
- open_clip_torch for CLIP model
- chromadb for vector database
- PIL/Pillow for image processing
- tqdm for progress bars
- Additional utilities for visualization and data handling

## Future Enhancements

- [ ] Multi-modal fusion techniques
- [ ] Advanced retrieval strategies
- [ ] Custom evaluation metrics
- [ ] Web interface for interactive testing
- [ ] API endpoint for service deployment
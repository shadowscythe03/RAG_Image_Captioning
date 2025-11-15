"""
Streamlit App for RAG Image Captioning.
Simple interface for uploading images, detecting objects, and generating captions.
"""
import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path

from src import RAGImageCaptioningPipeline, Config


# Page config
st.set_page_config(
    page_title="RAG Image Captioning",
    page_icon="🖼️",
    layout="wide"
)


@st.cache_resource
def load_pipeline():
    """Load and cache the pipeline."""
    config = Config()
    pipeline = RAGImageCaptioningPipeline(config)
    return pipeline


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location."""
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def display_variants(variants: dict):
    """Display original image and crops."""
    st.subheader("Image Variants & Detected Objects")
    
    # Display original
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(variants['original'], caption='Original', use_column_width=True)
    
    # Display segmentation crops
    seg_crops = variants.get('segmentation_crops', [])
    obj_crops = variants.get('object_detection_crops', [])
    detected_objects = variants.get('detected_objects', [])
    
    cols = [col2, col3, col4]
    col_idx = 0
    
    for i, crop in enumerate(seg_crops):
        if col_idx >= len(cols):
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            col_idx = 0
        
        with cols[col_idx]:
            st.image(crop, caption=f'Segment {i+1}', use_column_width=True)
        col_idx += 1
    
    for i, crop in enumerate(obj_crops):
        if col_idx >= len(cols):
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            col_idx = 0
        
        with cols[col_idx]:
            st.image(crop, caption=f'Object {i+1}', use_column_width=True)
            if i < len(detected_objects):
                obj_info = detected_objects[i]
                st.caption(f"Confidence: {obj_info['score']:.2%}")
        col_idx += 1


def main():
    st.title("🖼️ RAG Image Captioning")
    st.markdown("**Upload an image to detect objects and generate captions using RAG retrieval**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        init_button = st.button("🚀 Initialize Pipeline", type="primary")
        
        if init_button:
            with st.spinner("Loading models... This may take a minute..."):
                try:
                    pipeline = load_pipeline()
                    pipeline.initialize(load_db=True)
                    st.success("✅ Pipeline initialized!")
                    st.session_state['initialized'] = True
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state['initialized'] = False
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        use_advanced = st.checkbox(
            "Use Advanced Preprocessing", 
            value=True,
            help="Enable object detection and segmentation for better captions"
        )

        enable_llm = st.checkbox(
            "LLM Caption Generation (BLIP-2)",
            value=True,
            help="Use BLIP-2 to generate a final caption using retrieved context"
        )

        top_k = st.slider(
            "Top-K Captions",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of similar captions to retrieve"
        )

        show_variants = st.checkbox(
            "Show Image Variants",
            value=False,
            help="Display segmentation and detection crops"
        )

        show_stats = st.checkbox(
            "Show Retrieval Stats",
            value=False,
            help="Display detailed retrieval statistics"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses:
        - **CLIP** for image encoding
        - **DETR** for object detection/segmentation
        - **ChromaDB** for retrieval
        - **RAG** for context-aware captioning
        """)
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Upload an image to analyze"
    )
    
    if uploaded_file is not None:
        # Save and display uploaded image
        image_path = save_uploaded_file(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(Image.open(image_path), use_column_width=True)
        
        with col2:
            st.subheader("Actions")
            
            if not st.session_state.get('initialized', False):
                st.warning("⚠️ Please initialize the pipeline first (see sidebar)")
                st.info("Click '🚀 Initialize Pipeline' in the sidebar to load models")
            else:
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    try:
                        pipeline = load_pipeline()
                        # Enable LLM generation in config if selected
                        pipeline.config.ENABLE_LLM_GENERATION = enable_llm
                        if enable_llm:
                            pipeline.model_manager.load_blip2_model()
                        with st.spinner("Processing image..."):
                            # Process image
                            result = pipeline.caption_image(
                                image_path,
                                use_advanced=use_advanced,
                                top_k=top_k
                            )
                            st.success("✅ Analysis complete!")
                            # Display caption
                            if enable_llm and result.get('mode') == 'llm_generation':
                                st.markdown("### 📝 LLM-Generated Caption (BLIP-2)")
                                st.info(result.get('generated_caption', 'No caption generated'))
                                st.markdown("---")
                                st.markdown("#### Retrieved Context")
                                st.code(result.get('raw_retrieved_context', ''), language='text')
                            elif result.get('mode') == 'llm_error':
                                st.error(result.get('error', 'LLM caption generation failed'))
                            else:
                                st.markdown("### 📝 Generated Caption")
                                caption = result.get('aggregated_caption', 'No caption generated')
                                st.info(caption)
                            # Display metadata
                            st.markdown("### 📊 Processing Info")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric(
                                    "Processing Mode",
                                    result.get('mode', 'unknown').title()
                                )
                            with col_b:
                                st.metric(
                                    "Variants Processed",
                                    result.get('variants_processed', result.get('num_crops', 0))
                                )
                            # Show detected objects
                            detected_objects = result.get('detected_objects', [])
                            if detected_objects:
                                st.markdown("### 🎯 Detected Objects")
                                for i, obj in enumerate(detected_objects, 1):
                                    st.write(f"{i}. Label ID: {obj.get('label_id', '?')}, Confidence: {obj.get('score', 0):.2%}")
                            # Store result in session state for variant display
                            st.session_state['last_result'] = result
                            st.session_state['last_image_path'] = image_path
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
        
        # Show variants if requested and available
        if show_variants and st.session_state.get('last_image_path') == image_path:
            st.markdown("---")
            try:
                pipeline = load_pipeline()
                with st.spinner("Generating image variants..."):
                    variants = pipeline.get_image_variants(image_path)
                    display_variants(variants)
            except Exception as e:
                st.warning(f"Could not generate variants: {e}")
        
        # Show statistics if requested
        if show_stats and st.session_state.get('initialized', False):
            st.markdown("---")
            try:
                pipeline = load_pipeline()
                with st.spinner("Computing retrieval statistics..."):
                    stats = pipeline.get_stats(image_path, top_k=top_k)
                    
                    if stats:
                        st.subheader("📈 Retrieval Statistics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Captions Retrieved", stats.get('num_retrieved', 0))
                        
                        with col2:
                            st.metric(
                                "Avg Caption Length",
                                f"{stats.get('avg_caption_length', 0):.1f} words"
                            )
                        
                        with col3:
                            st.metric(
                                "Total Context",
                                f"{stats.get('total_context_length', 0)} words"
                            )
                        
                        if 'min_distance' in stats:
                            st.markdown("**Distance Metrics:**")
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Min Distance", f"{stats['min_distance']:.4f}")
                            
                            with col_b:
                                st.metric("Avg Distance", f"{stats['avg_distance']:.4f}")
                            
                            with col_c:
                                st.metric("Max Distance", f"{stats['max_distance']:.4f}")
            
            except Exception as e:
                st.warning(f"Could not compute statistics: {e}")
    
    else:
        st.info("👆 Upload an image to get started")
        
        # Show example info
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            **Steps:**
            1. Click **Initialize Pipeline** in the sidebar (first time only)
            2. Upload an image using the file uploader
            3. Adjust settings in the sidebar if needed
            4. Click **Analyze Image** to generate caption
            
            **Features:**
            - Basic mode: Direct image captioning
            - Advanced mode: Uses object detection and segmentation
            - Retrieval stats: Shows similarity metrics
            - Image variants: Displays detected regions
            """)


if __name__ == "__main__":
    main()

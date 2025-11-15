"""
Dummy Streamlit UI for RAG Image Captioning.
No backend functionality, just static UI elements for layout testing.
"""
import streamlit as st
from PIL import Image

# Page config
st.set_page_config(
    page_title="RAG Image Captioning (Dummy)",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ RAG Image Captioning (Dummy)")
st.markdown("**Upload an image to see the UI layout. No backend functionality.**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.button("🚀 Initialize Pipeline (Dummy)", type="primary")
    st.markdown("---")
    st.subheader("Settings")
    st.checkbox("Use Advanced Preprocessing", value=True, help="Enable object detection and segmentation for better captions")
    st.checkbox("LLM Caption Generation (BLIP-2)", value=True, help="Use BLIP-2 to generate a final caption using retrieved context")
    st.slider("Top-K Captions", min_value=1, max_value=20, value=5, help="Number of similar captions to retrieve")
    st.checkbox("Show Image Variants", value=False, help="Display segmentation and detection crops")
    st.checkbox("Show Retrieval Stats", value=False, help="Display detailed retrieval statistics")
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
    # Dummy image display
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded Image")
        st.image(Image.open(uploaded_file), width='stretch')
    with col2:
        st.subheader("Actions")
        st.button("🔍 Analyze Image (Dummy)", type="primary", width='stretch')
        st.markdown("### 📝 Generated Caption")
        st.info("This is a dummy caption. No backend processing.")
        st.markdown("### 📊 Processing Info")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Processing Mode", "Dummy")
        with col_b:
            st.metric("Variants Processed", 0)
        st.markdown("### 🎯 Detected Objects")
        st.write("No objects detected (dummy mode)")
        st.markdown("---")
        st.markdown("#### Retrieved Context")
        st.code("Dummy context: No retrieval performed.", language='text')
    # Dummy variants
    st.markdown("---")
    st.subheader("Image Variants & Detected Objects (Dummy)")
    st.image(Image.open(uploaded_file), caption='Original', width='stretch')
    st.image(Image.open(uploaded_file), caption='Segment 1 (Dummy)', width='stretch')
    st.image(Image.open(uploaded_file), caption='Object 1 (Dummy)', width='stretch')
    # Dummy stats
    st.markdown("---")
    st.subheader("📈 Retrieval Statistics (Dummy)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Captions Retrieved", 0)
    with col2:
        st.metric("Avg Caption Length", "0.0 words")
    with col3:
        st.metric("Total Context", "0 words")
    st.markdown("**Distance Metrics:**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Min Distance", "0.0000")
    with col_b:
        st.metric("Avg Distance", "0.0000")
    with col_c:
        st.metric("Max Distance", "0.0000")
else:
    st.info("👆 Upload an image to get started (dummy mode)")
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Steps:**
        1. Click **Initialize Pipeline** in the sidebar (dummy only)
        2. Upload an image using the file uploader
        3. Adjust settings in the sidebar if needed
        4. Click **Analyze Image** to see dummy output
        
        **Features:**
        - Basic mode: Direct image captioning (dummy)
        - Advanced mode: Uses object detection and segmentation (dummy)
        - Retrieval stats: Shows similarity metrics (dummy)
        - Image variants: Displays detected regions (dummy)
        """)

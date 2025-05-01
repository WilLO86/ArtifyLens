import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="ArtifyLens", layout="wide")
st.title("ArtifyLens - Image Processing App")

def apply_canny_edge(image, invert=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    if invert:
        edges = 255 - edges
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_sketch_filter(image):
    kernel = np.array([[-1,-1,-1], 
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

def apply_random_filter(image, invert_edges=False):
    edges = apply_canny_edge(image, invert_edges)
    sketch = apply_sketch_filter(image)
    # Blend the two filters with equal weights
    blended = cv2.addWeighted(edges, 0.5, sketch, 0.5, 0)
    return blended

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_np)
    
    with col2:
        st.subheader("Processed Image")
        filter_options = ["Edge Detection", "Artistic Sketch", "Random Filter"]
        selected_filter = st.radio("Select Filter", filter_options)
        
        invert_edges = st.checkbox("Invert Edge Colors", value=False)
        
        with st.spinner("Applying filter..."):
            if selected_filter == "Edge Detection":
                processed_image = apply_canny_edge(image_np, invert_edges)
                filter_title = f"Applied: Edge Detection Filter{'(Inverted)' if invert_edges else ''}"
            elif selected_filter == "Artistic Sketch":
                processed_image = apply_sketch_filter(image_np)
                filter_title = "Applied: Artistic Sketch Filter"
            else:  # Random Filter
                processed_image = apply_random_filter(image_np, invert_edges)
                filter_title = f"Applied: Random Filter Blend{'(Inverted Edges)' if invert_edges else ''}"
        
        st.markdown(f"### {filter_title}")
        st.image(processed_image)
        
        # Download button
        if st.button("Download Processed Image"):
            processed_pil = Image.fromarray(processed_image)
            buf = io.BytesIO()
            processed_pil.save(buf, format="PNG")
            st.download_button(
                label="Click to Download",
                data=buf.getvalue(),
                file_name="processed_image.png",
                mime="image/png"
            )

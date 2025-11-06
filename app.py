import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Image Convolution Explorer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, responsive design
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .info-card h3 {
        color: #667eea;
        margin-top: 0;
    }
    
    /* Filter buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Image containers */
    .image-container {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .image-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
    }
    
    /* Kernel matrix styling */
    .kernel-display {
        background: #2d3748;
        color: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Success/Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Upload section */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
    }
    
    /* Responsive image sizing */
    .stImage {
        max-width: 100%;
        height: auto;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        .main-header p {
            font-size: 0.9rem;
        }
        .image-container {
            padding: 0.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'kernel_used' not in st.session_state:
    st.session_state.kernel_used = None
if 'filter_name' not in st.session_state:
    st.session_state.filter_name = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None

# Define convolution kernels
KERNELS = {
    'blur': {
        'matrix': np.ones((3, 3), dtype=np.float32) / 9,
        'name': 'Blur (Averaging)',
        'explanation': """
        **How it works:** The blur filter uses a 3×3 averaging kernel where each element is 1/9. 
        This means each pixel in the output is the average of its 8 neighbors plus itself.
        
        **Effect:** Creates a smoothing effect by reducing sharp transitions between pixels, 
        useful for noise reduction and creating a softer appearance.
        
        **Formula:** New_Pixel = (Sum of 9 neighboring pixels) / 9
        """,
        'icon': '🌫️'
    },
    'sharpen': {
        'matrix': np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32),
        'name': 'Sharpen',
        'explanation': """
        **How it works:** The sharpen kernel emphasizes the center pixel (value: 5) while 
        subtracting the neighboring pixels (-1). This amplifies differences between the center 
        and its surroundings.
        
        **Effect:** Enhances edges and fine details, making the image appear crisper and more defined.
        
        **Formula:** New_Pixel = 5×Center - (Top + Bottom + Left + Right)
        """,
        'icon': '✨'
    },
    'edge': {
        'matrix': np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float32),
        'name': 'Edge Detection (Laplacian)',
        'explanation': """
        **How it works:** The Laplacian kernel strongly emphasizes the center pixel (value: 8) 
        while subtracting all 8 neighbors (-1 each). This highlights areas where pixel intensity 
        changes rapidly.
        
        **Effect:** Detects edges and boundaries by finding regions of high intensity change, 
        making edges appear bright against a dark background.
        
        **Formula:** New_Pixel = 8×Center - (Sum of 8 neighbors)
        """,
        'icon': '🔍'
    }
}

def apply_convolution(image, kernel):
    """Apply convolution filter to image"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Apply filter to each channel
    filtered = cv2.filter2D(img_array, -1, kernel)
    
    # Clip values to valid range
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    
    return Image.fromarray(filtered)

def format_kernel_display(kernel, name):
    """Format kernel matrix for display"""
    kernel_str = f"<div class='kernel-display'>"
    kernel_str += f"<h4 style='margin-top:0; color: #a0aec0;'>Convolution Kernel: {name}</h4>"
    kernel_str += "<pre style='font-size: 1.1rem; line-height: 1.8;'>"
    
    for row in kernel:
        row_str = "[ "
        for val in row:
            if val >= 0:
                row_str += f" {val:6.3f} "
            else:
                row_str += f"{val:6.3f} "
        row_str += "]"
        kernel_str += row_str + "\n"
    
    kernel_str += "</pre></div>"
    return kernel_str

def resize_image_for_display(image, max_width=500):
    """Resize image to appropriate display size"""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

# Header
st.markdown("""
    <div class='main-header'>
        <h1>🖼️ Image Convolution Explorer</h1>
        <p>Learn how convolution matrices transform images through interactive filtering</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📚 About Convolution")
    st.markdown("""
    <div class='info-card'>
        <h3>What is Convolution?</h3>
        <p>Convolution is a mathematical operation where a small matrix (kernel) slides across an image, 
        multiplying overlapping values and summing them to create a new pixel value.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧮 The Mathematics")
    st.latex(r"(I * K)(i,j) = \sum_{m}\sum_{n} I(i+m, j+n) \cdot K(m,n)")
    
    st.markdown("""
    <div style='font-size: 0.9rem; color: #666; padding: 0.5rem;'>
    Where:<br>
    • I = Input image<br>
    • K = Convolution kernel<br>
    • (i,j) = Pixel position<br>
    • * = Convolution operator
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Guide")
    st.markdown("""
    1. 📤 Upload an image
    2. 🎨 Choose a filter
    3. 👀 Compare results
    4. 💾 Download if you like!
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ Tips")
    st.info("• Use images < 5MB for best performance\n• Try different filters to see varied effects\n• Click Reset to start fresh")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to apply convolution filters"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file)
        display_original = resize_image_for_display(original_image, max_width=450)
        
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.markdown("<div class='image-label'>📸 Original Image</div>", unsafe_allow_html=True)
        st.image(display_original, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display image info
        st.markdown(f"""
        <div class='metric-card'>
            <strong>Dimensions:</strong> {original_image.size[0]} × {original_image.size[1]} pixels
        </div>
        """, unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        st.markdown("### 🎨 Apply Filters")
        
        # Filter buttons
        filter_cols = st.columns(3)
        
        with filter_cols[0]:
            if st.button(f"{KERNELS['blur']['icon']} Blur", use_container_width=True):
                st.session_state.processed_image = apply_convolution(original_image, KERNELS['blur']['matrix'])
                st.session_state.kernel_used = KERNELS['blur']['matrix']
                st.session_state.filter_name = KERNELS['blur']['name']
                st.session_state.explanation = KERNELS['blur']['explanation']
        
        with filter_cols[1]:
            if st.button(f"{KERNELS['sharpen']['icon']} Sharpen", use_container_width=True):
                st.session_state.processed_image = apply_convolution(original_image, KERNELS['sharpen']['matrix'])
                st.session_state.kernel_used = KERNELS['sharpen']['matrix']
                st.session_state.filter_name = KERNELS['sharpen']['name']
                st.session_state.explanation = KERNELS['sharpen']['explanation']
        
        with filter_cols[2]:
            if st.button(f"{KERNELS['edge']['icon']} Edge Detect", use_container_width=True):
                st.session_state.processed_image = apply_convolution(original_image, KERNELS['edge']['matrix'])
                st.session_state.kernel_used = KERNELS['edge']['matrix']
                st.session_state.filter_name = KERNELS['edge']['name']
                st.session_state.explanation = KERNELS['edge']['explanation']
        
        # Display processed image
        if st.session_state.processed_image is not None:
            display_processed = resize_image_for_display(st.session_state.processed_image, max_width=450)
            
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.markdown(f"<div class='image-label'>✨ {st.session_state.filter_name}</div>", unsafe_allow_html=True)
            st.image(display_processed, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download button
            buf = io.BytesIO()
            st.session_state.processed_image.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="💾 Download Processed Image",
                data=byte_im,
                file_name=f"processed_{st.session_state.filter_name.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div class='info-card' style='margin-top: 3rem;'>
            <h3>👈 Start by uploading an image</h3>
            <p>Upload an image on the left to begin exploring convolution filters!</p>
        </div>
        """, unsafe_allow_html=True)

# Reset button
if uploaded_file is not None:
    st.markdown("---")
    if st.button("🔄 Reset & Start Over", use_container_width=True):
        st.session_state.processed_image = None
        st.session_state.kernel_used = None
        st.session_state.filter_name = None
        st.session_state.explanation = None
        st.rerun()

# Display kernel and explanation
if st.session_state.kernel_used is not None:
    st.markdown("---")
    st.markdown("### 🧮 Kernel Matrix & Explanation")
    
    col_kernel, col_explain = st.columns([1, 1])
    
    with col_kernel:
        kernel_html = format_kernel_display(st.session_state.kernel_used, st.session_state.filter_name)
        st.markdown(kernel_html, unsafe_allow_html=True)
    
    with col_explain:
        st.markdown(f"""
        <div class='info-card'>
            <h3>{st.session_state.filter_name} Filter</h3>
            {st.session_state.explanation}
        </div>
        """, unsafe_allow_html=True)

# Educational content section
if uploaded_file is None:
    st.markdown("---")
    st.markdown("## 📖 Understanding Convolution Filters")
    
    cols = st.columns(3)
    
    for idx, (key, kernel_info) in enumerate(KERNELS.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class='info-card'>
                <h3>{kernel_info['icon']} {kernel_info['name']}</h3>
                {kernel_info['explanation']}
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        <p>🎓 Built with Streamlit • Made for Learning Image Processing</p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
            Try different filters to see how convolution matrices transform images!
        </p>
    </div>
""", unsafe_allow_html=True)

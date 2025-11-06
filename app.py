import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import base64

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
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    .info-card h3 {
        color: #667eea;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .info-card p {
        color: #333;
        line-height: 1.6;
        margin: 0.5rem 0;
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
    
    /* Image label styling */
    .image-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: white;
        margin: 1rem 0 0.5rem 0;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: inline-block;
        width: 100%;
    }
    
    /* Image sizing */
    .stImage {
        max-width: 500px !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    .stImage > img {
        max-width: 500px !important;
        max-height: 500px !important;
        width: auto !important;
        height: auto !important;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stImage > img:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Fullscreen modal */
    .fullscreen-modal {
        display: none;
        position: fixed;
        z-index: 9999;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.95);
        overflow: auto;
    }
    
    .fullscreen-modal.active {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .fullscreen-content {
        max-width: 95%;
        max-height: 95%;
        object-fit: contain;
        border-radius: 10px;
    }
    
    .close-fullscreen {
        position: absolute;
        top: 20px;
        right: 40px;
        color: #fff;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
        z-index: 10000;
        transition: color 0.3s ease;
    }
    
    .close-fullscreen:hover {
        color: #ff6b6b;
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
    
    /* Calculation steps styling */
    .calculation-box {
        background: #ffffff;
        border: 2px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .calculation-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Processing animation */
    .processing-container {
        background: #1a202c;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .processing-title {
        color: #fff;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
    }
    
    .matrix-animation {
        background: #2d3748;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        color: #48bb78;
        font-size: 0.95rem;
        border: 2px solid #667eea;
        animation: glow 1.5s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 10px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
    }
    
    .progress-step {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .step-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    .matrix-display {
        background: #1a202c;
        color: #fff;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
        overflow-x: auto;
    }
    
    .calculation-step {
        background: #fff;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 3px solid #48bb78;
        font-family: 'Courier New', monospace;
        color: #333;
    }
    
    .final-result {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
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
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_filter' not in st.session_state:
    st.session_state.current_filter = None

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
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=np.float32),
        'name': 'Sharpen',
        'explanation': """
        **How it works:** The sharpen kernel strongly emphasizes the center pixel (value: 9) while 
        subtracting all 8 neighboring pixels (-1 each). This amplifies differences between the center 
        pixel and its surroundings, creating a strong sharpening effect.
        
        **Effect:** Enhances edges and fine details significantly, making the image appear much crisper and more defined.
        
        **Formula:** New_Pixel = 9×Center - Sum(All 8 Neighbors)
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

def show_processing_animation(image, kernel, filter_name):
    """Show animated mathematical calculation step-by-step"""
    # Get sample region data
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    h, w = gray.shape
    x, y = w // 2, h // 2
    region = gray[y-1:y+2, x-1:x+2].astype(float)
    
    # Create a single placeholder for all steps
    anim_placeholder = st.empty()
    
    # Step 1: Show the image region (3x3 pixels)
    region_str = ""
    for row in region:
        region_str += "    [ "
        for val in row:
            region_str += f"{val:7.1f} "
        region_str += "]\n"
    
    anim_placeholder.markdown("""
        <div class='matrix-animation' style='min-height: 400px; display: flex; align-items: center; justify-content: center;'>
            <pre style='margin:0; color: #48bb78; font-size: 1.2rem;'>
Image Region (3×3 pixels):

""" + region_str + """
            </pre>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.3)
    
    # Step 2: Show the kernel
    kernel_str = ""
    for row in kernel:
        kernel_str += "    [ "
        for val in row:
            kernel_str += f"{val:7.3f} "
        kernel_str += "]\n"
    
    anim_placeholder.markdown("""
        <div class='matrix-animation' style='min-height: 400px; display: flex; align-items: center; justify-content: center;'>
            <pre style='margin:0; color: #ffd700; font-size: 1.2rem;'>
Convolution Kernel:

""" + kernel_str + """
            </pre>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.3)
    
    # Step 3: Show element-wise multiplication
    products = region * kernel
    
    calc_str = "Element-wise Multiplication:\n\n"
    for i in range(3):
        calc_str += "    [ "
        for j in range(3):
            calc_str += f"{region[i,j]:6.1f} × {kernel[i,j]:6.3f} "
        calc_str += "]\n"
    
    calc_str += "\n=\n\n"
    for i in range(3):
        calc_str += "    [ "
        for j in range(3):
            calc_str += f"{products[i,j]:9.2f} "
        calc_str += "]\n"
    
    anim_placeholder.markdown("""
        <div class='matrix-animation' style='min-height: 400px; display: flex; align-items: center; justify-content: center;'>
            <pre style='margin:0; color: #ff6b6b; font-size: 1rem;'>
""" + calc_str + """
            </pre>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.8)
    
    # Step 4: Show summation
    total = np.sum(products)
    
    sum_parts = []
    for i in range(3):
        for j in range(3):
            sum_parts.append(f"{products[i,j]:.2f}")
    
    sum_str = "Sum of all elements:\n\n"
    sum_str += "    " + " +\n    ".join(sum_parts[:3]) + " +\n"
    sum_str += "    " + " +\n    ".join(sum_parts[3:6]) + " +\n"
    sum_str += "    " + " +\n    ".join(sum_parts[6:9]) + "\n\n"
    sum_str += f"    = {total:.2f}\n"
    
    anim_placeholder.markdown("""
        <div class='matrix-animation' style='min-height: 400px; display: flex; align-items: center; justify-content: center;'>
            <pre style='margin:0; color: #a0d2ff; font-size: 1rem;'>
""" + sum_str + """
            </pre>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.8)
    
    # Step 5: Show final clipping
    final = np.clip(total, 0, 255)
    
    result_str = f"Raw Sum: {total:.2f}\n\n"
    result_str += "Clip to valid pixel range [0, 255]\n\n"
    result_str += f"Final Pixel Value: {final:.0f}"
    
    anim_placeholder.markdown("""
        <div class='matrix-animation' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 400px; display: flex; align-items: center; justify-content: center;'>
            <pre style='margin:0; color: #ffd700; font-size: 1.3rem; text-align: center; padding: 2rem;'>
""" + result_str + """
            </pre>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.2)
    
    # Step 6: Show completion message
    anim_placeholder.markdown("""
        <div class='matrix-animation' style='background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); min-height: 400px; display: flex; align-items: center; justify-content: center;'>
            <div style='text-align: center; padding: 2rem; color: white;'>
                <div style='font-size: 1.4rem; font-weight: bold; margin-bottom: 1rem;'>
                    ✓ Calculation Complete!
                </div>
                <div class='spinner'></div>
                <div style='margin-top: 1rem; font-size: 1.1rem;'>
                    Applying to entire {} × {} image...
                </div>
            </div>
        </div>
    """.format(image.size[0], image.size[1]), unsafe_allow_html=True)
    time.sleep(0.8)

def get_sample_region_calculation(image, kernel, x=None, y=None):
    """Get a sample calculation showing the convolution process"""
    img_array = np.array(image)
    
    # Convert to grayscale for simpler calculation display
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    h, w = gray.shape
    
    # Select a region from the middle if not specified
    if x is None or y is None:
        x = w // 2
        y = h // 2
    
    # Ensure we're not at the edge
    if x < 1: x = 1
    if y < 1: y = 1
    if x >= w - 1: x = w - 2
    if y >= h - 1: y = h - 2
    
    # Extract 3x3 region
    region = gray[y-1:y+2, x-1:x+2].astype(float)
    
    # Calculate convolution step by step
    steps = []
    calculation_parts = []
    
    for i in range(3):
        for j in range(3):
            pixel_val = region[i, j]
            kernel_val = kernel[i, j]
            product = pixel_val * kernel_val
            steps.append({
                'position': f"[{i},{j}]",
                'pixel': pixel_val,
                'kernel': kernel_val,
                'product': product
            })
            calculation_parts.append(f"{pixel_val:.0f} × {kernel_val:.3f}")
    
    total = np.sum(region * kernel)
    final_value = np.clip(total, 0, 255)
    
    return {
        'region': region,
        'kernel': kernel,
        'steps': steps,
        'calculation_parts': calculation_parts,
        'total': total,
        'final_value': final_value,
        'position': (x, y)
    }

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

def resize_image_for_display(image, max_width=500, max_height=500):
    """Resize image to fixed display size"""
    width, height = image.size
    
    # Calculate aspect ratio
    aspect = width / height
    
    # Resize to fixed dimensions while maintaining aspect ratio
    if aspect > 1:
        # Wider image
        new_width = min(width, max_width)
        new_height = int(new_width / aspect)
    else:
        # Taller image
        new_height = min(height, max_height)
        new_width = int(new_height * aspect)
    
    # Ensure it doesn't exceed max dimensions
    if new_width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect)
    if new_height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect)
    
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
    <div class='info-card' style='background: #ffffff;'>
        <h3 style='color: #667eea;'>What is Convolution?</h3>
        <p style='color: #333; line-height: 1.6;'>Convolution is a mathematical operation where a small matrix (kernel) slides across an image, 
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
    <div style='color: #333; line-height: 1.8;'>
    1. 📤 Upload an image<br>
    2. 🎨 Choose a filter<br>
    3. 👀 Compare results<br>
    4. 🧮 See live calculations<br>
    5. 💾 Download if you like!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ Tips")
    st.info("• Use images < 5MB for best performance\n• Try different filters to see varied effects\n• Click Reset to start fresh")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload Your Image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to apply convolution filters",
        label_visibility="visible"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file)
        display_original = resize_image_for_display(original_image, max_width=500, max_height=500)
        
        # Convert original image to base64 for fullscreen
        buf_original = io.BytesIO()
        original_image.save(buf_original, format='PNG')
        orig_img_base64 = base64.b64encode(buf_original.getvalue()).decode()
        
        st.markdown("<div class='image-label'>📸 Original Image (Click to Enlarge)</div>", unsafe_allow_html=True)
        
        # Fullscreen modal for original image
        st.markdown(f"""
            <div id="fullscreenModalOrig" class="fullscreen-modal">
                <span class="close-fullscreen" onclick="closeFullscreenOrig()">&times;</span>
                <img class="fullscreen-content" src="data:image/png;base64,{orig_img_base64}">
            </div>
            
            <div style='cursor: pointer;' onclick='openFullscreenOrig()' title='Click to view fullscreen'>
        """, unsafe_allow_html=True)
        
        st.image(display_original)
        
        st.markdown("""
            </div>
            
            <script>
            function openFullscreenOrig() {
                document.getElementById('fullscreenModalOrig').classList.add('active');
            }
            
            function closeFullscreenOrig() {
                document.getElementById('fullscreenModalOrig').classList.remove('active');
            }
            
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape') {
                    closeFullscreenOrig();
                }
            });
            
            document.getElementById('fullscreenModalOrig').addEventListener('click', function(event) {
                if (event.target === this) {
                    closeFullscreenOrig();
                }
            });
            </script>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='metric-card' style='margin-top: 0.5rem;'>
            <strong>Dimensions:</strong> {original_image.size[0]} × {original_image.size[1]} pixels
        </div>
        """, unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        st.markdown("<h3 style='margin-top: 0;'>🎨 Apply Filters</h3>", unsafe_allow_html=True)
        
        # Filter buttons
        filter_cols = st.columns(3)
        
        with filter_cols[0]:
            if st.button(f"{KERNELS['blur']['icon']} Blur", key="blur_btn"):
                st.session_state.processing = True
                st.session_state.current_filter = 'blur'
                st.rerun()
        
        with filter_cols[1]:
            if st.button(f"{KERNELS['sharpen']['icon']} Sharpen", key="sharpen_btn"):
                st.session_state.processing = True
                st.session_state.current_filter = 'sharpen'
                st.rerun()
        
        with filter_cols[2]:
            if st.button(f"{KERNELS['edge']['icon']} Edge Detect", key="edge_btn"):
                st.session_state.processing = True
                st.session_state.current_filter = 'edge'
                st.rerun()
        
        # Create a placeholder for animation/image
        result_placeholder = st.empty()
        
        # Show animation or processed image
        if st.session_state.processing and st.session_state.current_filter:
            # Show animation in place of everything
            filter_key = st.session_state.current_filter
            with result_placeholder.container():
                show_processing_animation(original_image, KERNELS[filter_key]['matrix'], KERNELS[filter_key]['name'])
            
            # After animation, process the image
            st.session_state.processed_image = apply_convolution(original_image, KERNELS[filter_key]['matrix'])
            st.session_state.kernel_used = KERNELS[filter_key]['matrix']
            st.session_state.filter_name = KERNELS[filter_key]['name']
            st.session_state.explanation = KERNELS[filter_key]['explanation']
            st.session_state.processing = False
            st.session_state.current_filter = None
            st.rerun()
        
        elif st.session_state.processed_image is not None:
            # Display processed image in the same box after animation is done
            with result_placeholder.container():
                display_processed = resize_image_for_display(st.session_state.processed_image, max_width=500, max_height=500)
                
                st.markdown(f"""
                    <div class='image-label'>✨ {st.session_state.filter_name} (Click to Enlarge)</div>
                """, unsafe_allow_html=True)
                
                # Convert image to base64 for fullscreen modal
                buf_fullscreen = io.BytesIO()
                st.session_state.processed_image.save(buf_fullscreen, format='PNG')
                img_base64 = base64.b64encode(buf_fullscreen.getvalue()).decode()
                
                # Display image with fullscreen capability
                st.markdown(f"""
                    <!-- Fullscreen Modal -->
                    <div id="fullscreenModal" class="fullscreen-modal">
                        <span class="close-fullscreen" onclick="closeFullscreen()">&times;</span>
                        <img class="fullscreen-content" id="fullscreenImg" src="data:image/png;base64,{img_base64}">
                    </div>
                    
                    <div style='cursor: pointer;' onclick='openFullscreen()' title='Click to view fullscreen'>
                """, unsafe_allow_html=True)
                
                st.image(display_processed)
                
                st.markdown("""
                    </div>
                    
                    <script>
                    function openFullscreen() {
                        document.getElementById('fullscreenModal').classList.add('active');
                    }
                    
                    function closeFullscreen() {
                        document.getElementById('fullscreenModal').classList.remove('active');
                    }
                    
                    document.addEventListener('keydown', function(event) {
                        if (event.key === 'Escape') {
                            closeFullscreen();
                        }
                    });
                    
                    document.getElementById('fullscreenModal').addEventListener('click', function(event) {
                        if (event.target === this) {
                            closeFullscreen();
                        }
                    });
                    </script>
                """, unsafe_allow_html=True)
                
                # Download button
                buf = io.BytesIO()
                st.session_state.processed_image.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 Download Processed Image",
                    data=byte_im,
                    file_name=f"processed_{st.session_state.filter_name.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )
            
            # Show real-time calculation below (outside placeholder)
            st.markdown("---")
            st.markdown("### 🧮 Real-Time Convolution Calculation")
            
            calc_data = get_sample_region_calculation(original_image, st.session_state.kernel_used)
            
            st.markdown("""
                <div class='calculation-box'>
                    <div class='calculation-header'>
                        📍 Sample Calculation at Position ({}, {})
                    </div>
                </div>
            """.format(calc_data['position'][0], calc_data['position'][1]), unsafe_allow_html=True)
            
            col_calc1, col_calc2 = st.columns(2)
            
            with col_calc1:
                st.markdown("**Step 1: Input Region (3×3 pixels)**")
                region_str = "<div class='matrix-display'><pre style='margin:0; color: #fff;'>"
                for row in calc_data['region']:
                    region_str += "[ "
                    for val in row:
                        region_str += f"{val:6.0f} "
                    region_str += "]\n"
                region_str += "</pre></div>"
                st.markdown(region_str, unsafe_allow_html=True)
                
                st.markdown("**Step 2: Convolution Kernel**")
                kernel_str = "<div class='matrix-display'><pre style='margin:0; color: #fff;'>"
                for row in calc_data['kernel']:
                    kernel_str += "[ "
                    for val in row:
                        kernel_str += f"{val:6.3f} "
                    kernel_str += "]\n"
                kernel_str += "</pre></div>"
                st.markdown(kernel_str, unsafe_allow_html=True)
            
            with col_calc2:
                st.markdown("**Step 3: Element-wise Multiplication**")
                st.markdown("<div class='step-container'>", unsafe_allow_html=True)
                
                # Show first few calculations
                for i, step in enumerate(calc_data['steps'][:9]):
                    calc_line = f"<div class='calculation-step'>"
                    calc_line += f"Position {step['position']}: {step['pixel']:.0f} × {step['kernel']:.3f} = {step['product']:.2f}"
                    calc_line += "</div>"
                    st.markdown(calc_line, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("**Step 4: Sum All Products**")
                sum_text = " + ".join([f"{step['product']:.2f}" for step in calc_data['steps'][:3]])
                sum_text += " + ... "
                st.markdown(f"<div class='step-container' style='color: #333;'><code>{sum_text}</code></div>", unsafe_allow_html=True)
            
            # Final result
            st.markdown(f"""
                <div class='final-result'>
                    🎯 Final Result: {calc_data['total']:.2f} → Clipped to [0, 255] → <strong>{calc_data['final_value']:.0f}</strong>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #ffc107;'>
                    <strong>💡 What Just Happened?</strong><br>
                    <span style='color: #333;'>
                    1. We selected a 3×3 region from the image<br>
                    2. Multiplied each pixel by the corresponding kernel value<br>
                    3. Summed all 9 products together<br>
                    4. This sum becomes the new pixel value at the center position!<br>
                    5. This process repeats for EVERY pixel in the image!
                    </span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-card' style='margin-top: 3rem; background: #ffffff; border: 1px solid #e0e0e0;'>
            <h3 style='color: #667eea;'>👈 Start by uploading an image</h3>
            <p style='color: #333;'>Upload an image on the left to begin exploring convolution filters!</p>
        </div>
        """, unsafe_allow_html=True)

# Reset button
if uploaded_file is not None:
    st.markdown("---")
    if st.button("🔄 Reset & Start Over", key="reset_btn"):
        st.session_state.processed_image = None
        st.session_state.kernel_used = None
        st.session_state.filter_name = None
        st.session_state.explanation = None
        st.session_state.processing = False
        st.session_state.current_filter = None
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
        <div class='info-card' style='background: #ffffff; border: 1px solid #e0e0e0;'>
            <h3 style='color: #667eea;'>{st.session_state.filter_name} Filter</h3>
            <div style='color: #333; line-height: 1.8;'>
            {st.session_state.explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Educational content section
if uploaded_file is None:
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #667eea;'>📖 Understanding Convolution Filters</h2>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    
    for idx, (key, kernel_info) in enumerate(KERNELS.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class='info-card' style='background: #ffffff; border: 1px solid #e0e0e0;'>
                <h3 style='color: #667eea;'>{kernel_info['icon']} {kernel_info['name']}</h3>
                <div style='color: #333; line-height: 1.8;'>
                {kernel_info['explanation']}
                </div>
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

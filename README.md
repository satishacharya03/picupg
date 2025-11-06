# 🖼️ Image Processing with Convolution Filters

An educational Python GUI application that demonstrates how convolution matrices (kernels) are used to manipulate and process images. Built with Streamlit for an interactive, web-based interface.

## 🎯 Features

- **📤 Image Upload**: Support for JPG and PNG formats
- **🎨 Three Filter Options**:
  - **Blur**: Smoothing using 3x3 averaging kernel
  - **Sharpen**: Edge enhancement with sharpening kernel
  - **Edge Detection**: Laplacian filter to detect boundaries
- **👀 Side-by-Side Comparison**: View original and processed images together
- **📊 Matrix Visualization**: See the exact convolution kernel used for each filter
- **📚 Educational Content**: Detailed explanations of how each filter works
- **💾 Download Results**: Save your processed images
- **🧮 Mathematical Formulas**: View convolution formulas in the sidebar
- **🔄 Reset Function**: Start over with a new image anytime

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this project**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Run the following command in your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload an Image**: Click the upload button and select a JPG or PNG image
2. **Choose a Filter**: Click one of the three filter buttons (Blur, Sharpen, or Edge Detection)
3. **View Results**: See your original and processed images side-by-side
4. **Examine the Matrix**: Check out the convolution kernel that was applied
5. **Read the Explanation**: Understand how the filter works mathematically
6. **Download**: Save your processed image using the download button
7. **Reset**: Click the reset button to start over with a new image

## 🧮 Understanding Convolution

### What is Convolution?

Convolution is a mathematical operation where a small matrix (kernel) slides across an image:
1. At each position, the kernel multiplies overlapping pixel values
2. All multiplied values are summed together
3. The sum becomes the new pixel value
4. This process transforms the entire image!

### The Math

For a kernel K and image I, convolution at position (i,j) is:

```
(I * K)(i,j) = ΣΣ I(i+m, j+n) · K(m,n)
```

### Filter Examples

**Blur Filter (3x3 Averaging)**:
```
[ 1/9  1/9  1/9 ]
[ 1/9  1/9  1/9 ]
[ 1/9  1/9  1/9 ]
```
Averages all 9 neighboring pixels for a smoothing effect.

**Sharpen Filter**:
```
[  0  -1   0 ]
[ -1   5  -1 ]
[  0  -1   0 ]
```
Emphasizes differences between center pixel and neighbors, enhancing edges.

**Edge Detection (Laplacian)**:
```
[ -1  -1  -1 ]
[ -1   8  -1 ]
[ -1  -1  -1 ]
```
Highlights areas of rapid intensity change, revealing edges.

## 📁 Project Structure

```
picupg/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Technologies Used

- **Streamlit**: Modern web-based GUI framework
- **NumPy**: Numerical computing and array operations
- **OpenCV**: Computer vision and image processing
- **Pillow**: Image file handling
- **Matplotlib**: Visualization support

## 🎓 Educational Value

This application is perfect for:
- Students learning image processing concepts
- Understanding convolution operations visually
- Exploring how filters transform images
- Seeing the connection between matrices and image effects
- Hands-on experimentation with different kernels

## 🤝 Troubleshooting

### App won't start?
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### Image not displaying?
- Ensure the image format is JPG or PNG
- Try a smaller image file (< 5MB recommended)

### Filters not working?
- Click the Reset button and try again
- Refresh the browser page

## 📝 License

This is an educational project - free to use and modify!

## 🌟 Future Enhancements

Potential additions:
- Custom kernel input
- More filter types (Gaussian blur, Sobel, etc.)
- Adjustable kernel sizes
- Image comparison slider
- Batch processing
- Filter strength adjustment

---

**Enjoy learning about image processing with matrices! 🎨📊**

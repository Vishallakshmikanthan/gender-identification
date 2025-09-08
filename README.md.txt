# Live Gender Detection Application

A real-time gender detection application using OpenCV and TensorFlow Lite for efficient face detection and gender classification.

## 🎯 Features

- **Real-time Processing**: Live webcam feed with instant gender detection
- **High Performance**: TensorFlow Lite integration for optimized inference
- **Multiple Face Support**: Detect and classify gender for multiple faces simultaneously  
- **Performance Monitoring**: Real-time FPS display and performance statistics
- **User-Friendly Interface**: Simple keyboard controls and visual feedback
- **Robust Error Handling**: Graceful handling of camera and model errors
- **Modular Architecture**: Clean, well-documented code structure

## 📋 Requirements

- Python 3.8 or higher
- Webcam (built-in or external)
- At least 4GB RAM
- CPU with AVX support (recommended for TensorFlow)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository or download the files
git clone <repository-url>
cd gender-detection-app

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Model

You have several options for the gender detection model:

#### Option A: Use a Pre-converted Model
Download a pre-converted TensorFlow Lite model:
```bash
# Create models directory
mkdir models

# Download a sample model (replace with actual URL)
# Example using wget or curl:
wget -O models/gender_model.tflite "https://example.com/gender_model.tflite"
```

#### Option B: Convert Your Own Model
If you have a TensorFlow/Keras model, convert it:

```python
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('your_gender_model.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('models/gender_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### Option C: Download Face Detection Models (Optional)
For better face detection, download DNN models:

```bash
cd models

# Download face detection prototxt
wget -O deploy.prototxt.txt "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

# Download face detection model
wget -O res10_300x300_ssd_iter_140000.caffemodel "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
```

### 3. Run the Application

```bash
# Basic usage
python main.py

# With custom model path
python main.py --model models/your_custom_model.tflite

# With different camera (if you have multiple cameras)
python main.py --camera 1

# With custom confidence threshold
python main.py --confidence 0.7
```

## 🎮 Controls

- **Q**: Quit the application
- **S**: Save current frame to `saved_frames/` directory
- **F**: Toggle fullscreen mode

## 📊 Model Requirements

Your TensorFlow Lite model should:
- Accept RGB images as input
- Output gender classification (binary or multi-class)
- Be optimized for inference (quantized recommended)

### Expected Model Formats

**Binary Classification (Sigmoid):**
- Output: Single value between 0-1
- Interpretation: >0.5 = Female, ≤0.5 = Male

**Multi-class Classification (Softmax):**
- Output: Two values [male_prob, female_prob]
- Interpretation: argmax(output) = predicted class

## 🔧 Configuration

### Command Line Arguments

```bash
python main.py --help
```

- `--model`: Path to TensorFlow Lite model (default: `models/gender_model.tflite`)
- `--camera`: Camera index (default: 0)
- `--confidence`: Face detection confidence threshold (default: 0.5)

### Environment Variables

You can also set configuration via environment variables:

```bash
export GENDER_MODEL_PATH="path/to/your/model.tflite"
export CAMERA_INDEX=0
export DETECTION_CONFIDENCE=0.5
```

## 🏗️ Project Structure

```
gender-detection-app/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── face_detector.py      # Face detection logic
│   ├── gender_classifier.py  # Gender classification
│   ├── video_processor.py    # Video processing utilities
│   └── performance_monitor.py # Performance monitoring
├── models/                   # Model files
│   ├── gender_model.tflite   # Your TensorFlow Lite model
│   ├── deploy.prototxt.txt   # Face detection config (optional)
│   └── res10_300x300_ssd_iter_140000.caffemodel # Face detection model (optional)
└── saved_frames/             # Saved screenshots (auto-created)
```

## 🔍 Troubleshooting

### Common Issues

#### Camera Not Found
```
❌ Error: Cannot access camera 0
```
**Solutions:**
- Check camera permissions in your OS
- Try different camera index: `python main.py --camera 1`
- Ensure no other applications are using the camera

#### Model Loading Error
```
❌ Error loading model
```
**Solutions:**
- Verify model file exists and path is correct
- Check model compatibility with TensorFlow Lite
- Ensure model file isn't corrupted

#### Poor Performance
```
FPS: 5.2 (low performance)
```
**Solutions:**
- Close other applications
- Reduce video resolution in code
- Use a quantized model
- Check CPU usage

#### Face Detection Issues
```
No faces detected
```
**Solutions:**
- Improve lighting conditions  
- Adjust `--confidence` threshold
- Ensure face is clearly visible
- Check if DNN models are loaded

### Performance Optimization

#### For Better FPS:
1. Use quantized TensorFlow Lite models
2. Reduce input image resolution
3. Process every Nth frame instead of all frames
4. Use threading for model inference

#### For Better Accuracy:
1. Use higher resolution models
2. Improve lighting conditions
3. Ensure faces are well-centered
4. Use data augmentation during training

## 🌐 Google Colab Usage

To run in Google Colab:

```python
# Install dependencies
!pip install opencv-python tensorflow

# Enable camera access
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

# Note: Google Colab doesn't support direct webcam access
# You'll need to modify the code to work with uploaded images
# or use Colab's camera capture widget
```

## 📈 Performance Metrics

The application monitors:
- **FPS**: Frames processed per second
- **Frame Time**: Time to process each frame
- **Detection Rate**: Percentage of frames with detected faces
- **Classification Confidence**: Model prediction confidence

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest

# Run tests
python -m pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow team for machine learning framework
- Contributors and testers

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with:
   - Error message
   - System specifications
   - Steps to reproduce

---

**Happy coding! 🚀**
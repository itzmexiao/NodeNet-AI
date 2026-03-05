# NodeNet-AI
NodeNet-AI is an end-to-end computer vision pipeline designed for the automated classification of lung cancer subtypes from medical imaging. The system leverages a deep Convolutional Neural Network (CNN) architecture to distinguish between specific malignant conditions and normal tissue with high precision.

# Technical Overview
The project consists of a model training pipeline developed in TensorFlow and a production-ready deployment interface built with Streamlit and ONNX Runtime.

## Classification Targets
The model is trained to identify four specific classes:
- Adenocarcinoma: Left lower lobe, T2 N0 M0 Ib.
- Large Cell Carcinoma: Left hilum, T2 N2 M0 IIIa.
- Squamous Cell Carcinoma: Left hilum, T1 N2 M0 IIIa.
- Normal: No malignant indicators present.

# System Workflow
1. The following stages define the end-to-end process from data preparation to clinical inference:
2. Data Preprocessing: Images are resized to 150x150 pixels and normalized using a 1/255 rescale factor.
3. Model Training: A Sequential CNN extracts spatial features through multiple convolutional and pooling layers.
4. Optimization: The model is compiled using the Adam optimizer and Categorical Crossentropy loss function.
5. Serialization: The trained Keras model is exported to the Open Neural Network Exchange (ONNX) format for optimized inference performance.
6. Deployment: A Streamlit interface handles image uploads, runs the ONNX session, and visualizes diagnostic confidence scores.

# Codeflow Architecture
The repository's logic is partitioned into two primary functional blocks:
# 1. Training Pipeline (Lung_cancer_detection.ipynb)
- **Image Augmentation**: Utilizes ImageDataGenerator for validation splitting and rescaling.
- **Feature Extraction**:
    - Layer 1: 32 filters (3x3), ReLU activation.
    - Layer 2: 64 filters (3x3), ReLU activation.
    - Layer 3: 128 filters (3x3), ReLU activation.
Classification Head: Flattens the 3D feature maps into a 1D vector followed by a 128-unit dense layer and a 50% Dropout regularization layer.
Final Output: 4-unit Softmax layer providing probability distribution across classes.

# 2. Inference Engine (app.py)
**Session Initialization:** Loads lung_cancer_model.onnx using onnxruntime.InferenceSession for CPU-optimized execution.
**Input Processing:** Converts uploaded images via PIL, resizes them to the 150x150 input shape, and casts them to float32 NumPy arrays.
**Diagnostic Visualization:**
Runs inference to generate confidence percentages.
Calculates relative risk and maps results to clinical survival statistics.

Renders dynamic Plotly charts to display classification probability.

Performance Metrics
The model demonstrates robust convergence over 50 epochs:

Training Accuracy: >99%

Validation Accuracy: ~93.4%

Validation Loss: ~0.57 (Epoch 43)

Installation and Execution
Requirements
Python 3.10+

Dependencies: streamlit, onnxruntime, tensorflow, numpy, pillow, plotly, pandas.

# DeepFER – Facial Emotion Recognition Using Deep Learning

DeepFER is a deep learning-based facial emotion recognition system that classifies facial images into **seven emotion categories** using a custom Convolutional Neural Network (CNN) implemented with PyTorch.

The trained model is deployed using **Streamlit**, providing a simple web interface where users can upload an image and receive the predicted emotion along with its confidence score.

## 🎯 Emotion Classes

The model recognizes the following seven emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## 🏗️ Project Architecture

```text
Input Facial Image
        ↓
Image Preprocessing
        ↓
Resize to 224 × 224
        ↓
Tensor Conversion + Normalization
        ↓
DeepFER CNN
        ↓
Softmax
        ↓
Emotion + Confidence
        ↓
Streamlit Web Interface
```

## 📂 Project Structure

```text
DeepFER/
│
├── app.py                  # Streamlit deployment application
├── DeepFER_model.pth       # Trained PyTorch model
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
│
└── archive-3/              # Dataset (used during training)
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    │
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
```

## 🧠 Model

DeepFER uses a custom CNN architecture developed using **PyTorch**.

The convolutional blocks progressively increase the number of filters:

```text
32 → 64 → 128 → 256
```

The network uses:

* Convolutional layers
* Batch Normalization
* ReLU activation
* Max Pooling
* Dropout
* Adaptive Average Pooling
* Fully Connected Layers

The final layer contains **7 output neurons**, corresponding to the seven emotion classes.

## 🖼️ Preprocessing

All images are resized to:

```text
224 × 224 pixels
```

Training augmentation includes:

* Random Horizontal Flip
* Random Rotation
* Color Jitter

Images are normalized using:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

These preprocessing steps are also applied consistently during inference.

## ⚙️ Training

The model was trained using **PyTorch** with GPU acceleration.

### Training Configuration

| Parameter         | Value                      |
| ----------------- | -------------------------- |
| Framework         | PyTorch                    |
| Input Size        | 224 × 224                  |
| Number of Classes | 7                          |
| Loss Function     | Cross Entropy Loss         |
| Optimizer         | Adam                       |
| Learning Rate     | 0.001                      |
| Weight Decay      | 1e-4                       |
| Epochs            | 20                         |
| GPU               | NVIDIA RTX 4060 Laptop GPU |

The best model was saved based on test accuracy.

## 📊 Evaluation

The model was evaluated using:

* Accuracy
* Classification Report
* Confusion Matrix
* Test Loss

The best recorded test accuracy during training was approximately **57.61%**.

Because the dataset contains class imbalance, particularly between classes such as **happy and disgust**, class-wise metrics should also be considered rather than relying only on overall accuracy.

## 💾 Model Checkpoint

The trained model is saved as:

```text
DeepFER_model.pth
```

The checkpoint stores:

```python
{
    "model_state_dict": ...,
    "class_names": ...,
    "image_size": ...
}
```

This allows the Streamlit application to reconstruct the model and use the same class mapping and image size during inference.

## 🌐 Streamlit Deployment

The Streamlit application provides an interactive interface for inference.

### Prediction Pipeline

```text
User Uploads Image
        ↓
Convert to RGB
        ↓
Resize to 224 × 224
        ↓
Tensor Conversion
        ↓
Normalization
        ↓
DeepFER CNN
        ↓
Softmax
        ↓
Predicted Emotion
        ↓
Confidence + Probability Distribution
```

The application displays:

* Uploaded image
* Predicted emotion
* Confidence percentage
* Probability distribution for all seven classes

## 🚀 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd DeepFER
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📦 Requirements

The main dependencies are:

```text
streamlit
torch
torchvision
Pillow
numpy
pandas
```

## 🔮 Future Improvements

Possible improvements include:

* Face detection before emotion classification
* Better handling of class imbalance
* Transfer learning using pretrained CNN architectures
* Hyperparameter tuning
* Improved performance on minority classes
* Real-time webcam-based emotion recognition
* Cloud deployment
* Model monitoring and performance tracking

## 👨‍💻 Project Summary

**DeepFER** demonstrates an end-to-end deep learning workflow:

```text
Data
 ↓
Preprocessing
 ↓
CNN Development
 ↓
Training
 ↓
Evaluation
 ↓
Model Serialization
 ↓
Streamlit Deployment
 ↓
Real-Time Inference
```

The project combines **Computer Vision, Deep Learning, PyTorch, and Streamlit** to transform a trained facial emotion classification model into an accessible application.

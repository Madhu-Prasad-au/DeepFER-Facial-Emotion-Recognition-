import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeepFER - Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL ARCHITECTURE
# Same architecture used during training
# ============================================================

class DeepFER_CNN(nn.Module):

    def __init__(self, num_classes=7):

        super(DeepFER_CNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "DeepFER_model.pth",
        map_location=device
    )

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = DeepFER_CNN(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, class_names, image_size


model, class_names, IMG_SIZE = load_model()


# ============================================================
# IMAGE PREPROCESSING
# Same preprocessing used during testing
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emotion(image):

    image = image.convert("RGB")

    image_tensor = test_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_emotion = class_names[
        predicted.item()
    ]

    confidence = confidence.item() * 100

    probabilities = probabilities[
        0
    ].cpu().numpy() * 100

    return (
        predicted_emotion,
        confidence,
        probabilities
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title("😊 DeepFER")
st.subheader("Facial Emotion Recognition Using Deep Learning")

st.write(
    "Upload a facial image and DeepFER will predict "
    "the most likely facial emotion."
)


st.divider()


# ============================================================
# IMAGE INPUT
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a facial image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Predict Emotion",
        type="primary"
    ):

        with st.spinner(
            "Analyzing facial expression..."
        ):

            emotion, confidence, probabilities = (
                predict_emotion(image)
            )


        st.success(
            f"Predicted Emotion: {emotion}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        st.divider()

        st.subheader(
            "Emotion Probability Distribution"
        )


        probability_df = pd.DataFrame({

            "Emotion": class_names,

            "Probability (%)": probabilities

        })


        probability_df = probability_df.sort_values(
            "Probability (%)",
            ascending=False
        )


        st.bar_chart(
            probability_df.set_index(
                "Emotion"
            )
        )


        st.subheader(
            "Prediction Details"
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DeepFER | Facial Emotion Recognition Using Deep Learning"
)
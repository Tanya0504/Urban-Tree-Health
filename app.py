import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Plant Disease Classifier",
    layout="wide",
    page_icon="🌿"
)

# Add background image from local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply background
add_bg_from_local("background.jpg")  # make sure this file exists

# Load the model
model = load_model("plant_disease_model.h5")

# Class labels
class_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Sidebar Info
st.sidebar.title("🧬 About this App")
st.sidebar.markdown("""
## 🧬 About This App

- Detects plant diseases from leaf images  
- Upload or capture an image → get prediction  
- Fast, accurate & simple to use  

---

## 🧠 Model Info

- Model: MobileNetV2 (Transfer Learning)  
- Dataset: 50,000+ images  
- Classes: 38 plant diseases  
- Accuracy: ~90%  

---

## 📂 Dataset

- Kaggle: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 👨‍💻 Developers

Abhishek, Tanya, Tauhid, Nakshtra  
B.Tech Final Year – Capstone Project
""")

# Page Title
st.markdown("<h1 style='text-align: center;'>🌿 Plant Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Choose image source and detect the disease in seconds!</h4>", unsafe_allow_html=True)
st.write("")

# Choose Input Method
st.markdown("## 📷 Choose Input Method")
input_option = st.radio(
    "Select how you want to provide the leaf image:",
    ("Upload from device", "Take photo using camera"),
    index=0
)

file = None
if input_option == "Upload from device":
    file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])
elif input_option == "Take photo using camera":
    file = st.camera_input("📸 Take a photo")

# Prediction
if file:
    # Process image
    img = Image.open(file).convert("RGB")
    resized_img = img.resize((224, 224))
    img_array = image.img_to_array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds)

    # ✋ Sanity check: is the confidence too low?
    if confidence < 0.5:
        st.image(img, caption="Input Image", use_container_width=True)
        st.error("❌ This image doesn't seem to be a valid plant leaf. Please try again with a clear leaf photo.")
    else:
        # Show result
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption="Leaf Image", use_container_width=True)

        with col2:
            st.markdown("### 🧠 Prediction")
            st.success(f"Detected: {pred_class}")
            st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
            st.progress(int(confidence * 100))

        # Confidence graph
        st.markdown("### 📊 Prediction Confidence Across All Classes")
        pred_df = pd.DataFrame(preds[0], index=class_labels, columns=["Confidence"])
        pred_df = pred_df.sort_values(by="Confidence", ascending=True)

        fig, ax = plt.subplots(figsize=(6, len(class_labels) // 2))
        pred_df.plot.barh(ax=ax, legend=False, color='teal')
        ax.set_xlabel("Confidence Score")
        ax.set_xlim([0, 1])
        st.pyplot(fig)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Made with ❤️ by <b>Abhishek, Tanya, Tauhid, Nakshtra</b></div>",
    unsafe_allow_html=True
)

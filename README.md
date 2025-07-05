# 🌿 Plant Disease Classification Web App

This is a deep learning-based web application for detecting plant diseases from leaf images. The model is trained using a Convolutional Neural Network (CNN) on an augmented dataset and deployed using Streamlit.

🔗 Live Demo: [Click here to try the app](https://plant-disease-app-abhishek.streamlit.app/)  

📽️ Demo Video: [Click here to watch the demo video](https://drive.google.com/file/d/1wQqX1X3hg1TbHYaCYQ3YYh1SM1fk_Tka/view?usp=sharing)

🛠️ Built using: TensorFlow, Keras, Streamlit, Python

---
## 📸 Demo Screenshots
📽️ Demo Video
[Click here to watch the demo video](https://drive.google.com/file/d/1wQqX1X3hg1TbHYaCYQ3YYh1SM1fk_Tka/view?usp=sharing)

### 🔍 Upload Page
![Upload Page](demo/ss1.png)

### 🧠 Prediction Result
![Prediction Result](demo/ss2.png)

### 📊 Confidence Chart
![Confidence Chart](demo/ss3.png)

---
## 🚀 Features

- Upload a leaf image
- Get predicted plant disease with confidence score
- Displays healthy vs infected result
- Uses deep learning (MobileNetV2 + fine-tuning)
- Trained on over 50,000+ augmented leaf images

---

## 📦 Technologies Used

| Tool/Library       | Purpose                          |
|--------------------|----------------------------------|
| TensorFlow / Keras | Model training and inference     |
| Streamlit          | Web interface for deployment     |
| Pillow             | Image processing in Python       |
| NumPy              | Image array handling             |
| Google Colab       | Model training environment       |
| Kaggle Dataset     | Training and validation data     |

---

## 🧠 Model Details

- Base Model: MobileNetV2 (Transfer Learning)
- Input Image Size: 224 x 224
- Final Accuracy: 92–95% on validation set
- Trained using ImageDataGenerator with data augmentation

---

## 🗂️ Dataset Source

This app was trained on the following Kaggle dataset:

🔗 [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 📂 Installation & Running Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/plant-disease-app.git
   cd plant-disease-app

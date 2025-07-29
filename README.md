# 🦠 AI-Based COVID Detection System

This is a multi-component AI project developed during my internship. It integrates *Computer Vision* and *Data Analysis* to assist in early detection and understanding of COVID-19. The entire system runs as a simple and interactive *Streamlit web application*.

---

## 📌 Project Features

### 1. 🖼 Chest X-ray Classifier
- Upload a chest X-ray image.
- Predicts whether the X-ray shows signs of:
  - *COVID*
  - *Normal*
  - *Viral Pneumonia*
- Powered by a deep learning model (covid_xray_model.keras) built using TensorFlow/Keras.

## 🧪 Sample Test Image

You can test the model using sample chest X-ray images provided in the sample_xrays/ folder.

To test:
1. Go to the *🩻 X-ray Classifier* tab in the app
2. Upload one of the sample images:
   - covid_sample.jpg
   - normal_sample.jpg

### 2. 📊 WHO COVID-19 Global Data Analysis
- Uses official WHO dataset (WHO-COVID-19-global-data.csv)
- Visualizes:
  - Top 10 countries with highest total reported cases
  - Daily new cases
  - Cumulative case trends by country
  - Raw data display for selected country

📄 The WHO COVID dataset is large and may not preview in GitHub, but it is included in the repository and used directly by the Streamlit app.

---

📦 *Note:* The model file (covid_xray_model.keras) is too large for GitHub.

➡ Please download it from the link below and place it inside the project folder:

🔗 [Download COVID X-ray Model (.keras)](https://drive.google.com/file/d/1CZ0Saujk9J6DfnZlOvdAp-EakWuasWfE/view?usp=sharing)

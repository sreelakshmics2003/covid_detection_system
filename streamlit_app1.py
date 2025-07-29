import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="COVID-19 AI System", layout="centered")

# ------------------------- SIDEBAR -------------------------
st.sidebar.image("https://i.imgur.com/NbA3RfI.png", use_container_width=True)  # Optional: Replace with your own logo
st.sidebar.title("🧠 COVID-19 AI Dashboard")
app_mode = st.sidebar.radio("📍 Select Module", ["🏠 Home", "🩻 X-ray Classifier", "📊 Global Data Analysis"])

# ------------------------- CUSTOM CSS -------------------------
st.markdown("""
    <style>
        .main { background-color: #f7f7f7; padding: 20px; border-radius: 10px; }
        .title { text-align: center; font-size: 36px; color: #4A90E2; font-weight: bold; }
        .subtitle { text-align: center; font-size: 18px; color: #444; }
    </style>
""", unsafe_allow_html=True)

# ------------------------- HOME PAGE -------------------------
if app_mode == "🏠 Home":
    st.markdown('<div class="main"><div class="title">COVID-19 Detection & Analysis</div><div class="subtitle">An integrated AI system using Deep Learning and WHO data</div></div>', unsafe_allow_html=True)
    st.write("Welcome to the COVID-19 AI dashboard. Use the sidebar to navigate between modules:")

    st.markdown("""
    - 🩻 **X-ray Classifier**: Upload a chest X-ray to detect COVID-19 using a deep learning model.
    - 📊 **Global Data Analysis**: Explore real-world trends using WHO global COVID-19 dataset.
    """)

# ------------------------- X-RAY PREDICTION -------------------------
elif app_mode == "🩻 X-ray Classifier":
    st.header("🩺 Chest X-ray COVID Prediction")

    uploaded_image = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        with st.spinner("🔍 Predicting..."):
            model = load_model("covid_xray_model.keras")
            img = image.load_img(uploaded_image, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0]
            labels = ['COVID', 'NORMAL', 'Viral Pneumonia']
            result = labels[np.argmax(pred)]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded_image, caption="Uploaded X-ray", use_container_width=True)
            with col2:
                st.success(f"🧠 **Predicted Condition:** `{result}`")

# ------------------------- DATA ANALYSIS -------------------------
elif app_mode == "📊 Global Data Analysis":
    st.header("📊 WHO COVID-19 Data Analysis")

    df = pd.read_csv("WHO-COVID-19-global-data.csv")
    df["Date_reported"] = pd.to_datetime(df["Date_reported"])

    top_countries = df.groupby("Country")["New_cases"].sum().sort_values(ascending=False).head(10)

    st.subheader("🌍 Top 10 Countries by Total Reported Cases")
    st.bar_chart(top_countries)

    st.subheader("📈 Trend for Selected Country")
    selected_country = st.selectbox("Choose a country", df["Country"].unique())
    country_data = df[df["Country"] == selected_country]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daily New Cases**")
        st.line_chart(country_data.set_index("Date_reported")[["New_cases"]])
    with col2:
        st.markdown("**Cumulative Cases Over Time**")
        st.line_chart(country_data.set_index("Date_reported")[["Cumulative_cases"]])

    with st.expander("📄 Show Raw Data"):
        st.dataframe(country_data.tail(10))

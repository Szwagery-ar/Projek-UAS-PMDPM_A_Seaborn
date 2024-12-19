import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: linear-gradient(rgba(0,0,0,0.2), rgba(0,0,0,2)), url("https://i.pinimg.com/736x/c9/6d/60/c96d601d5ba32d581f40a8a06d1e0971.jpg");  # Ganti dengan URL gambar Anda
             background-attachment: fixed;
             background-size: cover;
         }}
         
         /* Tambahan style untuk membuat konten lebih readable */
         .main {{
             background-color: rgba(255, 255, 255, 0.85);
             padding: 20px;
             border-radius: 10px;
         }}
         
         .sidebar .sidebar-content {{
             background-color: rgba(255, 255, 255, 0.9);
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

add_bg_from_url()

model = load_model(r"D:\Kuliah\.Semester 5\ML\UAS\New folder\BestModel_MobileNetCNN_Seaborn.h5")

class_names = ['PaprikaHijau', 'PaprikaKuning', 'PaprikaMerah']


def preprocess_image(image_array):
    return tf.cast(image_array, tf.float32) / 255.0


def classify_image(image_path):
    try:

        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

def custom_progress_bar(confidence, color_map):
    percentage1 = confidence[0] * 100  # Hijau
    percentage2 = confidence[1] * 100  # Kuning
    percentage3 = confidence[2] * 100  # Merah
    
    progress_html = f"""
    <div style="width: 100%; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: {color_map['PaprikaHijau']}; color: white; text-align: center; height: 24px; float: left;">
            {percentage1:.2f}%
        </div>
        <div style="width: {percentage2:.2f}%; background: {color_map['PaprikaKuning']}; color: white; text-align: center; height: 24px; float: left;">
            {percentage2:.2f}%
        </div>
        <div style="width: {percentage3:.2f}%; background: {color_map['PaprikaMerah']}; color: white; text-align: center; height: 24px; float: left;">
            {percentage3:.2f}%
        </div>
    </div>
    """
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

st.title("Prediksi Jenis Paprika - Seaborn") 

uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            label, confidence = classify_image(uploaded_file.name)

            if label != "Error":
                color_map = {
                    "PaprikaHijau": "#008000",
                    "PaprikaKuning": "#FFD700",
                    "PaprikaMerah": "#FF0000"
                }
                
                label_color = color_map.get(label, "#000000")

                st.sidebar.write(f"*Nama File:* {uploaded_file.name}")
                st.sidebar.markdown(f"<h4 style='color: {label_color};'>Prediksi: {label}</h4>", unsafe_allow_html=True)

                st.sidebar.write("*Confidence:*")
                display_names = ["PaprikaHijau", "PaprikaKuning", "PaprikaMerah"]  # Sesuai urutan model tanpa 'Paprika'
                for i, name in enumerate(display_names):
                    st.sidebar.write(f"- {name}: {confidence[i] * 100:.2f}%")

                custom_progress_bar(confidence, color_map)

                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
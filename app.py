import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import os
import platform

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="🧠 Reconocimiento de Imágenes", layout="centered")

# --- ESTILO PERSONALIZADO (COLORES) ---
st.markdown("""
    <style>
    body {
        background-color: #1e1e2f;
        color: #f2f2f2;
        font-family: 'Arial';
    }
    .stButton>button {
        background-color: #6a5acd;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #7b68ee;
        color: #fff;
        transform: scale(1.03);
        transition: 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO Y VERSIÓN ---
st.title("🧠 Reconocimiento de Imágenes")
st.write("Versión de Python:", platform.python_version())
st.markdown("---")

# --- VERIFICAR SI EXISTE EL MODELO ---
MODEL_PATH = "keras_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ No se encontró el archivo **keras_model.h5** en el directorio actual.")
    st.info("➡️ Sube el archivo del modelo o colócalo en la misma carpeta del script.")
    st.stop()

# --- CARGAR EL MODELO ---
try:
    model = load_model(MODEL_PATH)
    st.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# --- INTERFAZ ---
image = Image.open('OIG5.jpg')
st.image(image, width=350)

with st.sidebar:
    st.subheader("📸 Instrucciones")
    st.markdown("""
    1. Usa un modelo entrenado en **Teachable Machine**.  
    2. Sube o toma una foto.  
    3. La IA identificará lo que ve según tu modelo.
    """)

img_file_buffer = st.camera_input("Toma una Foto o súbela desde tu dispositivo:")

if img_file_buffer is not None:
    # Crear arreglo base
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Leer imagen
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir a array y normalizar
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Realizar predicción
    st.markdown("### 🔍 Procesando imagen...")
    prediction = model.predict(data)

    # Mostrar resultados
    st.markdown("---")
    st.subheader("📊 Resultado del modelo:")

    if prediction[0][0] > 0.5:
        st.success(f"➡️ **Clase 1 detectada (Izquierda)** con probabilidad: {prediction[0][0]:.3f}")
    elif prediction[0][1] > 0.5:
        st.success(f"⬆️ **Clase 2 detectada (Arriba)** con probabilidad: {prediction[0][1]:.3f}")
    else:
        st.warning("🤔 No se detectó una clase dominante. Intenta con otra imagen.")

    st.markdown("---")
    st.caption("Modelo cargado desde `keras_model.h5` usando Keras y Streamlit 💫")

else:
    st.info("📷 Esperando una imagen... Usa la cámara o súbela para analizarla.")




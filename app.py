import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# --- Configuración general ---
st.set_page_config(
    page_title="Visión IA - Reconocimiento",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Estilo CSS personalizado ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #F8E1F4 0%, #E5CFF7 50%, #D3B9F1 100%);
        }
        .main {
            background: rgba(255, 255, 255, 0.7);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0px 4px 15px rgba(150, 0, 200, 0.2);
        }
        h1 {
            color: #6A1B9A !important;
            text-align: center;
            font-weight: 800;
        }
        .stButton>button {
            background: linear-gradient(90deg, #8E24AA, #AB47BC);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #BA68C8, #CE93D8);
            transform: scale(1.05);
        }
        .result-box {
            background: rgba(255,255,255,0.85);
            border: 2px solid #AB47BC;
            border-radius: 12px;
            padding: 15px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Título principal ---
st.markdown("<h1>🌸 Reconocimiento de Imágenes con Inteligencia Artificial</h1>", unsafe_allow_html=True)
st.write("Versión de Python:", platform.python_version())

# --- Cargar modelo ---
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# --- Imagen inicial ---
st.image("OIG5.jpg", width=350, caption="Ejemplo de imagen de referencia")

# --- Barra lateral ---
with st.sidebar:
    st.subheader("📷 Instrucciones")
    st.write("""
    1. Haz clic en **Toma una Foto**.  
    2. Espera que la IA analice la imagen.  
    3. Observa el resultado en pantalla.
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("🌈 <b>Consejo:</b> Usa buena iluminación y fondo claro.", unsafe_allow_html=True)

# --- Entrada de cámara ---
img_file_buffer = st.camera_input("📸 Toma una Foto para analizar")

# --- Procesamiento de imagen ---
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))
    
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # --- Progreso visual ---
    with st.spinner("🧠 Analizando imagen..."):
        prediction = model.predict(data)

    # --- Mostrar resultado ---
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.image(img, caption="📸 Imagen analizada", width=300)
    
    if prediction[0][0] > 0.5:
        st.success(f"🌟 **Clase detectada:** Izquierda (probabilidad: {prediction[0][0]:.2f})")
    elif prediction[0][1] > 0.5:
        st.success(f"💜 **Clase detectada:** Arriba (probabilidad: {prediction[0][1]:.2f})")
    else:
        st.warning("⚠️ No se detectó ninguna clase con alta probabilidad.")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("✨ Toma una foto para comenzar el reconocimiento.")

# --- Pie de página ---
st.markdown("""
    <hr>
    <p style='text-align:center; color:#7B1FA2; font-size:14px;'>
    Creado con 💖 usando Streamlit, OpenCV y Keras
    </p>
""", unsafe_allow_html=True)





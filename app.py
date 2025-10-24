import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# --- Configuración de la página ---
st.set_page_config(
    page_title="Reconocimiento de Imágenes - IA",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Encabezado principal ---
st.markdown("""
    <h1 style="text-align:center; color:#4B8BBE;">🔍 Reconocimiento de Imágenes con IA</h1>
    <p style="text-align:center; color:#555;">
    Usa un modelo entrenado en <b>Teachable Machine</b> para identificar objetos o gestos desde tu cámara.
    </p>
    <hr style="border: 1px solid #DDD;">
""", unsafe_allow_html=True)

# --- Mostrar versión de Python ---
st.sidebar.markdown(f"🧩 <b>Versión de Python:</b> {platform.python_version()}", unsafe_allow_html=True)

# --- Cargar modelo ---
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# --- Imagen de referencia ---
st.image('OIG5.jpg', width=350, caption="Ejemplo de imagen", use_container_width=False)

# --- Barra lateral con información ---
with st.sidebar:
    st.subheader("📘 Instrucciones")
    st.write("""
    1. Presiona el botón **Toma una Foto**.  
    2. Espera que la IA procese la imagen.  
    3. Mira el resultado debajo.
    """)

# --- Entrada de cámara ---
img_file_buffer = st.camera_input("📸 Toma una Foto para analizar")

# --- Procesamiento de imagen ---
if img_file_buffer is not None:
    img = Image.open(img_file_buffer)

    # Redimensionar la imagen
    newsize = (224, 224)
    img = img.resize(newsize)
    
    # Convertir a array y normalizar
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    # Cargar en el array del modelo
    data[0] = normalized_image_array

    # --- Predicción ---
    prediction = model.predict(data)

    # --- Mostrar resultados ---
    st.markdown("<h3 style='color:#306998;'>📊 Resultado del reconocimiento:</h3>", unsafe_allow_html=True)
    st.image(img, caption="Imagen procesada", width=300)

    if prediction[0][0] > 0.5:
        st.success(f"🟢 **Clase detectada:** Izquierda (probabilidad: {prediction[0][0]:.2f})")
    elif prediction[0][1] > 0.5:
        st.success(f"🔵 **Clase detectada:** Arriba (probabilidad: {prediction[0][1]:.2f})")
    else:
        st.warning("⚠️ No se detectó ninguna clase con alta probabilidad.")

    st.markdown("<hr style='border: 1px solid #DDD;'>", unsafe_allow_html=True)
else:
    st.info("👆 Toma una foto para comenzar el reconocimiento.")

# --- Pie de página ---
st.markdown("""
    <p style="text-align:center; color:#888; font-size:13px;">
    Creado con ❤️ usando Streamlit, OpenCV y Keras.
    </p>
""", unsafe_allow_html=True)




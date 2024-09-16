import av
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página de Streamlit
st.set_page_config(page_title="Detector de Emociones", page_icon=":smiley:", layout="wide")

# Título y descripción de la aplicación
st.title('Detector de Emociones en Tiempo Real')
st.write("Este es un detector de emociones en tiempo real que utiliza un modelo preentrenado.")

# Cargar modelos
@st.cache_resource
def load_models():
    prototxtPath = "models/deploy.prototxt"
    weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
    modelPath = "models/modelFEC.h5"

    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath) or not os.path.exists(modelPath):
        st.error("Error: Archivo de modelo o clasificador no encontrado.")
        st.stop()

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    emotionModel = load_model(modelPath)
    return faceNet, emotionModel

faceNet, emotionModel = load_models()

# Tipos de emociones y colores para la gráfica
classes = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black']

# Función para predecir la emoción
def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)
            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))
            pred = emotionModel.predict(face2)
            preds.append(pred[0])
    return (locs, preds)

# Clase para el procesamiento de video con webrtc
class EmotionDetector(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = imutils.resize(img, width=640)
        (locs, preds) = predict_emotion(img, faceNet, emotionModel)

        preds_sum = [0] * len(classes)

        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(pred) * 100)
            cv2.rectangle(img, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
            cv2.putText(img, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(img, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

            preds_sum = [sum(x) for x in zip(preds_sum, pred)]

        # Normalizar las predicciones
        total = sum(preds_sum)
        if total > 0:
            preds_sum = [x / total for x in preds_sum]

        # Actualiza la variable de sesión con los últimos datos de predicción
        st.session_state['preds'] = preds_sum
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Crear dos columnas: una para el video y otra para el gráfico de barras
col1, col2 = st.columns(2)

with col1:
    # Aquí va la parte del video
    ctx = webrtc_streamer(
        key="example", 
        video_processor_factory=EmotionDetector,
        mode=WebRtcMode.SENDRECV, 
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    # Placeholder para el gráfico
    chart_placeholder = st.empty()

    # Placeholder para el registro de datos
    log_placeholder = st.empty()

# Función para actualizar la interfaz
def update_ui():
    if 'preds' in st.session_state:
        # Actualizar gráfico
        fig, ax = plt.subplots()
        ax.bar(range(len(classes)), st.session_state['preds'], color=colors, tick_label=classes)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        chart_placeholder.pyplot(fig)
        plt.close(fig)

        # Actualizar registro de datos
        log_text = "Últimas predicciones:\n"
        for emotion, value in zip(classes, st.session_state['preds']):
            log_text += f"{emotion}: {value:.2f}\n"
        log_placeholder.text_area("Registro en tiempo real:", log_text, height=200)

# Botón para actualizar manualmente
if st.button('Actualizar datos'):
    update_ui()

# Actualización automática
if ctx.state.playing:
    update_ui()

# Información adicional
st.sidebar.markdown('---')
st.sidebar.subheader('Creado por:')
st.sidebar.markdown('Alexander Oviedo Fadul')
st.sidebar.markdown("[GitHub](https://github.com/bladealex9848) | [Website](https://alexanderoviedofadul.dev/)")
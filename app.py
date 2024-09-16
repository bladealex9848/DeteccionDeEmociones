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
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Detector de Emociones",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state='collapsed',
)

# Título y descripción de la aplicación
st.title('Detector de Emociones en Tiempo Real')
st.write("""
    [![ver código fuente](https://img.shields.io/badge/Repositorio%20GitHub-gris?logo=github)](https://github.com/bladealex9848/DeteccionDeEmociones)
    ![Visitantes](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Femovision.streamlit.app&label=Visitantes&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
    """)
st.write("Este es un detector de emociones en tiempo real que utiliza un modelo preentrenado.")

# Verifica si los archivos del modelo y clasificador existen
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

# Función para predecir la emoción
def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
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

# Tipos de emociones y colores para la gráfica
classes = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black']

# Inicializa una variable de sesión para almacenar los datos del gráfico
if 'preds' not in st.session_state:
    st.session_state['preds'] = [0] * len(classes)

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
        
        # Registrar las predicciones
        logger.info(f"Predicciones actualizadas: {preds_sum}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

use_local_camera = st.checkbox("Usar cámara local (solo para pruebas locales)")

if use_local_camera:
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("Error: No se pudo acceder a la cámara web.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        frame_placeholder = st.empty()
    with col2:
        figura_placeholder = st.empty()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                st.error("Error: No se pudo leer el cuadro de la cámara web.")
                break
            frame = imutils.resize(frame, width=640)
            (locs, preds) = predict_emotion(frame, faceNet, emotionModel)
            
            preds_sum = [0] * len(classes)
            for (box, pred) in zip(locs, preds):
                (Xi, Yi, Xf, Yf) = box
                label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(pred) * 100)
                cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
                cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)
                preds_sum = [sum(x) for x in zip(preds_sum, pred)]
            
            # Normalizar las predicciones
            total = sum(preds_sum)
            if total > 0:
                preds_sum = [x / total for x in preds_sum]
            
            # Actualizar el marcador de posición con la nueva imagen
            frame_placeholder.image(frame, channels="BGR")

            # Actualizar gráfica
            fig, ax = plt.subplots()
            ax.bar(range(len(classes)), preds_sum, color=colors, tick_label=classes)
            ax.set_ylim([0, 1])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            figura_placeholder.pyplot(fig)
            plt.close(fig)

            # Registrar las predicciones
            logger.info(f"Predicciones (cámara local): {preds_sum}")

    except Exception as e:
        st.error(f"Ha ocurrido un error: {e}")
        logger.error(f"Error en el modo de cámara local: {e}")

    finally:
        cam.release()
        cv2.destroyAllWindows()
else:
    # Define dos columnas: una para el video y otra para el gráfico de barras
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
        # Dibuja el gráfico de barras utilizando los datos almacenados en la variable de sesión
        figura_placeholder = st.empty()
        
        def update_chart():
            if 'preds' in st.session_state:
                fig, ax = plt.subplots()
                ax.bar(range(len(classes)), st.session_state['preds'], color=colors, tick_label=classes)
                ax.set_ylim([0, 1])
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                figura_placeholder.pyplot(fig)
                plt.close(fig)
                logger.info(f"Gráfico actualizado con predicciones: {st.session_state['preds']}")

        if ctx.state.playing:
            st.button("Actualizar gráfico", on_click=update_chart)
            update_chart()  # Actualizar el gráfico inicialmente

st.sidebar.markdown('---')
st.sidebar.subheader('Creado por:')
st.sidebar.markdown('Alexander Oviedo Fadul')
st.sidebar.markdown("[GitHub](https://github.com/bladealex9848) | [Website](https://alexanderoviedofadul.dev/) | [LinkedIn](https://www.linkedin.com/in/alexander-oviedo-fadul/) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!%20)")

# Sección de registro de información (contraída por defecto)
with st.expander("Registro de Información", expanded=False):
    if st.button("Mostrar últimos registros"):
        st.text("Últimos registros:")
        log_entries = logger.handlers[0].stream.getvalue().split('\n')[-10:]
        for entry in log_entries:
            st.text(entry)
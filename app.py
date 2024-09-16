import av
import cv2
import numpy as np
import imutils
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import threading
import time

# Configuración de la página de Streamlit
st.set_page_config(page_title="Detector de Emociones", page_icon=":smiley:", layout="wide")

# Título y descripción de la aplicación
st.title('Detector de Emociones en Tiempo Real')
st.write("Este es un detector de emociones en tiempo real que utiliza un modelo preentrenado.")

# Verifica si los archivos del modelo y clasificador existen
@st.cache_resource
def load_models():
    prototxtPath = "models/deploy.prototxt"
    weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
    modelPath = "models/modelFEC.h5"

    for path in [prototxtPath, weightsPath, modelPath]:
        if not os.path.exists(path):
            st.error(f"Error: Archivo no encontrado: {path}")
            st.stop()

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    emotionModel = load_model(modelPath)
    return faceNet, emotionModel

faceNet, emotionModel = load_models()

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

# Tipos de emociones y colores para la gráfica
classes = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black']

# Inicializa variables de sesión
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

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Función para crear gráfico de Plotly
def create_plotly_chart():
    data = [go.Bar(x=classes, y=st.session_state['preds'], marker_color=colors)]
    layout = go.Layout(yaxis=dict(range=[0, 1]), title='Emociones Detectadas')
    fig = go.Figure(data=data, layout=layout)
    return fig

# Función para mostrar datos en texto
def show_text_data():
    for emotion, value in zip(classes, st.session_state['preds']):
        st.write(f"{emotion}: {value:.2f}")

# Función para actualizar la visualización
def update_visualization():
    while True:
        try:
            with chart_placeholder.container():
                st.plotly_chart(create_plotly_chart(), use_container_width=True)
        except Exception as e:
            st.error(f"Error al crear el gráfico: {e}")
            st.write("Mostrando datos en formato de texto como respaldo:")
            show_text_data()
        time.sleep(0.1)

# Crear dos columnas
col1, col2 = st.columns(2)

with col1:
    # Parte del video
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=EmotionDetector,
        mode=WebRtcMode.SENDRECV,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    # Placeholder para el gráfico o texto
    chart_placeholder = st.empty()

    # Iniciar hilo de actualización
    if ctx.state.playing:
        update_thread = threading.Thread(target=update_visualization)
        update_thread.daemon = True
        update_thread.start()

# Información adicional
st.sidebar.markdown('---')
st.sidebar.subheader('Creado por:')
st.sidebar.markdown('Alexander Oviedo Fadul')
st.sidebar.markdown("[GitHub](https://github.com/bladealex9848) | [Website](https://alexanderoviedofadul.dev/)")

# Sistema de recuperación ante errores
st.sidebar.markdown('---')
st.sidebar.subheader('Estado del Sistema')

if st.sidebar.button("Reiniciar Visualización"):
    st.experimental_rerun()

st.sidebar.info("Si no ves las estadísticas, intenta reiniciar la visualización o recargar la página.")
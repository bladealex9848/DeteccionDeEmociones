import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import streamlit as st

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Detector de Emociones",
    page_icon=":smiley:",
    layout="wide",  # Utiliza el ancho completo de la página
    initial_sidebar_state='collapsed',
    menu_items={
        'Get Help': 'https://www.alexanderoviedofadul.dev/',
        'Report a bug': None,
        'About': "Este es un detector de emociones en tiempo real que utiliza un modelo preentrenado."
    }
)

# Título y descripción de la aplicación
st.title('Detector de Emociones en Tiempo Real')
st.write("Este es un detector de emociones en tiempo real que utiliza un modelo preentrenado.")

# Verifica si los archivos del modelo y clasificador existen
prototxtPath = "models/deploy.prototxt"
weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
modelPath = "models/modelFEC.h5"

# Asegurarse de que los archivos del modelo existen
if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath) or not os.path.exists(modelPath):
    st.error("Error: Archivo de modelo o clasificador no encontrado.")
    st.stop()

# Cargamos el modelo de detección de rostros
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
emotionModel = load_model(modelPath)

# Se crea la captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    st.error("Error: No se pudo acceder a la cámara web.")
    st.stop()

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

# Crear marcadores de posición para la imagen y la gráfica
col1, col2 = st.columns(2)
with col1:
    frame_placeholder = st.empty()
with col2:
    figura_placeholder = st.empty()

# Bucle principal para procesar el video y predecir emociones
try:
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Error: No se pudo leer el cuadro de la cámara web.")
            break
        frame = imutils.resize(frame, width=640)
        (locs, preds) = predict_emotion(frame, faceNet, emotionModel)
        
        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            (angry, disgust, fear, happy, neutral, sad, surprise) = pred
            label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(angry, disgust, fear, happy, neutral, sad, surprise) * 100)
            cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
            cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)
            y = [angry, disgust, fear, happy, neutral, sad, surprise]
            
            # Actualizar el marcador de posición con la nueva imagen
            frame_placeholder.image(frame, channels="BGR")

            # Limpia y actualiza la figura en su marcador de posición
            figura1, ax = plt.subplots()  # Se define 'ax' aquí dentro del bucle
            ax.bar(range(len(classes)), [p for p in preds[0]], color=colors, tick_label=classes)
            ax.set_ylim([0, 1])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            figura_placeholder.pyplot(figura1)

except Exception as e:
    st.error(f"Ha ocurrido un error: {e}")

finally:
    # No olvides liberar la cámara y cerrar todas las ventanas de OpenCV al finalizar
    cam.release()
    cv2.destroyAllWindows()

# Footer
st.sidebar.markdown('---')
st.sidebar.subheader('Creado por:')
st.sidebar.markdown('Alexander Oviedo Fadul')
st.sidebar.markdown("[GitHub](https://github.com/bladealex9848) | [Website](https://www.alexanderoviedofadul.dev) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!%20)")
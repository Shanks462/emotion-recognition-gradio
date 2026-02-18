import gradio as gr
import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("emotion_model (2).keras")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def predict_emotion(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = image.reshape(1, 48, 48, 1)

    prediction = model.predict(image)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Emotion Recognition System",
)

interface.launch()

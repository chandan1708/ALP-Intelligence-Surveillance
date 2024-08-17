
import cv2, sys, numpy, os
import cv2
import numpy as np
import pickle

haar_file = 'haarcascade_frontalface_default (1).xml'
datasets = 'datasets'

# Load models
model_eigen = cv2.face.EigenFaceRecognizer_create()
model_fisher = cv2.face.FisherFaceRecognizer_create()
model_eigen.read('eigenface_model.xml')

# Try to load the Fisherface model
fisherface_model_loaded = True
try:
    model_fisher.read('fisherface_model.xml')
except cv2.error as e:
    print('Fisherface model not loaded:', e)
    fisherface_model_loaded = False

# Load the names dictionary
with open('names.pkl', 'rb') as f:
    names = pickle.load(f)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_file)

# Define width and height for resizing
(width, height) = (130, 100)

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        # Eigenface prediction
        prediction_eigen = model_eigen.predict(face_resize)
        if prediction_eigen[1] < 5000:
            label_text = f"Eigen: {names[prediction_eigen[0]]} - {prediction_eigen[1]:.0f}"
        else:
            label_text = "Eigen: Not recognized"

        # Fisherface prediction if the model is loaded
        if fisherface_model_loaded:
            prediction_fisher = model_fisher.predict(face_resize)
            if prediction_fisher[1] < 500:
                label_text += f" | Fisher: {names[prediction_fisher[0]]} - {prediction_fisher[1]:.0f}"
            else:
                label_text += " | Fisher: Not recognized"

        cv2.putText(im, label_text, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('OpenCV', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

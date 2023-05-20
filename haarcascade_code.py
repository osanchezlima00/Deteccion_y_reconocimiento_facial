import cv2
import os
import numpy as np

# imágenes de entrenamiento de la persona deseada
images_folder = 'database/train'
training_images = []

for filename in os.listdir(images_folder):
    img = cv2.imread(os.path.join(images_folder, filename))
    if img is not None:
        training_images.append(img)

# extraer características Utilizando el algoritmo de detección de rostros Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Crear una lista de características de las imágenes de entrenamiento
training_features = []

for img in training_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #faces = face_cascade.detectMultiScale(gray, 1.35, 1)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (100, 100))
        cv2.imshow("image",roi)
        # Espera a que el usuario presione una tecla
        cv2.waitKey(0)
        # Cierra todas las ventanas
        cv2.destroyAllWindows()

        training_features.append(roi.flatten())



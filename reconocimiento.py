import cv2
import os
import numpy as np

# Directorio de grabaciones
carpeta_grabaciones = 'grabacion'

# Crear un reconocedor de rostros usando LBPH (Local Binary Patterns Histograms)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cargar imágenes y entrenar el modelo
def entrenar_modelo(carpeta_grabaciones):
    imagenes = []
    etiquetas = []
    nombres = []
    etiqueta_actual = 0

    # Recorrer cada subcarpeta en la carpeta principal de grabaciones
    for nombre_carpeta in os.listdir(carpeta_grabaciones):
        ruta_carpeta = os.path.join(carpeta_grabaciones, nombre_carpeta)
        if not os.path.isdir(ruta_carpeta):
            continue

        nombres.append(nombre_carpeta)

        # Leer cada imagen en la subcarpeta
        for nombre_imagen in os.listdir(ruta_carpeta):
            ruta_imagen = os.path.join(ruta_carpeta, nombre_imagen)
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

            # Detectar rostros en la imagen
            detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            rostros = detector_rostros.detectMultiScale(imagen, scaleFactor=1.2, minNeighbors=8)

            for (x, y, w, h) in rostros:
                rostro = imagen[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (200, 200))  # Normaliza tamaño
                rostro = cv2.equalizeHist(rostro)         # Mejora el contraste
                imagenes.append(rostro)
                etiquetas.append(etiqueta_actual)

        etiqueta_actual += 1

    # Entrenar el reconocedor con las imágenes y etiquetas obtenidas
    recognizer.train(imagenes, np.array(etiquetas))

    return nombres


# Entrenar el modelo con las imágenes en la carpeta de grabaciones
nombres_usuarios = entrenar_modelo(carpeta_grabaciones)

# Iniciar la cámara
camara = cv2.VideoCapture(0)
detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = camara.read()
    if not ret:
        print("No es posible obtener la imagen")
        break

    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen en vivo
    rostros = detector_rostros.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in rostros:
        rostro_gris = gris[y:y+h, x:x+w]

        # Predecir la identidad usando el reconocedor LBPH
        etiqueta, confianza = recognizer.predict(rostro_gris)

        # Determinar el nombre si la confianza es aceptable
        if confianza < 50:  # Umbral para considerar una coincidencia
            nombre = nombres_usuarios[etiqueta]
        else:
            nombre = "Desconocido"

        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el video en vivo con los nombres
    cv2.imshow('Reconocimiento Facial', frame)

    # Presiona 'q' para salir del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
camara.release()
cv2.destroyAllWindows()

import cv2
from fer import FER


# Inicializar la cámara
camara = cv2.VideoCapture(0)

# Inicializar el detector de emociones
detector_emociones = FER(mtcnn=True)

# Diccionario para traducir emociones al español
emociones_traduccion = {
    "angry": "Enfadado",
    "disgust": "Disgustado",
    "fear": "Asustado",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Sorprendido",
    "neutral": "Normal"
}

while True:
    # Capturar un cuadro de la cámara
    ret, frame = camara.read()
    if not ret:
        break

    # Detectar la emoción en la imagen capturada
    resultados = detector_emociones.detect_emotions(frame)

    # Mostrar el resultado en la imagen
    for resultado in resultados:
        (x, y, ancho, alto) = resultado["box"]
        emocion = resultado["emotions"]
        emocion_predominante = max(emocion, key=emocion.get)

        # Traducir la emoción al español
        emocion_en_espanol = emociones_traduccion.get(emocion_predominante, emocion_predominante)

        # Dibujar un rectángulo alrededor de la cara detectada
        cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
        # Mostrar la emoción traducida sobre la imagen
        cv2.putText(frame, emocion_en_espanol, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el video en vivo
    cv2.imshow('Deteccion de Emociones', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar la cámara y las ventanas
camara.release()
cv2.destroyAllWindows()

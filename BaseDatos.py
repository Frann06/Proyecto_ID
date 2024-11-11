import cv2
import os

# Pedir el nombre de la persona desde la consola
nombre_usuario = input("Introduce el nombre de la persona: ")

# Crear la carpeta principal 'grabacion' si no existe
carpeta_principal = 'grabacion'
if not os.path.exists(carpeta_principal):
    os.makedirs(carpeta_principal)

# Crear una subcarpeta con el nombre del usuario dentro de 'grabacion'
carpeta_usuario = os.path.join(carpeta_principal, nombre_usuario)
if not os.path.exists(carpeta_usuario):
    os.makedirs(carpeta_usuario)

# Número de fotogramas a capturar
numero_fotograma = 0
numero_fotogramas = 50

# Variable para iniciar la grabación
grabando = False

# Iniciar la cámara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No es posible abrir la cámara")
    exit()

while numero_fotograma < numero_fotogramas:
    ret, frame = camara.read()
    if not ret:
        print("No es posible obtener la imagen")
        break

    # Mostrar la imagen en la ventana 'webcam'
    cv2.imshow('webcam', frame)

    # Iniciar grabación cuando se presiona la barra espaciadora (' ')
    if cv2.waitKey(100) == ord(' '):
        grabando = True
        print("Inicio grabación")

    # Si está grabando, guardar los fotogramas en la carpeta del usuario
    if grabando:
        ruta_fotograma = os.path.join(carpeta_usuario, f'fotograma{numero_fotograma}.jpg')
        print(f'Grabando {ruta_fotograma}')
        cv2.imwrite(ruta_fotograma, frame)
        numero_fotograma += 1

print("Fin grabación")

# Liberar la cámara y cerrar ventanas
camara.release()
cv2.destroyAllWindows()

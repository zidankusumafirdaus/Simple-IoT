import cv2 #Proses Gambar dan Video
import mediapipe as mp #Deteksi Objek

# Inisialisation MediaPipe
mp_drawing = mp.solutions.drawing_utils #menggambar landmark tangan pada gambar.
mp_hands = mp.solutions.hands #mendeteksi tangan.

cap = cv2.VideoCapture(0) #Kamera
if not cap.isOpened():
    print("Error: Kamera tidak tersedia.")
    exit()

hands = mp_hands.Hands() #mendeteksi tangan dalam gambar.
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, image = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Mendapatkan label tangan (kiri/kanan)
            hand_label = handedness.classification[0].label
            
            #Mendapatkan posisi landmark
            #Menampilkan label tangan di gambar
            label_position = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = image.shape

            #Menghitung koordinat x dan y dari posisi tersebut berdasarkan ukuran gambar.
            cx, cy = int(label_position.x * w), int(label_position.y * h)

            (text_width, text_height), _ = cv2.getTextSize(hand_label, font, 0.5, 2)
            # Calculate the x-coordinate to horizontally center the text
            x = int(cx - text_width / 2)
            cv2.putText(image, hand_label, (x, cy - -50), font, 0.5, (900, 255, 0), 2)

            # Menggambar landmark tangan
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    cv2.imshow('Hand Tracker', image)

    if cv2.waitKey(1) & 0xFF == ord('k'):
        break
cap.release()
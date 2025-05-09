import cv2
import mediapipe as mp
import os  # Untuk menjalankan perintah sistem membuka atau menutup aplikasi

# Inisialisasi
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Deteksi 1 tangan saja
mp_draw = mp.solutions.drawing_utils

# Ujung jari yang akan dicek (jempol, telunjuk, tengah, manis, kelingking)
finger_tips = [4, 8, 12, 16, 20]

# Fungsi untuk membuka atau menutup Chrome
def open_chrome():
    os.system("start chrome")  # Perintah untuk membuka Chrome

def close_chrome():
    os.system("taskkill /f /im chrome.exe")  # Perintah untuk menutup Chrome

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Bikin mirror + convert ke RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    count = 0
    gesture_detected = None  # Menyimpan gesture yang terdeteksi

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm = hand_landmark.landmark
            h, w, _ = frame.shape

            # Hitung jari (jempol pengecualian karena horizontal)
            if lm[finger_tips[0]].x < lm[finger_tips[0] - 1].x:
                count += 1

            # Cek 4 jari lainnya (vertikal)
            for tip in finger_tips[1:]:
                if lm[tip].y < lm[tip - 2].y:
                    count += 1

            # Cek Gesture untuk membuka atau menutup Chrome
            thumb_tip = lm[4]
            index_tip = lm[8]

            # Gesture Thumb + Index (menyentuh atau mendekat) untuk membuka Chrome
            if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
                gesture_detected = "Open Chrome"
                open_chrome()

            # Gesture Thumb + Index berjauhan untuk menutup Chrome
            elif abs(thumb_tip.x - index_tip.x) > 0.15 and abs(thumb_tip.y - index_tip.y) > 0.15:
                gesture_detected = "Close Chrome"
                close_chrome()

            # Gambar titik tangan
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # Tampilkan jumlah jari dan gesture yang terdeteksi
    cv2.putText(frame, f"Gesture: {gesture_detected or 'No Gesture'}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Fixcode Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

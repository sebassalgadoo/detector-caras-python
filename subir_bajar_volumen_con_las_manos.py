import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            x1, y1 = int(lm[4].x * frame.shape[1]), int(lm[4].y * frame.shape[0])
            x2, y2 = int(lm[8].x * frame.shape[1]), int(lm[8].y * frame.shape[0])
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            distancia = math.hypot(x2 - x1, y2 - y1)
            vol = int(distancia / 150 * 100)
            vol = max(0, min(100, vol))
            volume.SetMasterVolumeLevelScalar(vol / 100, None)

    cv2.imshow("Detector de manos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
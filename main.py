import cv2
import time
import imutils
import numpy as np
import mediapipe as mp
from imutils.video import FPS, VideoStream
import pyfiglet
from colorama import init, Fore, Style
import warnings
warnings.filterwarnings("ignore")
init(autoreset=True)
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

print('[Status] Boubker The GOAT Loading...')
print('[Status] Starting Video Stream...')
ascii_banner = pyfiglet.figlet_format("FACE MAPPING")
ascii_banner2 = pyfiglet.figlet_format("BY BOUBKER")
print(Fore.RED + ascii_banner)
print(Fore.BLUE + ascii_banner2)
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


while True:
    warnings.filterwarnings("ignore")
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    landmarks_frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result_face_mesh = face_mesh.process(frame_rgb)

    detected_objects = []
    if result_face_mesh.multi_face_landmarks:
        for face_landmarks in result_face_mesh.multi_face_landmarks:

            x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            face_gray = frame_gray[y_min:y_max, x_min:x_max]
            face_color = landmarks_frame[y_min:y_max, x_min:x_max]

            smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(face_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
                detected_objects.append("Smile Detected")

            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(landmarks_frame, (x, y), 4, (0, 255, 0), -1)
            
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))


    result_hands = hands.process(frame_rgb)
    
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))


            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(landmarks_frame, (x, y), 4, (0, 255, 0), -1)


    if "Smile Detected" in detected_objects:
        cv2.putText(landmarks_frame, "Smile Detected!", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


    cv2.imshow("Frame", frame)
    cv2.imshow("Landmarks Only", landmarks_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()

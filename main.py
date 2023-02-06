import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(
    color = (0, 255, 0),
    thickness = 2,
    circle_radius = 0.5
)

cam_device_id = 0

cam = cv2.VideoCapture(cam_device_id)

# Initiating Holistic Model

# For static images:
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cam.isOpened():
        ret, frame = cam.read()
        
        # Recoloring         
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detection
        result = holistic.process(img)
        
        # Recoloring back
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Drawing Face landmarks
        """
            To Draw we Have:
                face_landmarks,
                pose_landmarks,
                left_hand_landmarks,
                right_hand_landmarks
        """
        mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        cv2.imshow("AI Full Body Detections", img)
        # Press Escape (Esc) Button to Stop    
        if cv2.waitKey(10) & 0xFF == 27:
            break

cam.release()
cv2.destroyAllWindows()
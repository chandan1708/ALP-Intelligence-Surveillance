import math
from email.mime import image
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt


mp_pose=mp.solutions.pose
pose =mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3,model_complexity=2)
mp_drawing=mp.solutions.drawing_utils

import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    results = pose.process(image)
    

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    

    cv2.imshow('Pose Estimation', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

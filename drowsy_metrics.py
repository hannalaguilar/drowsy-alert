import numpy as np


# Function to calculate distances between two landmarks
def calculate_distance(landmark1, landmark2, frame_shape):
    height, width, _ = frame_shape
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye_landmarks, frame_shape):
    vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5], frame_shape)
    vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4], frame_shape)
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3], frame_shape)
    return (vertical1 + vertical2) / (2.0 * horizontal)


# Function to calculate mouth aspect ratio (MAR)
def calculate_mouth_openness(landmarks, frame_shape):
    vertical = calculate_distance(landmarks[13], landmarks[14], frame_shape)
    horizontal = calculate_distance(landmarks[78], landmarks[308], frame_shape)
    return vertical / horizontal


def calculate_fainted_chin(landmarks, frame_shape):
    chin = landmarks[152]
    nose = landmarks[1]
    height, width, _ = frame_shape
    chin_y = chin.y * height
    nose_y = nose.y * height
    drop_ratio = (chin_y - nose_y) / height
    return drop_ratio

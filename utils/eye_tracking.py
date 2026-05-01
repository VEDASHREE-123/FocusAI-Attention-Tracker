import cv2
import numpy as np

# Mediapipe landmark indices for left and right eyes
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_indices, image_width, image_height):
    """
    Calculates Eye Aspect Ratio (EAR) given landmarks.
    """
    points = []
    for idx in eye_indices:
        landmark = landmarks[idx]
        points.append([landmark.x * image_width, landmark.y * image_height])
    
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    p5 = points[4]
    p6 = points[5]
    
    vertical_1 = get_distance(p2, p6)
    vertical_2 = get_distance(p3, p5)
    horizontal = get_distance(p1, p4)
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0
    return ear

def check_eyes_closed(landmarks, image_width, image_height, ear_threshold=0.22):
    """
    Checks if eyes are closed based on EAR threshold.
    Returns boolean (True if closed) and the average EAR.
    """
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_INDICES, image_width, image_height)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES, image_width, image_height)
    
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear < ear_threshold, avg_ear

def get_head_pose(landmarks, image_width, image_height):
    """
    Estimates head pose to determine if the user is looking away.
    Returns a boolean for looking away, direction string, and X, Y angles.
    """
    # Using a robust 2D heuristic based on facial landmark ratios
    # Left eye outer (33), Right eye outer (263), Nose tip (1)
    # Top of head (10), Bottom of chin (152)
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    top = landmarks[10]
    bottom = landmarks[152]
    
    # Calculate yaw ratio
    # 0.5 is perfectly straight. < 0.4 is left, > 0.6 is right
    eye_width = right_eye.x - left_eye.x
    if eye_width == 0:
        yaw_ratio = 0.5
    else:
        yaw_ratio = (nose.x - left_eye.x) / eye_width
        
    # Calculate pitch ratio
    # 0.5 is perfectly straight. < 0.4 is up, > 0.7 is down
    face_height = bottom.y - top.y
    if face_height == 0:
        pitch_ratio = 0.5
    else:
        pitch_ratio = (nose.y - top.y) / face_height
    
    is_looking_away = False
    direction = "Center"
    
    if yaw_ratio < 0.35:
        direction = "Right"  # Mirrored
        is_looking_away = True
    elif yaw_ratio > 0.65:
        direction = "Left"   # Mirrored
        is_looking_away = True
    elif pitch_ratio < 0.35:
        direction = "Up"
        is_looking_away = True
    elif pitch_ratio > 0.70:
        direction = "Down"
        is_looking_away = True
        
    # Returning ratios as x_angle and y_angle for logging
    x_angle = pitch_ratio * 100
    y_angle = yaw_ratio * 100
        
    return is_looking_away, direction, x_angle, y_angle

def check_yawn(landmarks, image_width, image_height, mar_threshold=0.5):
    """
    Checks if the user is yawning using Mouth Aspect Ratio (MAR).
    """
    # Outer lip landmarks
    left = [landmarks[61].x * image_width, landmarks[61].y * image_height]
    right = [landmarks[291].x * image_width, landmarks[291].y * image_height]
    top = [landmarks[13].x * image_width, landmarks[13].y * image_height]
    bottom = [landmarks[14].x * image_width, landmarks[14].y * image_height]
    
    horizontal = get_distance(left, right)
    vertical = get_distance(top, bottom)
    
    mar = vertical / horizontal if horizontal != 0 else 0
    return mar > mar_threshold, mar

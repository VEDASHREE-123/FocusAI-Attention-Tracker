import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

def get_face_mesh_model():
    """Initializes and returns the Mediapipe Face Landmarker model."""
    # Ensure the model exists
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        import urllib.request
        print("Downloading face landmarker model...")
        urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=5)
    
    return vision.FaceLandmarker.create_from_options(options)

def extract_landmarks(image, face_landmarker):
    """
    Extracts face landmarks from the given image using the Face Landmarker.
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect
    results = face_landmarker.detect(mp_image)
    return results

def draw_landmarks(image, results):
    """
    Draws simple points for the face mesh since drawing_utils might be unavailable.
    """
    annotated_image = image.copy()
    h, w, _ = annotated_image.shape
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)
    return annotated_image

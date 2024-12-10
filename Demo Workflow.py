import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load RGB and Depth Sample Data
def load_sample_data(rgb_path, depth_path):
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    return rgb_image, depth_image

# Human Detection
def segment_human(depth_image):
    _, segmented = cv2.threshold(depth_image, 50, 255, cv2.THRESH_BINARY)
    return segmented

# Calculate Distance from Sensor
def calculate_distance(depth_image):
    valid_pixels = depth_image[depth_image > 0]
    if len(valid_pixels) == 0:
        return float('inf')  # No valid data
    return np.mean(valid_pixels)

# Face Detection using Viola-Jones
def detect_faces(rgb_image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Gesture Detection for Moving Hand Backward
def detect_gesture(prev_depth, curr_depth, roi):
    x, y, w, h = roi
    prev_roi = prev_depth[y:y+h, x:x+w]
    curr_roi = curr_depth[y:y+h, x:x+w]
    
    # Calculate average depth in the region of interest
    prev_depth_avg = np.mean(prev_roi[prev_roi > 0]) if np.any(prev_roi > 0) else float('inf')
    curr_depth_avg = np.mean(curr_roi[curr_roi > 0]) if np.any(curr_roi > 0) else float('inf')
    
    # Check if the hand moved backward (depth increased significantly)
    return curr_depth_avg > prev_depth_avg + 10  # Adjust threshold as needed

# SURF Feature Extraction
def extract_surf_features(image):
    surf = cv2.SIFT_create()  # Use SIFT (OpenCV 4+ replacement for SURF)
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return keypoints, descriptors

# KLT Tracking
def track_features_klt(prev_image, curr_image, keypoints):
    points_prev = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    points_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_image, curr_image, points_prev, None)
    return points_next[status == 1]

# Morph Detected Face into Avatar
def morph_face(rgb_image, face_coords):
    x, y, w, h = face_coords
    face_region = rgb_image[y:y+h, x:x+w]
    avatar = np.full_like(face_region, (255, 192, 203))  # Pink for avatar
    blended = cv2.addWeighted(face_region, 0.5, avatar, 0.5, 0)
    rgb_image[y:y+h, x:x+w] = blended
    return rgb_image

# CNN Model for Face Recognition
def cnn_model(input_shape, num_classes=4):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Pipeline Execution
def main_pipeline(rgb_path, depth_path, prev_depth_path=None):
    rgb_image, depth_image = load_sample_data(rgb_path, depth_path)
    prev_depth_image = cv2.imread(prev_depth_path, cv2.IMREAD_GRAYSCALE) if prev_depth_path else None

    # Step 1: Human Detection and Distance Calculation
    segmented_human = segment_human(depth_image)
    distance = calculate_distance(depth_image)
    if distance > 2.5:
        print("Please move closer to the sensor.")
        return
    else:
        print("Subject is within range. Proceeding...")

    # Step 2: Face Detection
    faces = detect_faces(rgb_image)
    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        # Step 3: Background Preprocessing
        face_rgb = rgb_image[y:y+h, x:x+w]

        # Step 4: Gesture Recognition
        if prev_depth_image is not None:
            gesture_detected = detect_gesture(prev_depth_image, depth_image, (x, y, w, h))
            if not gesture_detected:
                print("Gesture not recognized. Please perform the proper gesture.")
                continue

        # Step 5: Feature Extraction and Tracking
        keypoints, descriptors = extract_surf_features(face_rgb)
        if keypoints is None or descriptors is None:
            print("No keypoints detected for this face.")
            continue

        if prev_depth_image is not None:
            tracked_features = track_features_klt(rgb_image, rgb_image, keypoints)

        # Step 6: Augmented Reality (Face Morphing)
        rgb_image = morph_face(rgb_image, (x, y, w, h))

        # Step 7: CNN for Recognition (Placeholder for Real Training)
        cnn = cnn_model((128, 128, 3))
        print("Face recognition model ready for training.")

        # Display Results
        cv2.imshow("Morphed Image", rgb_image)
        cv2.waitKey(0)

    print("Therapy estimation complete.")

# Run the pipeline
main_pipeline("data/rgb/sample_rgb.jpg", "data/depth/sample_depth.jpg", prev_depth_path="data/depth/prev_depth.jpg")

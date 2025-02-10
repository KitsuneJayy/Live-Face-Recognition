import threading
import cv2
import dlib
import numpy as np
import os

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Thread safety variables
recognized_name = "Unknown"
face_match = False
face_match_lock = threading.Lock()

# Load reference faces
def load_reference_faces(folder_path):
    reference_faces = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = detector(img, 1)  # Detect faces directly
            for face in faces:
                landmarks = predictor(img, face)
                face_descriptor = face_rec_model.compute_face_descriptor(img, landmarks)
                reference_faces[filename.split('.')[0]] = np.array(face_descriptor)
    return reference_faces

reference_folder = 'reference_images'
reference_faces = load_reference_faces(reference_folder)

if not reference_faces:
    print("No reference faces found in the folder!")
    exit()

# Face recognition function
def check_face(frame):
    global face_match, recognized_name
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    with face_match_lock:
        recognized_name = "Unknown"
        face_match = False

        for face in faces:
            landmarks = predictor(frame, face)
            face_descriptor = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))

            # Compare with reference faces
            for name, reference_descriptor in reference_faces.items():
                if np.linalg.norm(reference_descriptor - face_descriptor) < 0.4:
                    recognized_name = name
                    face_match = True
                    return  # Stop checking once a match is found

# Background thread for face detection
import time

def detection_thread():
    global recognized_name
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        check_face(frame)
        time.sleep(0.1)  # Use sleep instead of cv2.waitKey to reduce CPU usage

# Start detection thread
threading.Thread(target=detection_thread, daemon=True).start()

# Main loop for video display
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with face_match_lock:
            text = f"Matched: {recognized_name}" if face_match else "Unknown"
            color = (0, 255, 0) if face_match else (0, 0, 255)
            cv2.putText(frame, text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted. Exiting gracefully...")
finally:
    cap.release()
    cv2.destroyAllWindows()
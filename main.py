import threading
import cv2
import dlib
import numpy as np
import os

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
face_match_lock = threading.Lock()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  

def load_reference_faces(folder_path):
    reference_faces = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                face_descriptor = face_rec_model.compute_face_descriptor(img, landmarks)
                reference_faces[filename.split('.')[0]] = np.array(face_descriptor)  # Save descriptor by person's name
    return reference_faces

reference_folder = 'reference_images' 
reference_faces = load_reference_faces(reference_folder)

if not reference_faces:
    print("No reference faces found in the folder!")
    exit()

recognized_name = "Unknown"

def check_face(frame):
    global face_match
    global recognized_name
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Enhance contrast in low-light conditions

        faces = detector(gray, 1)  # Use 1 for upsampling for better accuracy

        if len(faces) > 0:
            for face in faces:
                landmarks = predictor(gray, face)
                face_descriptor = face_rec_model.compute_face_descriptor(frame, landmarks)
                face_descriptor = np.array(face_descriptor)

                # Compare embeddings with reference faces
                face_match = False
                recognized_name = "Unknown"
                for name, reference_descriptor in reference_faces.items():
                    distance = np.linalg.norm(reference_descriptor - face_descriptor)
                    if distance < 0.3:  # Lower threshold for more precision
                        face_match = True
                        recognized_name = name
                        break
        else:
            face_match = False
            recognized_name = "Unknown"
    except Exception as e:
        face_match = False
        recognized_name = "Error"
try:
    while True:
        ret, frame = cap.read()  
        print("Frame captured:", ret)

        if ret:
            # Run face check every 30 frames 
            if counter % 30 == 0:
                try:
                    threading.Thread(target=check_face, args=(frame.copy(),)).start()
                except ValueError:
                    pass
            counter += 1

            with face_match_lock:
                if recognized_name == "Unknown":
                    cv2.putText(frame, "Unknown", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"Matched: {recognized_name}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            cv2.imshow('frame', frame) 

        key = cv2.waitKey(1)
        if key == ord('q'):  
            break

except KeyboardInterrupt:
    print("\nProgram interrupted. Exiting gracefully...")
finally:
    cap.release()
    cv2.destroyAllWindows()
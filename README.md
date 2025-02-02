# Live Face Recognition

This project uses OpenCV and Dlib to perform live face recognition using your webcam. The system compares faces in real-time against a set of reference images, identifying or matching known faces from a folder. If no match is found, it will display "Unknown" in red text. 

## Features

- Real-time face detection using Dlib.
- Face recognition by comparing embeddings against reference images.
- Text overlay with recognition status (green for recognized faces, red for unknown).
- Support for adding multiple reference faces via images stored in a folder.

## Requirements

- Python 3.6 or later
- OpenCV
- Dlib
- Numpy

## Installation

1.  Clone this repository:
2.	Set up a virtual environment: ```python3 -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate```
3.	Install required dependencies: ```pip install -r requirements.txt```
4.	Download the necessary Dlib models:
  •	Download shape_predictor_68_face_landmarks.dat and place it in the project directory.
  •	Download dlib_face_recognition_resnet_model_v1.dat and place it in the project directory.
5.	Place your reference images (e.g., person1.jpg, person2.jpg) in a folder named reference_images/. The system will use these images to recognize faces.

   Contributing

Feel free to fork this repository, make changes, and submit pull requests. Contributions are welcome!

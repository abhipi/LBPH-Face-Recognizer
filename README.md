# LBPH based Face Recognizer

This repository provides code and resources for building a simple face recognition system in Python using OpenCV. The system allows for capturing face images, training a model on a dataset of faces, and using the trained model to recognize faces in real-time video frames.


## Features

- Capture face images using the face_detection.py script to build a dataset.
- Train a face recognition model using the Local Binary Patterns Histograms algorithm with the face_recognition.py script.
- Predict and recognize faces in real-time video frames from a webcam using the predict_face_recognizer.py script.
- Display predicted labels and confidence percentages for recognized faces.



## Usage

Follow these steps to use this face recognition system:

1. Capture face images:
    * Run the face_detection.py script.
    * Specify the location of the dataset folder in the script.
    * This script will capture face images from a webcam and save them as cropped images in the dataset folder.
2. Train the face recognition model:
    * Run the face_recognition.py script.
    * Specify the location for the trained model YAML file in the script.
    * This script will train the face recognition model using the captured face images and save the model to the specified file.
3. Perform face recognition:
    * Run the predict_face_recognizer.py script.
    * Provide the location of the trained model YAML file and the Haar cascade XML file in the script.
4. Connect a webcam to your system.
    * This script will recognize faces in real-time video frames from the webcam and display the predicted labels and confidence percentages.


## Folder Structure
- face_detection.py: This script captures face images and builds a dataset. It uses OpenCV's face detection algorithm to detect faces in video frames from a webcam. Detected faces are saved as cropped images in a specified folder.

- face_recognition.py: This script trains a face recognition model. It uses the Local Binary Patterns Histograms algorithm to train the model on the dataset of face images. The trained model is saved in a YAML file.

- predict_face_recognizer.py: This script implements the face recognition system. It loads the pre-trained model and recognizes faces in real-time video frames from a webcam. Recognized faces are displayed with rectangles, along with their predicted labels and confidence percentages.

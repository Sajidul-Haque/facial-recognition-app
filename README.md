"""
author: Sajidul Haque
name: FacialRecognitionApp.py
desc: This Python program implements a face detection application using OpenCV, tkinter, Pillow, facenet_pytorch, and deepface libraries.
    It utilizes a webcam to capture images, detects faces using MTCNN and OpenCV's Haar cascade classifier, analyzes the captured images
    using DeepFace for age, gender, race, and emotion estimation, and displays the results on a GUI built with tkinter. Additionally, it includes
    functionalities for updating the video feed, saving captured images, and updating labels with analysis results.

input:
    - Webcam input for capturing images.
    - Various image processing operations such as face detection and analysis.

output:
    - GUI display of video feed with face detection boxes.
    - Analysis results including estimated age, gender, race, and emotion displayed on the GUI.
    - Saved images in the 'cache' directory.
    - Analysis results stored in a Pandas DataFrame.
"""

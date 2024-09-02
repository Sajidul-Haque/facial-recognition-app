import cv2
import tkinter as tk
from PIL import Image, ImageTk
from facenet_pytorch import MTCNN
import imutils
import os
from datetime import datetime
from deepface import DeepFace
import pandas as pd

# Initialize the MTCNN detector
mtcnn = MTCNN()

# Load the pre-trained face classifier from OpenCV
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Directory to save captured images
save_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(save_dir, 'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Initialize Pandas DataFrame for storing analysis results
results_df = pd.DataFrame(columns=['Image', 'Age', 'Gender', 'Race', 'Emotion'], data=[])

def detect_faces(img):
    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(img)

    # Draw rectangles around the detected faces
    if boxes is not None:
        for box in boxes:
            (x, y, w, h) = box.astype(int)
            cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)

    return img, boxes

def capture_image():
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Save the image without the face detection boxes
    image_path = os.path.join(cache_dir, f"captured_image_{capture_image.counter}.png")
    cv2.imwrite(image_path, frame)
    print(f"Image saved: {image_path}")

    # Analyze the captured image
    result_dict = analyze_image(image_path)

    # Update GUI with analysis results
    update_gui(result_dict)

    # Increment counter for the next image
    capture_image.counter += 1

def analyze_image(image_path):
    try:
        # Analyze the image using DeepFace
        result = DeepFace.analyze(image_path)
        print("DeepFace result:", result)  # Debug statement

        # Extract relevant information
        result_dict = result[0]
        age = result_dict['age']
        gender = result_dict['dominant_gender']
        race = result_dict['dominant_race']
        emotion = result_dict['dominant_emotion']
        print(f"Age: {age}, Gender: {gender}, Race: {race}, Emotion: {emotion}")

        return result_dict
    except Exception as e:
        print(f"Error analyzing image '{image_path}': {str(e)}")
        return None

def update_gui(result):
    if result is not None:
        # Clear previous information
        age_label.config(text="")
        gender_label.config(text="")
        race_label.config(text="")
        emotion_label.config(text="")

        # Update with new information
        age_label.config(text=f"Estimate Age: {result['age']}")
        gender_label.config(text=f"Gender: {result['dominant_gender']}")
        race_label.config(text=f"Race: {result['dominant_race']}")
        emotion_label.config(text=f"Emotion: {result['dominant_emotion']}")
    else:
        # Clear labels
        age_label.config(text="Estimate Age: -")
        gender_label.config(text="Gender: -")
        race_label.config(text="Race: -")
        emotion_label.config(text="Emotion: -")

def update_video():
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=800)

    # Convert the frame from BGR to RGB (required by MTCNN)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    processed_frame, _ = detect_faces(rgb_frame)

    # Convert the processed frame to an ImageTk object
    img = Image.fromarray(processed_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule the next update
    video_label.after(10, update_video)

    # Update date on GUI
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_label.config(text=current_date)

# Create the main Tkinter window
root = tk.Tk()
root.title("Sajidul's Face Detection App")

# Create a label to display the video feed
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

# Create a button to capture images
capture_button = tk.Button(root, text="Estimate Age, Gender, Race, Emotion", command=capture_image)
capture_button.pack(padx=10, pady=10)

# Labels to display analysis results
age_label = tk.Label(root, text="")
age_label.pack(padx=10, pady=5)
gender_label = tk.Label(root, text="")
gender_label.pack(padx=10, pady=5)
race_label = tk.Label(root, text="")
race_label.pack(padx=10, pady=5)
emotion_label = tk.Label(root, text="")
emotion_label.pack(padx=10, pady=5)

# Label to display current date
date_label = tk.Label(root, text="")
date_label.pack(side=tk.RIGHT, padx=10, pady=10)

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize counter for captured images
capture_image.counter = 1

# Schedule the first update
update_video()

# Run the Tkinter event loop
root.mainloop()

# Release the webcam
cap.release()

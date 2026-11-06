# Workshop-2: Object detection using web camera
## Name: Sowmiya R
## Register Number: 212225040420

# Aim:
To access the laptop camera, capture an image, and detect objects present in the captured image using the YOLOv8 object detection model.

# Software Requirements
    -Anaconda
    -Jupyter Notebook
    -Python
    -OpenCV (cv2)
    -Ultralytics YOLOv8
    -Laptop Camera

# Procedure
1. Open Jupyter Notebook using Anaconda.
2. Access your laptop camera using OpenCV.
3. Wait for 5 seconds and capture an image.
4. Display the captured image.
5. Apply YOLOv8 object detection.
6. Display the detected image with bounding boxes and labels.

# Algorithm
```
       Laptop Camera
             |
             v
       Capture Image
             |
             v
       Display Image
             |
             v
      YOLOv8 Detection
             |
             v
   Identify Detected Objects
             |
             v
 Display Bounding Boxes
       and Labels
             |
             v
       Final Output
```

# Program
```
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load a more accurate YOLO model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open webcam.")

else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = model(
            frame,
            conf=0.60,
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Display webcam output
        cv2.imshow("Real-Time Object Detection", annotated_frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam
cap.release()
cv2.destroyAllWindows()
```

# Output
<img width="376" height="297" alt="Screenshot 2026-11-06 225208" src="https://github.com/user-attachments/assets/a612328c-15c6-4962-b0b1-3bbd740289d0" />

# Result
Thus, object detection was successfully performed using the YOLOv8 model by accessing the laptop camera, capturing images, and identifying objects with bounding boxes and labels across three different scenes.

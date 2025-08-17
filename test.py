import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO("D:\yoloFire\runs\detect\fire_detection\weights\best.pt")

# Test on a single image
img_path = "D:\yoloFire\fireimage.jpg"  # Replace with the path to a test image
img = cv2.imread(img_path)  # Read the image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for display

# Run inference (detect objects)
results = model(img_path)

# Since results is a list, access the first item
result = results[0]

# Print the results (labels, boxes, etc.)
print(result.verbose())  # Prints results like confidence, boxes, and class labels

# Show the image with the detected bounding boxes
result.show()  # Automatically draws bounding boxes on the image and shows it

# Optionally, you can save the image with bounding boxes if needed
output_img = result.save()  # Saves the image with bounding boxes
print(f"Result saved to: {output_img}")

import cv2
import numpy as np
import tensorflow as tf
from real_time_detection import litter

# Load the neural network
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph_coco.pb", "mask_rcnn_inception_v2_coco.pbtxt")

# Generate random colors
colors = np.random.randint(0, 255, (80, 3))

# print(colors)
# Read the input image
camera = cv2.VideoCapture(0)

while True:
    # Capture an image using the laptop camera
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture image")
        break

    # Save the image to a temporary file
    img_path = "s.jpg"
    cv2.imwrite(img_path, frame)
    break
img = cv2.imread("s.jpg")
height, width, _ = img.shape

# Create a black image
black_image = np.zeros((height, width, 3), np.int8)
black_image[:] = (100, 100, 0)

# Prepare the image for the neural network
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)

# Forward pass to get the bounding boxes and masks
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
detection_count = boxes.shape[2]

for i in range(detection_count):
    box = boxes[0, 0, i]
    class_id = int(box[1])
    score = box[2]
    if score < 0.5:
        continue

    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    roi = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape

    mask = masks[i, class_id]
    mask = cv2.resize(mask, (roi_width, roi_height))
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

    cv2.rectangle(img, (x, y), (x2, y2), (255, 255, 255), 3)

    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = colors[class_id]
    for cnt in contours:
        cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))

# Display the images
cv2.imshow("Image", img)
cv2.imshow("Mask image", black_image)
litter()
cv2.waitKey(0)
cv2.destroyAllWindows()

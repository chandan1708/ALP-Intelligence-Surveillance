import cv2
import numpy as np
import tensorflow as tf

def litter():
    # Load the model
    model_path = 'models/waste_detection_model.h5'
    model = tf.keras.models.load_model(model_path)

    # Class labels
    class_labels = {0: 'glass', 1: 'paper', 2: 'cardboard', 3: 'plastic', 4: 'metal', 5: 'trash'}


    def preprocess_frame(frame):
        img = cv2.resize(frame, (128, 128))  # Resize to match the input shape of the model
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img


    def predict_waste(frame, model):
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        class_idx = np.argmax(prediction)
        return class_labels[class_idx]


    # Load the image
    image_path = 'C:/Users/Dell/PycharmProjects/MaskRCNN/s.jpg'
    frame = cv2.imread(image_path)

    if frame is not None:
        # Make prediction whether the object is litter of not
        label = predict_waste(frame, model)
        if label in class_labels:
            print(f"Predicted waste: {label}")
        # Display the label on the frame
        cv2.putText(frame, "Throwing waste", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Waste Detection', frame)

        # Wait for 15 seconds or a key press, whichever comes first
        cv2.waitKey(15000)
        cv2.destroyAllWindows()
    else:
        print(f"Failed to load image from {image_path}")

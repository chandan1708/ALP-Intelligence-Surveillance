
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import os
import pickle

haar_file = 'haarcascade_frontalface_default (1).xml'
datasets = 'datasets'

print('Training...')

(images, labels, names, id) = ([], [], {}, 0)
width, height = (130, 100)

for subdir in os.listdir(datasets):
    names[id] = subdir
    subjectpath = os.path.join(datasets, subdir)
    for filename in os.listdir(subjectpath):
        path = os.path.join(subjectpath, filename)
        label = id
        images.append(cv2.resize(cv2.imread(path, 0), (width, height)))
        labels.append(label)
    id += 1

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Train Eigenface Recognizer
model_eigen = cv2.face.EigenFaceRecognizer_create()
model_eigen.train(images, labels)
model_eigen.save('eigenface_model.xml')

# Check if we have more than one class before training Fisherface
if len(np.unique(labels)) > 1:
    # Train Fisherface Recognizer
    model_fisher = cv2.face.FisherFaceRecognizer_create()
    model_fisher.train(images, labels)
    model_fisher.save('fisherface_model.xml')
    print('Fisherface model trained and saved successfully.')
else:
    print('Fisherface model training skipped: at least two classes are required.')

# Save the names dictionary
with open('names.pkl', 'wb') as f:
    pickle.dump(names, f)

print('Training completed successfully.')


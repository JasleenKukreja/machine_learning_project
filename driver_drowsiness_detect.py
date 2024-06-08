import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Define paths and classes
Datadirectory = "dataset"
Classes = ["Close-Eyes", "Open-Eyes"]
img_size = 224

# Create training data
training_Data = []

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb, (img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_Data()

# Shuffle data
random.shuffle(training_Data)

# Separate features and labels
X = []
y = []

for features, label in training_Data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
y = np.array(y)

# Normalize data
X = X / 255.0

# Display the distribution of data
plt.hist(y)
plt.show()
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models

# Load pre-trained MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=50, validation_split=0.2)
model.save('my_model2.h5')
import cv2
import numpy as np
import tensorflow as tf
import winsound

# Load the trained model
model = tf.keras.models.load_model('my_model2.h5')

# Set up face and eye detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

counter = 0
frequency = 2500
duration = 1500

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]
            final_image = cv2.resize(eyes_roi, (img_size, img_size))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0
            
            Predictions = model.predict(final_image)
            
            if Predictions >= 0.3:
                status = "Open Eyes"
                counter = 0
                color = (0, 255, 0)
                text = "Active"
            else:
                counter += 1
                status = "Closed Eyes"
                color = (0, 0, 255)
                text = "Sleep Alert !!!" if counter > 10 else "Drowsy"
                
                if counter > 2:
                    winsound.Beep(frequency, duration)
                    counter = 0
            
            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            break

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#this is temp
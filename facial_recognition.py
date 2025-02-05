import cv2
import numpy as np
import os

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the face recogniser (LBPH recogniser)
recogniser = cv2.face.LBPHFaceRecognizer_create()

# Path to training dataset
data_path = "lfw_funneled"  # Create a folder named 'dataset' and store images

def prepare_training_data(data_folder):
    faces = []
    labels = []
    label_dict = {}
    label_id = 0
    
    for person_name in os.listdir(data_folder):
        person_path = os.path.join(data_folder, person_name)
        if not os.path.isdir(person_path):
            continue
        
        label_dict[label_id] = person_name  # Store label name
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
            
            for (x, y, w, h) in faces_detected:
                faces.append(image[y:y+h, x:x+w])  # Extract face region
                labels.append(label_id)
        
        label_id += 1
    
    return faces, np.array(labels), label_dict

# Train the recogniser
print("Preparing training data...")
faces, labels, label_dict = prepare_training_data(data_path)
recogniser.train(faces, labels)
print("Training complete!")

# Real-time face recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        label, confidence = recogniser.predict(face)
        name = label_dict.get(label, "Unknown")
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({round(confidence, 2)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

# Model file paths
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

# Mean values for the models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Labels and middle points for age ranges
age_mid_points = [1, 5, 10, 17, 28, 40, 50, 80]
gender_list = ['Male', 'Female']

# Load networks
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 🔹 Load your image here (replace with your file path)
image_path = "person.jpg"
frame = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

for (x, y, w, h) in faces:
    pad = 15
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

    face_img = frame[y1:y2, x1:x2].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()[0]
    estimated_age = int(np.sum(np.array(age_mid_points) * age_preds))
    lower_age = max(1, estimated_age - 2)
    upper_age = min(100, estimated_age + 2)
    age_range = f"({lower_age}-{upper_age})"

    label = f"{gender}, {age_range} yrs"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show result
cv2.imshow("Gender & Age Prediction", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

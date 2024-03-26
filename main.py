import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
import cv2
import joblib
import numpy as np

model = joblib.load("svm_model.pkl")

label_map = {0: "Empty", 1: "Full", 2: "Low", 3: "Medium"}

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    flattened = resized.flatten()
    return flattened

def classify_and_print_label():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow("Traffic Signal", frame)

        preprocessed_frame = preprocess_image(frame)
        preprocessed_frame = preprocessed_frame.reshape(1, -1)

        predicted_label = model.predict(preprocessed_frame)
        predicted_class = label_map[predicted_label[0]]

        decision_scores = model.decision_function(preprocessed_frame)
        
        confidence = np.max(np.abs(decision_scores))
        
        print("Predicted label:", predicted_class, "with confidence:", round(confidence,2))

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # 'q' or 'esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

classify_and_print_label()

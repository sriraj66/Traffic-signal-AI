import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def load_data(dataset_dir):
    images = []
    labels = []
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if image_path.endswith('.jpg') or image_path.endswith('.png'):
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    images.append(image)
                    labels.append(class_name)
    return images, labels

def preprocess_images(images, target_size=(100, 100)):
    processed_images = []
    for image in images:
        image = cv2.resize(image, target_size)
        processed_images.append(image.flatten())  # Flatten the image
    return processed_images

def train_classifier(images, labels):
    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    # Convert images to numpy array
    images_array = np.array(images)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images_array, numeric_labels, test_size=0.2, random_state=42)

    # Initialize and train the classifier
    classifier = SVC(kernel='linear', C=1)
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    accuracy = classifier.score(X_test, y_test)
    print("Classifier accuracy:", accuracy)

    return classifier

def save_model(model, model_file):
    joblib.dump(model, model_file)
    print("Model saved successfully as", model_file)

def main():
    dataset_dir = "dataset"
    model_file = "svm_model.pkl"
    images, labels = load_data(dataset_dir)
    processed_images = preprocess_images(images)
    model = train_classifier(processed_images, labels)
    save_model(model, model_file)

if __name__ == "__main__":
    main()

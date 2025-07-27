import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import random

# 1. Load Pretrained ResNet50
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = resnet.predict(img, verbose=0)
    return features.flatten()

# 2. Load Images (max 1000 per class)
def load_dataset(data_dir, max_images_per_class=500):
    X, y = [], []
    class_names = os.listdir(data_dir)
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        image_paths = os.listdir(class_dir)
        random.shuffle(image_paths)
        image_paths = image_paths[:max_images_per_class]  # Limit to 1000
        for image_name in tqdm(image_paths, desc=f"Processing {class_name}"):
            image_path = os.path.join(class_dir, image_name)
            try:
                features = extract_features(image_path)
                X.append(features)
                y.append(class_name)
            except:
                print(f"[ERROR] Failed on {image_path}")
    return np.array(X), np.array(y)

# 3. Prepare Data
data_dir = r"C:\Users\india\OneDrive - Chandigarh University\Desktop\New folder\dataset"  # Change this
X, y = load_dataset(data_dir, max_images_per_class=500)
print(f"Features shape: {X.shape}, Labels count: {len(y)}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 4. Train Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

# 5. Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

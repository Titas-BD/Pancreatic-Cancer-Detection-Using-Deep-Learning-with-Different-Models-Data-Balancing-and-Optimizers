import os
import random
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import subprocess
import time

# SETTINGS
IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30
LIMIT = 18000
SEED = 42

# Dataset Path
dataset_path = r"E:\Work folder\AIUB\Reasearch Folder\Thesis_G42\PancreasProject\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"

-
# HELPER FUNCTIONS
def load_dicom_paths_and_labels(root_path):
    class_folders = [os.path.join(root_path, cls) for cls in os.listdir(root_path)
                     if os.path.isdir(os.path.join(root_path, cls))]
    image_paths, labels = [], []
    for idx, folder in enumerate(class_folders):
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.dcm'):
                    image_paths.append(os.path.join(root, file))
                    labels.append(idx)
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    combined = combined[:LIMIT]
    return zip(*combined)

def read_dicom_file(path):
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        assert img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        return img
    except Exception as e:
        print(f"Skipped file {path}, error: {e}")
        return None

def dicom_generator(paths, labels, batch_size):
    while True:
        x_batch, y_batch = [], []
        zipped = list(zip(paths, labels))
        random.shuffle(zipped)
        for path, label in zipped:
            img = read_dicom_file(path)
            if img is not None:
                x_batch.append(img)
                y_batch.append(label)
                if len(x_batch) == batch_size:
                    yield (
                        np.array(x_batch),
                        tf.keras.utils.to_categorical(np.array(y_batch), num_classes=NUM_CLASSES)
                    )
                    x_batch, y_batch = [], []
        if x_batch:
            yield (
                np.array(x_batch),
                tf.keras.utils.to_categorical(np.array(y_batch), num_classes=NUM_CLASSES)
            )


# LOAD DATA
all_paths, all_labels = load_dicom_paths_and_labels(dataset_path)
all_paths, all_labels = list(all_paths), list(all_labels)
NUM_CLASSES = len(set(all_labels))
print(f"Detected {NUM_CLASSES} unique classes: {set(all_labels)}")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, random_state=SEED
)

train_gen = dicom_generator(train_paths, train_labels, BATCH_SIZE)
val_gen = dicom_generator(val_paths, val_labels, BATCH_SIZE)

steps_per_epoch = len(train_paths) // BATCH_SIZE
validation_steps = len(val_paths) // BATCH_SIZE


# BUILD MODEL
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output_tensor = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])


# PREPARE VALIDATION DATA
val_images, val_labels_clean = [], []
for path, label in zip(val_paths, val_labels):
    img = read_dicom_file(path)
    if img is not None:
        val_images.append(img)
        val_labels_clean.append(label)
val_images = np.array(val_images)
val_labels_cat = tf.keras.utils.to_categorical(val_labels_clean, num_classes=NUM_CLASSES)


# GPU LOGGING FUNCTION
def log_gpu_usage(interval=5):
    """Print GPU memory and utilization every 'interval' seconds."""
    try:
        while True:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(result.stdout)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped GPU logging.")


# TRAIN MODEL ON GPU
print("Training on GPU...")
with tf.device('/GPU:0'):
    # Optional: run GPU logger in a separate thread if needed
    # import threading
    # t = threading.Thread(target=log_gpu_usage, args=(10,), daemon=True)
    # t.start()

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(val_images, val_labels_cat),
        validation_steps=validation_steps,
        epochs=EPOCHS,
        verbose=1
    )


# EVALUATION
print("\n")
print("STARTING EVALUATION")
print("\n")

if len(val_images) == 0:
    print("No validation images loaded. Cannot evaluate.")
else:
    y_pred = model.predict(val_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.array(val_labels_clean)

    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='macro')
    recall = recall_score(y_true, y_pred_classes, average='macro')
    f1 = f1_score(y_true, y_pred_classes, average='macro')

    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\n Classification Report:")
    print(classification_report(y_true, y_pred_classes))

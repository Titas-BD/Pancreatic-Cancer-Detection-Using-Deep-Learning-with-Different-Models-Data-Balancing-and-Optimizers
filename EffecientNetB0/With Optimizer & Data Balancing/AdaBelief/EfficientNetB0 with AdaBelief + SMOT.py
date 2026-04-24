import os
import random
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from imblearn.over_sampling import RandomOverSampler
from adabelief_tf import AdaBelief


# SETTINGS
IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30
LIMIT = 18000
SEED = 42

# Dataset Path
DATASET_PATH = r"E:\Work folder\AIUB\Reasearch Folder\Thesis_G42\PancreasProject\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# GPU Check
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU detected: {gpus[0].name}" if gpus else "No GPU found, running on CPU.")


# HELPER FUNCTIONS
def load_dicom_paths_and_labels(root_path):
    """
    Collect .dcm file paths and integer labels from subfolders under root_path.
    Each subfolder is treated as one class.
    """
    class_folders = [
        os.path.join(root_path, cls)
        for cls in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, cls))
    ]

    image_paths = []
    labels = []

    for idx, folder in enumerate(class_folders):
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".dcm"):
                    image_paths.append(os.path.join(root, file))
                    labels.append(idx)

    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    combined = combined[:LIMIT]

    image_paths, labels = zip(*combined)
    return list(image_paths), list(labels)


def read_dicom_file(path):
    """Read and preprocess a single DICOM file to a 128×128×3 float32 image."""
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        return img

    except Exception as e:
        print(f"Skipped file {path}, error: {e}")
        return None


def dicom_generator(paths, labels, batch_size, num_classes, datagen=None):
    """
    Yield batches of (optionally augmented) DICOM images and one-hot labels.
    """
    while True:
        x_batch, y_batch = [], []
        zipped = list(zip(paths, labels))
        random.shuffle(zipped)

        for path, label in zipped:
            img = read_dicom_file(path)
            if img is None:
                continue

            if datagen is not None:
                img = datagen.random_transform(img)

            x_batch.append(img)
            y_batch.append(label)

            if len(x_batch) == batch_size:
                yield (
                    np.array(x_batch, dtype=np.float32),
                    tf.keras.utils.to_categorical(
                        np.array(y_batch), num_classes=num_classes
                    ),
                )
                x_batch, y_batch = [], []

        if x_batch:
            yield (
                np.array(x_batch, dtype=np.float32),
                tf.keras.utils.to_categorical(
                    np.array(y_batch), num_classes=num_classes
                ),
            )


# LOAD DATA
all_paths, all_labels = load_dicom_paths_and_labels(DATASET_PATH)

NUM_CLASSES = len(set(all_labels))
print(f"Detected {NUM_CLASSES} unique classes.")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=0.2,
    random_state=SEED,
    stratify=all_labels,
)


# ==========================
# SMOT: RandomOverSampler on indices
# ==========================
train_labels_array = np.array(train_labels)
idx = np.arange(len(train_labels_array)).reshape(-1, 1)

ros = RandomOverSampler(random_state=SEED)
idx_resampled, y_resampled = ros.fit_resample(idx, train_labels_array)

idx_resampled = idx_resampled.ravel()
train_paths_resampled = [train_paths[i] for i in idx_resampled]
train_labels_resampled = list(y_resampled)

print(f"Original train size: {len(train_paths)}")
print(f"Resampled train size (SMOT): {len(train_paths_resampled)}")


# DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
)


# GENERATORS (use resampled training set)
train_gen = dicom_generator(
    train_paths_resampled,
    train_labels_resampled,
    BATCH_SIZE,
    num_classes=NUM_CLASSES,
    datagen=datagen,
)

val_gen = dicom_generator(
    val_paths,
    val_labels,
    BATCH_SIZE,
    num_classes=NUM_CLASSES,
    datagen=None,
)

steps_per_epoch = len(train_paths_resampled) // BATCH_SIZE


# PREPARE VALIDATION DATA (for final evaluation)
val_images = []
val_labels_clean = []

for path, label in zip(val_paths, val_labels):
    img = read_dicom_file(path)
    if img is not None:
        val_images.append(img)
        val_labels_clean.append(label)

val_images = np.array(val_images, dtype=np.float32)
val_labels_clean = np.array(val_labels_clean, dtype=np.int32)
val_labels_cat = tf.keras.utils.to_categorical(val_labels_clean, num_classes=NUM_CLASSES)


# MODEL: EfficientNetB0 backbone + custom classifier
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=input_tensor,
)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
output_tensor = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

optimizer = AdaBelief(learning_rate=0.001, rectify=False)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# CALLBACKS
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
]


# TRAIN (AdaBelief + SMOT, no class_weight)
model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=(val_images, val_labels_cat),
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks,
)


# EVALUATE
y_pred = model.predict(val_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_labels_clean

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average="macro")
recall = recall_score(y_true, y_pred_classes, average="macro")
f1 = f1_score(y_true, y_pred_classes, average="macro")

print("\nResults (EfficientNetB0 + AdaBelief + SMOT):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

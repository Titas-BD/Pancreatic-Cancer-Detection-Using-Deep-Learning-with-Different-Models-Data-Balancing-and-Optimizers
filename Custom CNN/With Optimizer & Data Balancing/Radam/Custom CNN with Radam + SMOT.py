import os
import pydicom
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import gc
from collections import Counter
import tensorflow as tf
import tensorflow_addons as tfa
from imblearn.over_sampling import RandomOverSampler

np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_dicom(dcm_path, img_size=(128, 128)):
    """Load and preprocess DICOM file."""
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        if not hasattr(dcm, "pixel_array"):
            print(f"Skipped {dcm_path}: No pixel array")
            return None

        img = dcm.pixel_array

        if len(img.shape) == 3:
            img = img[img.shape[0] // 2]

        img = img.astype(np.float32)
        if hasattr(dcm, "RescaleSlope"):
            img = img * float(dcm.ResscaleSlope) if hasattr(dcm, "ResscaleSlope") else img * float(dcm.RescaleSlope)
        if hasattr(dcm, "RescaleIntercept"):
            img = img + float(dcm.RescaleIntercept)

        img_min, img_max = np.min(img), np.max(img)
        if img_max <= img_min:
            print(f"Skipped {dcm_path}: Invalid pixel range ({img_min}, {img_max})")
            return None

        img = (img - img_min) / (img_max - img_min)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)
        img = cv2.medianBlur(img, 3)

        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(img, axis=-1)

    except Exception as e:
        print(f"Error processing {dcm_path}: {str(e)}")
        return None


def load_large_dataset(base_path, max_images=18000, img_size=(128, 128)):
    """
    loading DICOM images from subfolders under base_path.
    """
    images = []
    labels = []
    count = 0

    print(f"[INFO] Loading up to {max_images} images...")

    class_counts = Counter()
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(".dcm"):
                class_counts[os.path.basename(root)] += 1

    valid_classes = {cls for cls, c in class_counts.items() if c >= 2}
    if not valid_classes:
        raise ValueError("No classes with sufficient samples (minimum 2 per class).")

    print(f"Valid classes: {len(valid_classes)}")
    for cls, c in class_counts.items():
        print(f"Class {cls}: {c} samples")

    for root, _, files in os.walk(base_path):
        class_name = os.path.basename(root)
        if class_name not in valid_classes:
            continue

        for file in files:
            if count >= max_images:
                break

            if file.lower().endswith(".dcm"):
                img = load_and_preprocess_dicom(os.path.join(root, file), img_size)
                if img is not None:
                    images.append(img)
                    labels.append(class_name)
                    count += 1

                    if count % 500 == 0:
                        print(f"Loaded {count} images")
                        gc.collect()

    print(f"\nFinished loading. Total: {count} images")
    return np.array(images, dtype=np.float16), np.array(labels)


def build_model(input_shape, num_classes):
    """trained with RAdam."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    base_path = r"D:\NBIA Pancreatic images\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
    img_size = (128, 128)
    max_images = 18000
    test_size = 0.2
    val_size = 0.25
    random_state = 42
    batch_size = 64

    try:
        images, labels = load_large_dataset(base_path, max_images, img_size)

        unique_labels = np.unique(labels)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        y_int = np.array([label_to_idx[label] for label in labels])

        n_samples, h, w, c = images.shape
        X_flat = images.reshape(n_samples, -1)

        ros = RandomOverSampler(random_state=random_state)
        X_res_flat, y_res_int = ros.fit_resample(X_flat, y_int)

        X_res = X_res_flat.reshape(-1, img_size[0], img_size[1], 1)

        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X_res,
            y_res_int,
            test_size=test_size,
            random_state=random_state,
            stratify=y_res_int
        )

        X_train, X_val, y_train_int, y_val_int = train_test_split(
            X_train,
            y_train_int,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_int
        )

        del images, labels, X_flat, X_res_flat
        gc.collect()

        num_classes = len(unique_labels)

        y_train = to_categorical(y_train_int, num_classes=num_classes)
        y_val = to_categorical(y_val_int, num_classes=num_classes)
        y_test = to_categorical(y_test_int, num_classes=num_classes)

        model = build_model((img_size[0], img_size[1], 1), num_classes)
        model.summary()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n=== Final Evaluation ===")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = y_test_int

        print("\nClassification Report:")
        print(classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=unique_labels
        ))

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        metrics = ["Precision", "Recall", "F1-Score"]
        values = [
            precision_score(y_true_classes, y_pred_classes, average="weighted"),
            recall_score(y_true_classes, y_pred_classes, average="weighted"),
            f1_score(y_true_classes, y_pred_classes, average="weighted")
        ]
        plt.bar(metrics, values)
        plt.title("Model Metrics on Test Set (RAdam + SMOT)")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
